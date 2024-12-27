# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch
from torch import nn
import torch.functional as F

from model.base import BaseModule
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion, StftDiffusion
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility, maximum_path_numpy
from model.text_encoder import TextEncoder, DurationPredictor
from model.stochastic_duration_predictor import StochasticDurationPredictor
from model.context_predictor import ContextPredictorResnet
from model.commons import rand_slice_segments, slice_segments
from model.stft_loss import MultiResolutionSTFTLoss
from model.pqmf import PQMF

from model.rel_transformer import RelTransformerEncoder
from model.transformer import MultiheadAttention, FFTBlocks
from model.conv import ConvBlocks
from model.nar_tts_modules import SyntaDurationPredictor
from model.modules import Embedding
from model.align_ops import clip_mel2token_to_multiple, build_word_mask, expand_states, mel2ph_to_mel2word
from model.seq_utils import group_hidden_by_segs
from model.align import mel2token_to_dur


ENCODERS = {
    'rel_fft': lambda hp, dict_size: RelTransformerEncoder(
        dict_size, hp['hidden_size'], hp['hidden_size'],
        hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'],
        hp['enc_ffn_kernel_size'], hp['dropout'], prenet=hp['enc_prenet'], pre_ln=hp['enc_pre_ln']),
}

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """

        :param x: [B, T]
        :return: [B, T, H]
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, :, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class GradTTSDependencyGraph(BaseModule):
    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, dec_dim, beta_min, beta_max, pe_scale, ph_dict_size, word_dict_size, synta_params):
        
        super(GradTTSDependencyGraph, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.synta_params = synta_params

        self.emb = torch.nn.Embedding(n_vocab, n_enc_channels)

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        if synta_params['use_word_encoder']:
            self.word_encoder = RelTransformerEncoder(
                word_dict_size, self.n_enc_channels, self.n_enc_channels, self.n_enc_channels, 2,
                synta_params['word_enc_layers'], synta_params['enc_ffn_kernel_size'])
        if synta_params['dur_level'] == 'word':
            if synta_params['word_encoder_type'] == 'rel_fft':
                self.ph2word_encoder = RelTransformerEncoder(
                    0, self.n_enc_channels, self.n_enc_channels, self.n_enc_channels, 2,
                    synta_params['word_enc_layers'], synta_params['enc_ffn_kernel_size'])
            if synta_params['word_encoder_type'] == 'fft':
                self.ph2word_encoder = FFTBlocks(
                    self.n_enc_channels, synta_params['word_enc_layers'], 1, num_heads=synta_params['num_heads'])
            self.sin_pos = SinusoidalPosEmb(self.n_enc_channels)
            self.enc_pos_proj = nn.Linear(2 * self.n_enc_channels, self.n_enc_channels)
            self.dec_query_proj = nn.Linear(2 * self.n_enc_channels, self.n_enc_channels)
            self.dec_res_proj = nn.Linear(2 * self.n_enc_channels, self.n_enc_channels)
            self.attn = MultiheadAttention(self.n_enc_channels, 1, encoder_decoder_attention=True, bias=False)
            self.attn.enable_torch_version = False
            if synta_params['text_encoder_postnet']:
                self.text_encoder_postnet = ConvBlocks(
                    self.n_enc_channels, self.n_enc_channels, [1] * 3, 5, layers_in_block=2)
        else:
            self.sin_pos = SinusoidalPosEmb(self.n_enc_channels)

        self.encoder = ENCODERS[synta_params['encoder_type']](synta_params, ph_dict_size)

        self.decoder = Diffusion(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)

        self.proj_w = SyntaDurationPredictor(
            self.n_enc_channels,
            n_chans=synta_params['predictor_hidden'],
            n_layers=synta_params['dur_predictor_layers'],
            dropout_rate=synta_params['predictor_dropout'],
            kernel_size=synta_params['dur_predictor_kernel'])
        

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, graph_lst, etypes_lst, temperature=1.0, stoc=False, spk=None, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """

        ret = {}
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        x = self.emb(x) * math.sqrt(self.n_enc_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x_dp = torch.detach(x)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, tgt_nonpadding = self.run_text_encoder(
            txt_tokens, word_tokens, ph2word, x_lengths.max(), mel2word, mel2ph, ret, graph_lst=graph_lst, etypes_lst=etypes_lst)

        logw = self.proj_w(x_dp, x_mask, reverse=True)

        mu_x = mu_x * tgt_nonpadding

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics

        # decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)

        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        # return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    
        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length], ret

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def run_text_encoder(self, txt_tokens, word_tokens, ph2word, word_len, mel2word, mel2ph, ret, graph_lst, etypes_lst):
        word2word = torch.arange(word_len)[None, :].to(ph2word.device) + 1  # [B, T_mel, T_word]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        if self.synta_params['use_word_encoder']:
            ph_encoder_out = ph_encoder_out + expand_states(word_encoder_out, ph2word)
        
        dur_input = ph_encoder_out * src_nonpadding
        if self.synta_params['dur_level'] == 'word':
            word_encoder_out = 0
            h_ph_gb_word = group_hidden_by_segs(ph_encoder_out, ph2word, word_len)[0]
            word_encoder_out = word_encoder_out + self.ph2word_encoder(h_ph_gb_word)
            if self.synta_params['use_word_encoder']:
                word_encoder_out = word_encoder_out + self.word_encoder(word_tokens)
            mel2word = self.forward_dur(dur_input, mel2word, ret, ph2word=ph2word, word_len=word_len, graph_lst=graph_lst, etypes_lst=etypes_lst)
            mel2word = clip_mel2token_to_multiple(mel2word, self.synta_params['frames_multiple'])
            ret['mel2word'] = mel2word
            tgt_nonpadding = (mel2word > 0).float()[:, :, None]
            enc_pos = self.get_pos_embed(word2word, ph2word)  # [B, T_ph, H]
            dec_pos = self.get_pos_embed(word2word, mel2word)  # [B, T_mel, H]
            dec_word_mask = build_word_mask(mel2word, ph2word)  # [B, T_mel, T_ph]
            x, weight = self.attention(ph_encoder_out, enc_pos, word_encoder_out, dec_pos, mel2word, dec_word_mask)
            if self.synta_params['add_word_pos']:
                x = x + self.word_pos_proj(dec_pos)
            ret['attn'] = weight
        else:
            mel2ph = self.forward_dur(dur_input, mel2ph, ret)
            mel2ph = clip_mel2token_to_multiple(mel2ph, self.synta_params['frames_multiple'])
            mel2word = mel2ph_to_mel2word(mel2ph, ph2word)
            x = expand_states(ph_encoder_out, mel2ph)
            if self.synta_params['add_word_pos']:
                dec_pos = self.get_pos_embed(word2word, mel2word)  # [B, T_mel, H]
                x = x + self.word_pos_proj(dec_pos)
            tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        if self.synta_params['use_word_encoder']:
            x = x + expand_states(word_encoder_out, mel2word)
        return x, tgt_nonpadding

    def attention(self, ph_encoder_out, enc_pos, word_encoder_out, dec_pos, mel2word, dec_word_mask):
        ph_kv = self.enc_pos_proj(torch.cat([ph_encoder_out, enc_pos], -1))
        word_enc_out_expend = expand_states(word_encoder_out, mel2word)
        word_enc_out_expend = torch.cat([word_enc_out_expend, dec_pos], -1)
        if self.synta_params['text_encoder_postnet']:
            word_enc_out_expend = self.dec_res_proj(word_enc_out_expend)
            word_enc_out_expend = self.text_encoder_postnet(word_enc_out_expend)
            dec_q = x_res = word_enc_out_expend
        else:
            dec_q = self.dec_query_proj(word_enc_out_expend)
            x_res = self.dec_res_proj(word_enc_out_expend)
        ph_kv, dec_q = ph_kv.transpose(0, 1), dec_q.transpose(0, 1)
        x, (weight, _) = self.attn(dec_q, ph_kv, ph_kv, attn_mask=(1 - dec_word_mask) * -1e9)
        x = x.transpose(0, 1)
        x = x + x_res
        return x, weight

    def run_decoder(self, x, tgt_nonpadding, ret, infer, tgt_mels=None, global_step=0,
                    mel2word=None, ph2word=None, graph_lst=None, etypes_lst=None):
        if not self.synta_params['use_fvae']:
            x = self.decoder(x)
            x = self.mel_out(x)
            ret['kl'] = 0
            return x * tgt_nonpadding
        else:
            # x is the phoneme encoding
            x = x.transpose(1, 2)  # [B, H, T]
            tgt_nonpadding_BHT = tgt_nonpadding.transpose(1, 2)  # [B, H, T]
            if infer:
                z = self.fvae(cond=x, infer=True, mel2word=mel2word, ph2word=ph2word, graph_lst=graph_lst, etypes_lst=etypes_lst)
            else:
                tgt_mels = tgt_mels.transpose(1, 2)  # [B, 80, T]
                z, ret['kl'], ret['z_p'], ret['m_q'], ret['logs_q'] = self.fvae(
                    tgt_mels, tgt_nonpadding_BHT, cond=x, mel2word=mel2word, ph2word=ph2word, graph_lst=graph_lst, etypes_lst=etypes_lst)
                if global_step < self.synta_params['posterior_start_steps']:
                    z = torch.randn_like(z)
            x_recon = self.fvae.decoder(z, nonpadding=tgt_nonpadding_BHT, cond=x).transpose(1, 2)
            ret['pre_mel_out'] = x_recon
            return x_recon

    def forward_dur(self, dur_input, mel2word, ret, **kwargs):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        word_len = kwargs['word_len']
        ph2word = kwargs['ph2word']
        graph_lst = kwargs['graph_lst']
        etypes_lst = kwargs['etypes_lst']
        src_padding = dur_input.data.abs().sum(-1) == 0
        dur_input = dur_input.detach() + self.synta_params['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding, ph2word, graph_lst, etypes_lst)

        B, T_ph = ph2word.shape
        dur = torch.zeros([B, word_len.max() + 1]).to(ph2word.device).scatter_add(1, ph2word, dur)
        dur = dur[:, 1:]
        ret['dur'] = dur
        if mel2word is None:
            mel2word = self.length_regulator(dur).detach()
        return mel2word

    def get_pos_embed(self, word2word, x2word):
        x_pos = build_word_mask(word2word, x2word).float()  # [B, T_word, T_ph]
        x_pos = (x_pos.cumsum(-1) / x_pos.sum(-1).clamp(min=1)[..., None] * x_pos).sum(1)
        x_pos = self.sin_pos(x_pos.float())  # [B, T_ph, H]
        return x_pos

    def store_inverse_all(self):
        def remove_weight_norm(m):
            try:
                if hasattr(m, 'store_inverse'):
                    m.store_inverse()
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)

    def add_dur_loss(self, dur_pred, mel2token, word_len):
        T = word_len.max()
        dur_gt = mel2token_to_dur(mel2token, T).float()
        nonpadding = (torch.arange(T).to(dur_pred.device)[None, :] < word_len[:, None]).float()
        dur_pred = dur_pred * nonpadding
        dur_gt = dur_gt * nonpadding
        wdur = F.l1_loss((dur_pred + 1).log(), (dur_gt + 1).log(), reduction='none')
        wdur = (wdur * nonpadding).sum() / nonpadding.sum()
        if self.synta_params['lambda_word_dur'] > 0:
            return wdur * self.synta_params['lambda_word_dur']
        if self.synta_params['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.l1_loss(sent_dur_p, sent_dur_g, reduction='mean')

            return sdur_loss.mean() * self.synta_params['lambda_sent_dur']


    def compute_loss(self, x, x_lengths, y, y_lengths, dur_pred, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, x_dp, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad(): 
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = maximum_path_numpy(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        w = attn.sum(2)
        logw = self.proj_w(x_dp, x_mask, w)
        l_lengths = logw / torch.sum(x_mask)

        dur_loss = self.add_dur_loss(dur_pred, mel2token, word_len)
        # dur_loss = torch.sum(l_lengths) - torch.tensor([1])[0]

        # _, ids_str = rand_slice_segments(y)
        # sliced_y = slice_segments(y, ids_str*hop_length, segment_size)
        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            
            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of score-based decoder
        # diff_loss, xt, stft_loss = self.decoder.compute_loss(y, y_mask, mu_y, spk)

        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        # return dur_loss, prior_loss, diff_loss
    
        return dur_loss, prior_loss, diff_loss


class GradTTSStft(BaseModule):
    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, dec_dim, beta_min, beta_max, pe_scale, stft_params):
        
        super(GradTTSStft, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size)

        # self.decoder = Diffusion(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)
        self.decoder = StftDiffusion(n_feats, dec_dim, **stft_params)

        self.proj_w = StochasticDurationPredictor(n_enc_channels, 192, enc_kernel, enc_dropout)
        

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, x_dp, x_mask = self.encoder(x, x_lengths, spk)

        logw = self.proj_w(x_dp, x_mask, reverse=True)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics

        # decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)

        decoder_outputs, y_g_hat, y_mb_hat = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        # return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    
        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length], y_g_hat, y_mb_hat


    def compute_loss(self, x, x_lengths, y, y_lengths, y_hat_mb, fft_sizes, hop_sizes, win_lengths, segment_size, hop_length, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, x_dp, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad(): 
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = maximum_path_numpy(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        w = attn.sum(2)
        logw = self.proj_w(x_dp, x_mask, w)
        l_lengths = logw / torch.sum(x_mask)
        dur_loss = torch.sum(l_lengths) - torch.tensor([1])[0]

        # _, ids_str = rand_slice_segments(y)
        # sliced_y = slice_segments(y, ids_str*hop_length, segment_size)
        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            
            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of score-based decoder
        # diff_loss, xt, stft_loss = self.decoder.compute_loss(y, y_mask, mu_y, spk)

        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        reshaped_y = y.view(16, 1, -1)
        pqmf = PQMF(reshaped_y.device)
        y_mb = pqmf.analysis(reshaped_y)

        stft_loss = self.subband_stft_loss(y_mb, y_hat_mb, fft_sizes, hop_sizes, win_lengths)
        
        # return dur_loss, prior_loss, diff_loss
    
        return dur_loss, prior_loss, diff_loss, stft_loss
    
    def subband_stft_loss(self, y_mb, y_hat_mb, fft_sizes, hop_sizes, win_lengths):
        sub_stft_loss = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths)
        y_mb =  y_mb.view(-1, y_mb.size(2))
        y_hat_mb = y_hat_mb.view(-1, y_hat_mb.size(2))
        sub_sc_loss, sub_mag_loss = sub_stft_loss(y_hat_mb[:, :y_mb.size(-1)], y_mb)
        
        return sub_sc_loss+sub_mag_loss


    
class GradTTSSDP(BaseModule):
    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, dec_dim, beta_min, beta_max, pe_scale):
        super(GradTTSSDP, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size)

        self.decoder = Diffusion(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)

        self.proj_w = StochasticDurationPredictor(n_enc_channels, 192, enc_kernel, enc_dropout)
        

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, x_dp, x_mask = self.encoder(x, x_lengths, spk)

        logw = self.proj_w(x_dp, x_mask, reverse=True)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics

        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]
    


    def compute_loss(self, x, x_lengths, y, y_lengths, y_mb, y_hat_mb, fft_sizes, hop_sizes, win_lengths, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, x_dp, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad(): 
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = maximum_path_numpy(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        w = attn.sum(2)
        logw = self.proj_w(x_dp, x_mask, w)
        l_lengths = logw / torch.sum(x_mask)
        dur_loss = torch.sum(l_lengths) - torch.tensor([1])[0]

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            
            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)

        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        return dur_loss, prior_loss, diff_loss
    

class GradTTS(BaseModule):
    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, dec_dim, beta_min, beta_max, pe_scale):
        super(GradTTS, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size)
        self.decoder = Diffusion(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)

        self.proj_w = DurationPredictor(n_enc_channels, 256, enc_kernel, enc_dropout)

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, x_dp, x_mask = self.encoder(x, x_lengths, spk)

        logw = self.proj_w(x_dp, x_mask)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, x_dp, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad(): 
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = maximum_path_numpy(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        
        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw = self.proj_w(x_dp, x_mask)
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            
            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        return dur_loss, prior_loss, diff_loss


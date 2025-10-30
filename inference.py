# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS, GradTTSSDP, GradTTSStft

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from librosa import pyin

import sys
sys.path.append('./hifi-gan/')
from hifi_gan.env import AttrDict
from hifi_gan.models import Generator as HiFiGAN
# from model.stft_loss import MultiResolutionSTFTLoss
from meldataset import mel_spectrogram
import epitran
from text.bn_phonemiser import bangla_text_normalize, replace_number_with_text

# import IPython.display as ipd

HIFIGAN_CONFIG = '/mnt/Stuff/TTS/speech_synthesis_in_bangla-master/checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = '/mnt/Stuff/TTS/speech_synthesis_in_bangla-master/checkpts/hifigan.pt'


ipa_dict = {'ɖ̤': 0,
                'kʰ': 1,
                'n': 2,
                'm': 3,
                'ɡ̤': 4,
                'ʃ': 5,
                'b̤': 6,
                'd̪̤': 7,
                'ŋ': 8,
                'pʰ': 9,
                'ɽ̤': 10,
                'k': 11,
                'a': 12,
                'b': 13,
                'r': 14,
                'ʈʰ': 15,
                'V': 16,
                'ɖ': 17,
                't̪ʰ': 18,
                'p': 19,
                'z': 20,
                'e': 21,
                't̪': 22,
                'u': 23,
                'j': 24,
                'd̪': 25,
                'o': 26,
                'i': 27,
                'd͡z': 28,
                's': 29,
                'd͡z̤': 30,
                'ঃ': 31,
                'h': 32,
                '্': 33,
                'ɽ': 34,
                '̃': 35,
                'l': 36,
                'ʈ': 37,
                'ɡ': 38,
                'ɔ': 39,
                ' ': 40}


def get_ipa_tokens_from_text(tokens):
    ipa_tokens = []
    for tok in tokens:
        if tok in ipa_dict.keys():
            ipa_tokens.append(ipa_dict[tok])

    return ipa_tokens

def get_text(text, add_blank=True):
    bn_phonemizer = epitran.Epitran('ben-Beng-east')
    text_norm = replace_number_with_text(text)
    text_norm = bangla_text_normalize(text_norm)
    text_tokens = bn_phonemizer.trans_list(text_norm)
    ipa_tokens = get_ipa_tokens_from_text(text_tokens)
    # text_norm = torch.IntTensor(ipa_tokens)
    return ipa_tokens

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    # parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    # parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    # parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    # args = parser.parse_args()



    # if not isinstance(args.speaker_id, type(None)):
    #     assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
    #     spk = torch.LongTensor([args.speaker_id]).cuda()
    # else:
    spk = None
    device = 'cuda'

    
    print('Initializing model...')
    """ GradTTS"""
    # generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
    #                     params.n_enc_channels, params.filter_channels,
    #                     params.filter_channels_dp, params.n_heads, params.n_enc_layers,
    #                     params.enc_kernel, params.enc_dropout, params.window_size,
    #                     params.n_feats, params.dec_dim, params.beta_min, params.beta_max,
    #                     pe_scale=1000)
    # generator.load_state_dict(torch.load('/mnt/Stuff/TTS/speech_synthesis_in_bangla-master/models/gradtts.pt', map_location=lambda loc, storage: loc))

    """ GradTTSSDP """
    # generator = GradTTSSDP(len(symbols)+1, params.n_spks, params.spk_emb_dim,
    #                     params.n_enc_channels, params.filter_channels,
    #                     params.filter_channels_dp, params.n_heads, params.n_enc_layers,
    #                     params.enc_kernel, params.enc_dropout, params.window_size,
    #                     params.n_feats, params.dec_dim, params.beta_min, params.beta_max,
    #                     pe_scale=1000)  # pe_scale=1 for `grad-tts-old.pt`
    # generator.load_state_dict(torch.load('/mnt/Stuff/TTS/speech_synthesis_in_bangla-master/models/gradtts_sdp.pt', map_location=lambda loc, storage: loc))


    """GradTTSStft"""
    generator = GradTTSStft(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max,
                        pe_scale=1000, stft_params=params.stft_config).to(device)
    generator.load_state_dict(torch.load('/mnt/Stuff/TTS/speech_synthesis_in_bangla-master/models/gradtts_stft.pt', map_location=lambda loc, storage: loc))

    generator = generator.to(device)

    _ = generator.eval()
    print(f'Number of parameters: {generator.nparams}')
    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h).to(device)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.eval()
    vocoder.remove_weight_norm()

    latent_min = 50
    latent_max = 800

    texts = ["আমাদের আচার-ব্যবহার শিষ্টাচার পালা-পার্বণ লোকসঙ্গীত চারুশিল্প গ্রামীণ কারুকার্য এসব অবশ্যই আমাদের সংস্কৃতির অঙ্গ।", 
             "কিন্তু এসব দেদারসে বিক্রি হচ্ছে, ইউরোপীয় দেশের মতো এখানেও ব্যাংক থেকে সিকিউরিটির ভিত্তিতে এসব কেনার জন্য ধারের ব্যবস্থা হয়েছে।",
             "আর কেউ জানে না। শুধু আমরা তিনজন। কাল যে ঈদ। আপাকে তো আর গুণ্ডাপুলিশ দিনের বেলায় আসতে দেবে না, আপা তাই রাতে আসবে আমাদের দেখতে।",
             "টিক্কা খান বহাল তবিয়তে বেঁচে আছে, একের পর এক মার্শাল ল'র বাঁধন-বেড়ি প্রচার করে যাচ্ছে।"]
    
    ids = [2308268038, 2309031328, 2311015182, 2311231870]

    prefix = "gradtts"

    for i, text in enumerate(texts):
        print(text)
        x = torch.LongTensor(get_text(text)).to(device)[None]
        x_lengths = torch.LongTensor([x.shape[-1]]).to(device)

        t = dt.datetime.now()

        """For GradTTSStft"""
        y_enc, y_dec, attn, y_g_hat, y_mb_hat = generator.forward(x, x_lengths, n_timesteps=100, temperature=1.3,
                                            stoc=False, spk=None if params.n_spks==1 else torch.LongTensor([15]),
                                            length_scale=0.91)
        
        """For other models"""
        # y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=100, temperature=1.3,
        #                                     stoc=False, spk=None if params.n_spks==1 else torch.LongTensor([15]),
        #                                     length_scale=0.91)
        
        y_dec = y_dec

        audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).detach().numpy() * 32768).astype(np.int16)
        write(f'{prefix}_sample_{i+1}.wav', 22050, audio)

    # rtf_text = "যেহেতু হোটেল রিজার্ভেশন আপনার এনআইডি দেওয়া সেক্ষেত্রে পুলিশ কিন্তু আপনাকেই গ্রেফতার করবে এর ফলে শ্রোতা পর পর যে শব্দগুলো উচ্চারণ করছে তা যেন ভিন্ন ভিন্ন পথ অতিক্রম করে কিছুটা কালিক ব্যবধান নিয়ে শ্রোতার কানে এসে ওপরিপাতন ঘটাচ্ছে তারা বাদে তিনি নিজেই দেখে, উপভোগ করে তৃপ্তি থাকেনা আমাদের জন্য এঁকে যান একের পর এক ক্যানভাসে জীবন্ত চিত্র, অমর করে রাখেন তার স্বপ্নের পদ্মপুকুরকে"
    
    # rtf_values = []
    # with torch.no_grad():
    #     txt = ""
    #     for i in range(len(rtf_text)):
    #         txt += rtf_text[i]
    #         print(f'Synthesizing text...', end=' ')
    #         x = torch.LongTensor(get_text(txt)).cuda()[None]
    #         x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                    
    #         t = dt.datetime.now()
    #         # y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=100, temperature=1.5,
    #         #                                             stoc=False, spk=spk, length_scale=0.91)
    #         """For GradTTSStft"""
    #         y_enc, y_dec, attn, y_g_hat, y_mb_hat = generator.forward(x, x_lengths, n_timesteps=100, temperature=1.3,
    #                                             stoc=False, spk=None if params.n_spks==1 else torch.LongTensor([15]),
    #                                             length_scale=0.91)
    #         t = (dt.datetime.now() - t).total_seconds()
    #         print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')
    #         rtf = t * 22050 / (y_dec.shape[-1] * 256)
    #         rtf_values.append(rtf)
    
    #         # audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.float32)
    #         # len_ = len(audio) / 22050
    #         # write(f'./out/sample.wav', 22050, audio)
    
    # with open('grad_tts_stft_rtf.txt', 'w') as f:
    #     f.write(json.dumps(rtf_values))


    # print('Done. Check out `out` folder for samples.')

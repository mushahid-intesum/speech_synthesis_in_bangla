import math

import torch

from model.base import BaseModule
from model.utils import sequence_mask, convert_pad_shape
from torch import nn
import numpy as np


# class LayerNorm(BaseModule):
#     def __init__(self, channels, eps=1e-4):
#         super(LayerNorm, self).__init__()
#         self.channels = channels
#         self.eps = eps

#         self.gamma = torch.nn.Parameter(torch.ones(channels))
#         self.beta = torch.nn.Parameter(torch.zeros(channels))

#     def forward(self, x):
#         n_dims = len(x.shape)
#         mean = torch.mean(x, 1, keepdim=True)
#         variance = torch.mean((x - mean)**2, 1, keepdim=True)

#         x = (x - mean) * torch.rsqrt(variance + self.eps)

#         shape = [1, -1] + [1] * (n_dims - 2)
#         x = x * self.gamma.view(*shape) + self.beta.view(*shape)
#         return x


class MultiHeadAttention(BaseModule):
    def __init__(self, channels, out_channels, n_heads, window_size=None, 
                 heads_share=True, p_dropout=0.0, proximal_bias=False, 
                 proximal_init=False):
        super(MultiHeadAttention, self).__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = torch.nn.Parameter(torch.randn(n_heads_rel, 
                             window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = torch.nn.Parameter(torch.randn(n_heads_rel, 
                             window_size * 2 + 1, self.k_channels) * rel_stddev)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)
        
    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        
        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, 
                                                                    dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, 
                                                                value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = torch.nn.functional.pad(
                            relative_embeddings, convert_pad_shape([[0, 0], 
                            [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:,
                                   slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(x, convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = torch.nn.functional.pad(x_flat, convert_pad_shape([[0,0],[0,0],[0,length-1]]))
        x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
        x_flat = x.view([batch, heads, length**2 + length*(length - 1)])
        x_flat = torch.nn.functional.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
        return x_final

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(BaseModule):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, 
                 p_dropout=0.0):
        super(FFN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, 
                                      padding=kernel_size//2)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size, 
                                      padding=kernel_size//2)
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.conv_1(x)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x)
        return x
    
class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

    
class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ContextPredictorTransformer(BaseModule):
    def __init__(self, q, hidden_channels, filter_channels, n_heads, n_layers, 
                 kernel_size=1, p_dropout=0.0, window_size=None):
        super(ContextPredictorTransformer, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.q = q

        self.drop = torch.nn.Dropout(p_dropout)

        self.input_module = torch.nn.ModuleList()
        self.output_module = torch.nn.ModuleList()


        self.input_attn = MultiHeadAttention(hidden_channels, hidden_channels,
                                    n_heads, window_size=window_size, p_dropout=p_dropout)
        self.input_layernorm1 = torch.nn.LayerNorm(hidden_channels)
        self.input_ffn = FFN(hidden_channels, hidden_channels,
                                       filter_channels, kernel_size, p_dropout=p_dropout)
        self.input_layernorm2 = torch.nn.LayerNorm(hidden_channels)

        for i in range(q):

            self.output_attn = MultiHeadAttention(hidden_channels, hidden_channels,
                                        n_heads, window_size=window_size, p_dropout=p_dropout)
            self.output_layernorm1 = torch.nn.LayerNorm(hidden_channels)
            self.output_ffn = FFN(hidden_channels, hidden_channels,
                                        filter_channels, kernel_size, p_dropout=p_dropout)
            self.output_layernorm2 = torch.nn.LayerNorm(hidden_channels)
            block1 = nn.ModuleList([self.output_attn, self.output_layernorm1, self.output_ffn, self.input_layernorm2])
            block2 = nn.ModuleList([self.output_attn, self.output_layernorm1, self.output_ffn, self.input_layernorm2])

            self.output_blocks = nn.ModuleList([block1, block2])

    def forward(self, x):
        y = self.input_attn(x, x)
        y = self.drop(y)
        x = self.input_layernorm1(x + y)
        y = self.input_ffn(x)
        y = self.drop(y)
        x_prime = self.input_layernorm2(x+y)

        outputs = []

        for block in self.output_blocks:
            block1 = block[0]
            block2 = block[1]
            y = block1[0](x_prime, x_prime)
            y = self.drop(y)
            x = block1[1](x_prime, y)
            y = block1[2](x)
            y = self.drop(y)
            outputs.append(block1[3](x+y))

            y = block2[0](x_prime, x_prime)
            y = self.drop(y)
            x = block2[1](x_prime, y)
            y = block2[2](x)
            y = self.drop(y)
            outputs.append(block2[3](x+y))

        return outputs


class ResidualBlock(BaseModule):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, 1),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, out_channels, 1))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
    
class ContextPredictorResnet(BaseModule):
    def __init__(self, q, n_feats):

        super(ContextPredictorResnet, self).__init__()
        self.q = q
        self.n_feats = n_feats

        self.input_resblock = ResidualBlock(n_feats, n_feats)
        self.output_resblocks = nn.ModuleList()

        for i in range(2*q):
            out_block = ResidualBlock(n_feats, n_feats)
            self.output_resblocks.append(out_block)

    def forward(self, x):
        # x = torch.flatten(x)
        x = self.input_resblock(x)

        outputs = []

        for block in self.output_resblocks:
            outputs.append(block(x))

        return outputs


class ConvReluNorm(BaseModule):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, 
                 n_layers, p_dropout):
        super(ConvReluNorm, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, 
                                                kernel_size, padding=kernel_size//2))
        self.norm_layers.append(torch.nn.LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(torch.nn.Conv1d(hidden_channels, hidden_channels, 
                                                    kernel_size, padding=kernel_size//2))
            self.norm_layers.append(torch.nn.LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x)
            x_prime = x.detach().reshape(x.shape[1], x.shape[0])
            x = self.norm_layers[i](x_prime)
            x = x.reshape(x.shape[1], x.shape[0])
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x
    

class ContextPredictorConvReluNorm(BaseModule):
    def __init__(self, k, hidden_channels, filter_channels, n_layers, 
                 kernel_size=1, p_dropout=0.0, window_size=None):
        super(ContextPredictorConvReluNorm, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.k = k

        self.input_block = ConvReluNorm(hidden_channels, hidden_channels, hidden_channels, kernel_size, n_layers, p_dropout)
        self.middle_block = ConvReluNorm(hidden_channels, hidden_channels, hidden_channels, kernel_size, n_layers, p_dropout)
        self.output_blocks = nn.ModuleList()

        for i in range(2*k):
            self.output_blocks.append(ResidualBlock(hidden_channels, hidden_channels))


    def forward(self, x):
        x = self.input_block(x)
        x = self.middle_block(x)
        outputs = []
        # x = torch.flatten(x)

        for block in self.output_blocks:
            outputs.append(block(x))

        return outputs
    

class ContextPredictorLSTM(BaseModule):
    def __init__(self, k, hidden_channels, filter_channels, n_layers, 
                 kernel_size=1, p_dropout=0.0, window_size=None):
        super(ContextPredictorLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.k = k

        self.output_blocks = nn.ModuleList()

        self.lstm = nn.LSTM(hidden_channels, hidden_channels, num_layers=3, 
                    dropout=p_dropout, batch_first=True)
        self.dropout = nn.Dropout(p_dropout)
        self.fc = nn.Linear(hidden_channels, hidden_channels)

        for i in range(2*k):
            self.output_blocks.append(ResidualBlock(hidden_channels, hidden_channels))


    def forward(self, x, hidden):
        print(x.shape, hidden.shape)
        x = self.lstm(x, hidden)
        x = self.dropout(x)
        x = self.fc(x)
        outputs = []
        # x = torch.flatten(x)

        for block in self.output_blocks:
            outputs.append(block(x))

        return outputs
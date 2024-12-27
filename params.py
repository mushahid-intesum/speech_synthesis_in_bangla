# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from model.utils import fix_len_compatibility


# data parameters
train_filelist_path = '/kaggle/input/bntts-data/resources/resources/data/train.txt'
valid_filelist_path = '/kaggle/input/bntts-data/resources/resources/data/val.txt'
test_filelist_path = '/kaggle/input/bntts-data/resources/resources/data/test.txt'
cmudict_path = './resources/cmu_dictionary'
add_blank = True
n_mels = 80
n_spks = 1  # 247 for Libri-TTS filelist and 1 for LJSpeech
spk_emb_dim = 80
n_feats = 80
n_fft = 1024
sample_rate = 22050
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = 'logs/multistream_stft_diffusion'
test_size = 4
n_epochs = 100
batch_size = 16
learning_rate = 1e-4
seed = 37
save_every = 1
out_size = fix_len_compatibility(2*22050//256)

# stft_config = {
#                 "subbands": 4,
#                 "gen_istft_n_fft": 16,
#                 "gen_istft_hop_size": 4,
#                 "inter_channels": 192,
#                 "hidden_channels": 192,
#                 "filter_channels": 768,
#                 "n_heads": 2,
#                 "n_layers": 6,
#                 "kernel_size": 3,
#                 "p_dropout": 0.1,
#                 "resblock": 1,
#                 "resblock_kernel_sizes": [3,7,11],
#                 "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
#                 "upsample_rates": [4,4],
#                 "upsample_initial_channel": 512,
#                 "upsample_kernel_sizes": [16,16],
#                 "n_layers_q": 3,
#                 "use_spectral_norm": False,
#             }

stft_config = {
                "initial_channel": 80,
                "resblock": 1,
                "resblock_kernel_sizes": [3,7,11],
                "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
                "upsample_rates": [4,4],
                "upsample_initial_channel": 192,
                "upsample_kernel_sizes": [16,16],
                "gen_istft_n_fft": 16,
                "gen_istft_hop_size": 4,
                "subbands": 4
            }
stft_loss_config = {
    "fft_sizes": [384, 683, 171],
    "hop_sizes": [30, 60, 10],
    "win_lengths": [150, 300, 60],
    "segment_size": 8192,
    "hop_length": 256,
}

segment_size = 8192
hop_length = 256
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from model.utils import fix_len_compatibility


# data parameters
train_filelist_path = '/mnt/Stuff/TTS/speech_synthesis_in_bangla-master/resources/data/train.txt'
valid_filelist_path = '/mnt/Stuff/TTS/speech_synthesis_in_bangla-master/resources/data/val.txt'
test_filelist_path = '/mnt/Stuff/TTS/speech_synthesis_in_bangla-master/resources/data/test.txt'
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
n_enc_channels = 256
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
log_dir = 'logs/dependency_parser'
test_size = 4
n_epochs = 100
batch_size = 1
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

synta_params = {
    "hidden_size": 192,
    "ffn_hidden_size": 768,
    "enc_ffn_kernel_size": 5,
    "enc_layers": 4,
    "dur_level": "word",
    "encoder_type": "rel_fft",
    "use_word_encoder": True,
    "num_heads": 2,

    "word_enc_layers": 4,
    "word_encoder_type": "rel_fft",
    "use_pitch_embed": False,
    "enc_prenet": True,
    "enc_pre_ln": True,
    "text_encoder_postnet": True,
    "dropout": 0.0,
    "add_word_pos": True,

    "predictor_hidden": -1,
    "dur_predictor_kernel": 3,
    "dur_predictor_layers": 2,
    "predictor_kernel": 5,
    "predictor_layers": 5,
    "predictor_dropout": 0.5,
    "hidden_size": 256,

    "use_fvae": False,

    "use_prior_flow": True,
    "prior_flow_hidden": 64,
    "prior_flow_kernel_size": 3,
    "prior_flow_n_blocks": 4,

    "lambda_kl": 1.0,
    "kl_min": 0.0,
    "lambda_sent_dur": 1.0,
    "kl_start_steps": 10000,
    "posterior_start_steps": 0,
    "frames_multiple": 4,
    "num_valid_plots": 10,
    "lr": 0.0002,
    "warmup_updates": 8000,
    "max_tokens": 40000,
    "valid_infer_interval": 10000,
    "max_sentences": 80,
    "max_updates": 480000,

    "min_sil_duration": 0.1,
    "processed_data_dir": './resources/process',
    "sample_rate": 22050,
    "hop_length": 256,
    "max_frames": 1548,
    "max_input_tokens": 1550,
    "predictor_grad": 0.1
}
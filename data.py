# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np

import torch
import torchaudio as ta

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from params import seed as random_seed

import sys
sys.path.insert(0, 'hifi-gan')
from utils import parse_filelist, intersperse, wav_to_mel
from text.bn_phonemiser import bangla_text_normalize, replace_number_with_text
import epitran
from meldataset import mel_spectrogram
from model.dataset_utils import collate_1d_or_2d
from model.align import get_mel2ph

class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        self.filepaths_and_text = parse_filelist(filelist_path)
        # self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max

        self.bn_phonemizer = epitran.Epitran('ben-Beng-east')
        random.seed(random_seed)
        random.shuffle(self.filepaths_and_text)
        self.ipa_dict = {'ɖ̤': 0,
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

    def get_data(self, filepath_and_text):
        filepath = '/mnt/Stuff/TTS/speech_synthesis_in_bangla-master/resources/data/wavs/'+filepath_and_text[0] + '.wav'
        text = filepath_and_text[1]
        text, ipa_tokens = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)

        return text, mel

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = wav_to_mel(audio, self.n_fft, 80, self.sample_rate, self.hop_length, self.win_length, self.f_min, self.f_max, center=False).squeeze()   
        return mel

    def get_ipa_tokens_from_text(self, tokens):
        ipa_tokens = []
        for tok in tokens:
            if tok in self.ipa_dict.keys():
                ipa_tokens.append(self.ipa_dict[tok])

        return ipa_tokens

    def get_text(self, text, add_blank=True):
        text_norm = replace_number_with_text(text)
        text_norm = bangla_text_normalize(text_norm)
        text_to_phonemes = self.bn_phonemizer.trans_list(text_norm)
        ipa_tokens = self.get_ipa_tokens_from_text(text_to_phonemes)
        text_norm = torch.IntTensor(ipa_tokens)

        # print(f'Text" {text}')
        # print(f'IPA tokens" {ipa_tokens}')
        # print(f'Text tokens" {text_to_phonemes}')
        # print(f'Text norm" {text_norm}')

        return text_norm, ipa_tokens

    def __getitem__(self, index):
        text, mel = self.get_data(self.filepaths_and_text[index])
        item = {'y': mel, 'x': text}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch

class TextMelBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_ = item['y'], item['x']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)

        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths}


class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank
        random.seed(random_seed)
        random.shuffle(self.filelist)

    def get_triplet(self, line):
        filepath, text, speaker = line[0], line[1], line[2]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        speaker = self.get_speaker(speaker)
        return (text, mel, speaker)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        text, mel, speaker = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []

        for i, item in enumerate(batch):
            y_, x_, spk_ = item['y'], item['x'], item['spk']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk.append(spk_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk}

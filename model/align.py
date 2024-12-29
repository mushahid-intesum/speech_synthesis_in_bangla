import re

import torch
import numpy as np
import librosa

from model.text_encoder import is_sil_phoneme

from textgrid import TextGrid, IntervalTier, Interval
import librosa

def mel_to_textgrid(mel_spec, sr=22050, hop_length=256, min_interval_length=0.1, energy_threshold=0.5):    
    if mel_spec.is_cuda:
        mel_spec = mel_spec.cpu()
    mel_spec = mel_spec.detach().numpy()
    
    # If the input is 3D (batch × n_mels × n_frames), take the first example
    if len(mel_spec.shape) == 3:
        mel_spec = mel_spec[0]
    
    # Calculate frame times
    n_frames = mel_spec.shape[1]
    frame_times = librosa.frames_to_time(range(n_frames), sr=sr, hop_length=hop_length)
    
    # Create TextGrid object
    tg = TextGrid(minTime=0, maxTime=frame_times[-1])
    
    # Calculate energy profile (average across mel bands)
    energy_profile = np.mean(mel_spec, axis=0)
    energy_profile = (energy_profile - energy_profile.min()) / (energy_profile.max() - energy_profile.min())
    
    # Create energy-based intervals
    energy_tier = IntervalTier(name='energy_activity', minTime=0, maxTime=frame_times[-1])
    
    # Find continuous segments of activity
    is_active = energy_profile > energy_threshold
    change_points = np.where(np.diff(is_active.astype(int)) != 0)[0]
    
    # Handle edge cases
    if len(change_points) == 0:
        # Single interval for the whole file
        mark = "active" if is_active[0] else "silence"
        energy_tier.add(0, frame_times[-1], mark)
    else:
        # Add first interval
        start_time = 0
        mark = "active" if is_active[0] else "silence"
        
        for change_idx in change_points:
            end_time = frame_times[change_idx]
            
            # Only add interval if it meets minimum length requirement
            if end_time - start_time >= min_interval_length:
                energy_tier.add(start_time, end_time, mark)
            
            start_time = end_time
            mark = "active" if not mark == "active" else "silence"
        
        # Add final interval
        if frame_times[-1] - start_time >= min_interval_length:
            energy_tier.add(start_time, frame_times[-1], mark)
    
    # Create frequency band tiers
    n_mels = mel_spec.shape[0]
    band_edges = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)
    
    # Create tier for prominent frequency bands
    freq_tier = IntervalTier(name='frequency_bands', minTime=0, maxTime=frame_times[-1])
    
    # Analyze frequency content in chunks
    chunk_size = int(min_interval_length * sr / hop_length)
    for i in range(0, n_frames-1, chunk_size):
        chunk_end = min(i + chunk_size, n_frames-1)
        chunk_data = mel_spec[:, i:chunk_end]
        
        # Find most prominent frequency band
        avg_energy = np.mean(chunk_data, axis=1)
        dominant_band = np.argmax(avg_energy)
        
        # Create interval with frequency range label
        start_time = frame_times[i]
        end_time = frame_times[chunk_end]

        # if start_time < end_time:
        band_label = f"{int(band_edges[dominant_band-1])}-{int(band_edges[dominant_band])}Hz"
        freq_tier.add(start_time, end_time, band_label)
    
    # Add tiers to TextGrid
    tg.append(energy_tier)
    tg.append(freq_tier)
    
    return tg

def get_mel2ph(ph, mel, hop_size, audio_sample_rate, min_sil_duration=0):
    itvs = mel_to_textgrid(mel)
    itvs_ = []
    for i in range(len(itvs)):
        if itvs[i].maxTime - itvs[i].minTime < min_sil_duration and i > 0 and is_sil_phoneme(itvs[i].mark):
            itvs_[-1].maxTime = itvs[i].maxTime
        else:
            itvs_.append(itvs[i])
    itvs.intervals = itvs_
    # itv_marks = [itv.mark for itv in itvs]
    # assert tg_len == ph_len, (tg_len, ph_len, itv_marks, ph_list, tg_fn)
    mel2ph = np.zeros([mel.shape[0]], int)
    i_itv = 0
    i_ph = 0
    while i_itv < len(itvs):
        itv = itvs[i_itv]

        start_frame = int(itv.minTime * audio_sample_rate / hop_size + 0.5)
        end_frame = int(itv.maxTime * audio_sample_rate / hop_size + 0.5)
        mel2ph[start_frame:end_frame] = i_ph + 1
        i_ph += 1
        i_itv += 1
    mel2ph[-1] = mel2ph[-2]
    assert not np.any(mel2ph == 0)
    T_t = len(ph)
    dur = mel2token_to_dur(mel2ph, T_t)

    return mel2ph.tolist(), dur.tolist()


def split_audio_by_mel2ph(audio, mel2ph, hop_size, audio_num_mel_bins):
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    if isinstance(mel2ph, torch.Tensor):
        mel2ph = mel2ph.numpy()
    assert len(audio.shape) == 1, len(mel2ph.shape) == 1
    split_locs = []
    for i in range(1, len(mel2ph)):
        if mel2ph[i] != mel2ph[i - 1]:
            split_loc = i * hop_size
            split_locs.append(split_loc)

    new_audio = []
    for i in range(len(split_locs) - 1):
        new_audio.append(audio[split_locs[i]:split_locs[i + 1]])
        new_audio.append(np.zeros([0.5 * audio_num_mel_bins]))
    return np.concatenate(new_audio)

    return None


def mel2token_to_dur(mel2token, T_txt=None, max_dur=None):
    is_torch = isinstance(mel2token, torch.Tensor)
    has_batch_dim = True
    if not is_torch:
        mel2token = torch.LongTensor(mel2token)
    if T_txt is None:
        T_txt = mel2token.max()
    if len(mel2token.shape) == 1:
        mel2token = mel2token[None, ...]
        has_batch_dim = False
    B, _ = mel2token.shape
    dur = mel2token.new_zeros(B, T_txt + 1).scatter_add(1, mel2token, torch.ones_like(mel2token))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    if not is_torch:
        dur = dur.numpy()
    if not has_batch_dim:
        dur = dur[0]
    return dur

    return None

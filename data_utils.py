import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import os
import torchaudio  # Or librosa, depending on your audio loader
import matplotlib.pyplot as plt
import numpy as np
import webrtcvad
import librosa

def trim_audio_silence(audio, sample_rate, aggressiveness=3, rmse_threshold=0.05, silence_frame_window=18, delta=0.01):
    """
    Trims silence from numpy audio array using VAD and RMSE.
    """
    audio = librosa.util.normalize(audio)
    vad = webrtcvad.Vad(aggressiveness)
    
    frame_duration = 30  # ms
    frame_size = int(sample_rate * frame_duration / 1000)
    hop_length = frame_size
    rmse = librosa.feature.rms(y=audio, frame_length=frame_size, hop_length=hop_length)[0]
    
    pcm_audio = (audio * 32768).astype(np.int16).tobytes()
    num_frames = len(audio) // frame_size

    speech_start_idx = 0
    speech_end_idx = len(audio)
    silence_counter = 0
    speech_start_detected = False

    for i in range(len(rmse)):
        start_byte = i * frame_size * 2  # 2 bytes per sample
        end_byte = start_byte + frame_size * 2
        frame = pcm_audio[start_byte:end_byte]

        if len(frame) < frame_size * 2:
            continue

        is_speech = vad.is_speech(frame, sample_rate)
        is_non_silent = rmse[i] > rmse_threshold

        if not speech_start_detected:
            if is_non_silent or is_speech:
                speech_start_detected = True
                speech_start_idx = i * hop_length
        else:
            if is_non_silent and is_speech:
                silence_counter = 0
                speech_end_idx = min((i + 1) * hop_length, len(audio))
            else:
                silence_counter += 1
                if (
                    silence_counter >= silence_frame_window and
                    all(abs(rmse[j] - rmse[j - 1]) < delta for j in range(i - silence_frame_window + 1, i + 1)) and
                    all(rmse[j] < rmse_threshold for j in range(i - silence_frame_window + 1, i + 1))
                ):
                    break

    trimmed_audio = audio[speech_start_idx:speech_end_idx]
    return trimmed_audio


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key


class Dataset_Custom(Dataset):
    def __init__(self, list_IDs, label_list, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.label_list = label_list
        self.max_len = 64600

        # For histogram
        self.original_durations = []
        self.trimmed_durations = []

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        file_name = self.list_IDs[index]
        label = self.label_list[index]
        file_path = os.path.join(self.base_dir, file_name)

        audio, sample_rate = torchaudio.load(file_path)
        audio = audio.mean(dim=0).numpy()  # Convert to mono numpy array
        self.original_durations.append(len(audio))

        trimmed_audio = trim_audio_silence(audio, sample_rate)
        self.trimmed_durations.append(len(trimmed_audio))

        padded_audio = pad(trimmed_audio, self.max_len)
        padded_audio = torch.tensor(padded_audio, dtype=torch.float32)

        return padded_audio, label, file_name

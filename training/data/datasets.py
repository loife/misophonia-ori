from __future__ import annotations
 
import json
from pathlib import Path
import config 

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset as TorchDataset

import librosa

 
class StrongDataset(TorchDataset):
    def __init__(self, dataset_root, split, label_encoder,
                 audio_length=10.0, sample_rate=16000, train=True):
        self.root = Path(dataset_root)
        self.split = split
        self.encoder = label_encoder
        self.sr = sample_rate
        self.audio_length = audio_length
        self.train = train
 
        self.fps = self.encoder.sr / self.encoder.frame_hop / self.encoder.net_pooling # Number of frames in each second, should be 25
        self.win_frames = int(round(audio_length * self.fps)) 
        self.win_samples = int(round(audio_length * sample_rate))
        self.samples_per_frame = sample_rate // int(round(self.fps))
 
        self.labels = config.labels
        self.encoder.labels = self.labels
        self.n_classes = len(self.labels)
 
        with open(self.root / f"{split}.json") as f:
            ann = json.load(f)
 
        audio_dir = self.root / str(sample_rate) / split
 
        self.audios: list[torch.Tensor] = [] # wavs
        self.events: list[list[tuple]] = []
        self.total_frames: list[int] = [] # how many frames fit in each clip
        self.filenames: list[str] = []
 
        for fname, ev_list in ann.items():
            wav = audio_dir / fname
            y, sr = sf.read(str(wav), dtype="float32", always_2d=False)

            if y.ndim > 1: # To mono from stereo
                y = y.mean(axis=1)

            if sr != sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
            self.audios.append(torch.from_numpy(y))
            # ms to s
            evs = [(e["label"], e["start"] / 1000.0, e["end"] / 1000.0) for e in ev_list]
            self.events.append(evs)
            self.total_frames.append(int(round(len(y) / self.samples_per_frame)))
            self.filenames.append(fname)
 
        if not self.train:
            self.index = []
            for ci, tf in enumerate(self.total_frames):
                n_win = max(1, tf // self.win_frames)
                for w in range(n_win):
                    self.index.append((ci, w * self.win_frames))
        else:
            self.index = None
 
    def __len__(self):
        return len(self.audios) if self.train else len(self.index)
 
    def _make_target(self, events, t0):
        t1 = t0 + self.audio_length
        rows = []
        for lab, on, off in events:
            lo, hi = max(on, t0), min(off, t1)
            if hi > lo:
                rows.append({"event_label": lab, "onset": lo - t0, "offset": hi - t0})
        df = pd.DataFrame(rows, columns=["event_label", "onset", "offset"])
        y = self.encoder.encode_strong_df(df)
        # (n_classes, win_frames)
        return torch.from_numpy(y).float().transpose(0, 1) 
 
    def _crop_audio(self, audio, frame_offset):
        start = frame_offset * self.samples_per_frame
        clip = audio[start:start + self.win_samples]
        if clip.shape[0] < self.win_samples:
            clip = torch.nn.functional.pad(clip, (0, self.win_samples - clip.shape[0]))
        return clip
 
    def __getitem__(self, i):
        if self.train:
            ci = i
            max_off = max(0, self.total_frames[ci] - self.win_frames)
            frame_offset = int(torch.randint(0, max_off + 1, (1,)).item())
        else:
            ci, frame_offset = self.index[i]
 
        t0 = frame_offset / self.fps
        frame_times = (t0 + (torch.arange(self.win_frames) + 0.5) / self.fps) * 1000.0
        return {
            "audio": self._crop_audio(self.audios[ci], frame_offset),
            "strong": self._make_target(self.events[ci], t0),
            "filename": self.filenames[ci],
            "offset": t0,
            "timestamps" : frame_times
        }

    
class WeakDataset(TorchDataset): 
    def __init__(self, dataset_root, split, label_encoder,
                 audio_length=10.0, sample_rate=16000, train=True):
        self.root = Path(dataset_root)
        self.split = split
        self.encoder = label_encoder
        self.sr = sample_rate
        self.audio_length = audio_length
        self.train = train
 
        self.win_samples = int(round(audio_length * sample_rate))
 
        self.labels = config.labels
        self.encoder.labels = self.labels
        self.n_classes = len(self.labels)
 
        with open(self.root / f"{split}.json") as f:
            ann = json.load(f)
 
        audio_dir = self.root / str(sample_rate) / split
 
        self.audios: list[torch.Tensor] = [] # wavs
        self.weak_labels: list[list[str]] = [] # list of label strings per clip
        self.filenames: list[str] = []
 
        for fname, label_list in ann.items():
            wav = audio_dir / fname
            y, sr = sf.read(str(wav), dtype="float32", always_2d=False)
 
            if y.ndim > 1: # to mono from stereo
                y = y.mean(axis=1)
            if sr != sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
 
            self.audios.append(torch.from_numpy(y))
            self.weak_labels.append([l for l in label_list if l in self.labels])
            self.filenames.append(fname)
 
    def __len__(self):
        return len(self.audios)
 
    def _make_target(self, labels):
        y = self.encoder.encode_weak(list(labels))
        return torch.from_numpy(y).float()
 
    def _crop_audio(self, audio, start):
        clip = audio[start:start + self.win_samples]
        if clip.shape[0] < self.win_samples:
            clip = torch.nn.functional.pad(clip, (0, self.win_samples - clip.shape[0]))
        return clip
 
    def __getitem__(self, i):
        audio = self.audios[i]
        n = audio.shape[0]
 
        if n <= self.win_samples:
            start = 0
        elif self.train:
            start = int(torch.randint(0, n - self.win_samples + 1, (1,)).item())
        else:
            start = (n - self.win_samples) // 2 # center crop for deterministic eval
 
        return {
            "audio": self._crop_audio(audio, start),
            "weak": self._make_target(self.weak_labels[i]),
            "filename": self.filenames[i],
        }


# Modified from https://github.com/fschmid56/PretrainedSED
class MixupDataset(TorchDataset):
    def __init__(self, dataset, beta=2.0, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            batch1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            batch2 = self.dataset[idx2]
            x1, x2 = batch1["audio"] - batch1["audio"].mean(), batch2["audio"] - batch2["audio"].mean()
            y1, y2 = batch1["strong"], batch2["strong"]
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            y = (y1 * l + y2 * (1. - l))
            batch1["audio"] = x
            batch1["strong"] = y
            return batch1
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
 
 
def get_strong_training_ds(dataset_root, label_encoder, split="train",
                           audio_length=10.0, sample_rate=16000, wavmix_p=0.0):
    ds = StrongDataset(dataset_root, split, label_encoder,
                       audio_length=audio_length, sample_rate=sample_rate, train=True)
    if wavmix_p > 0:
        ds = MixupDataset(ds, rate=wavmix_p)
    return ds
 
def get_strong_eval_ds(dataset_root, label_encoder, split="valid",
                       audio_length=10.0, sample_rate=16000):
    return StrongDataset(dataset_root, split, label_encoder,
                         audio_length=audio_length, sample_rate=sample_rate, train=False)


def get_weak_training_ds(dataset_root, label_encoder, split="train",
                         audio_length=10.0, sample_rate=16000):
    return WeakDataset(dataset_root, split, label_encoder,
                           audio_length=audio_length, sample_rate=sample_rate, train=True)

 
def get_weak_eval_ds(dataset_root, label_encoder, split="valid",
                     audio_length=10.0, sample_rate=16000):
    return WeakDataset(dataset_root, split, label_encoder,
                       audio_length=audio_length, sample_rate=sample_rate, train=False)

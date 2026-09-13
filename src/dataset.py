"""
Dataset, split loading, audio preprocessing, and spectrogram extraction for InsectSet459.
Standardizes 48 kHz wideband bioacoustic audio into (1, 128, 235) log-mel spectrograms.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

from src.config import (
    SAMPLE_RATE, WINDOW_SECONDS, HOP_SECONDS,
    N_FFT, HOP_LENGTH, N_MELS, F_MIN, F_MAX, EXPECTED_FRAMES,
    NUM_CLASSES, NUM_GENERA
)


class LogMelSpectrogram(nn.Module):
    """
    Per-instance normalized log-mel spectrogram extraction:
    48 kHz audio -> STFT (n_fft=4096, hop=1024) -> 128 Mel bands (400 Hz - 22000 Hz) ->
    log(1 + 10000 * power) -> zero mean, unit variance per instance (eps=1e-5).
    """
    def __init__(self, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH, n_mels: int = N_MELS,
                 f_min: float = F_MIN, f_max: float = F_MAX):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            normalized=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (1, time_samples) or (time_samples,)
        returns: (1, n_mels, frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel_transform(waveform)
        # Log scaling
        log_mel = torch.log(1.0 + 10000.0 * mel)
        # Per-instance zero mean, unit variance normalization
        mean = log_mel.mean()
        std = log_mel.std(unbiased=False) + 1e-5
        norm_mel = (log_mel - mean) / std

        # Ensure exact frame width (235 frames for 5.0 seconds at hop=1024)
        if norm_mel.shape[-1] < EXPECTED_FRAMES:
            pad_amount = EXPECTED_FRAMES - norm_mel.shape[-1]
            norm_mel = torch.nn.functional.pad(norm_mel, (0, pad_amount))
        elif norm_mel.shape[-1] > EXPECTED_FRAMES:
            norm_mel = norm_mel[..., :EXPECTED_FRAMES]

        return norm_mel


def load_species_and_genus_mapping(annotation_csv: str):
    """Loads species taxonomy and extracts genus prefix."""
    df = pd.read_csv(annotation_csv)
    species_list = sorted(df["species_name"].unique())
    id_to_species = {i: sp for i, sp in enumerate(species_list)}
    species_to_id = {sp: i for i, sp in id_to_species.items()}

    # Genus extraction (first token before underscore)
    genera = sorted(list(set(sp.split("_")[0] for sp in species_list)))
    genus_to_id = {g: i for i, g in enumerate(genera)}
    class_to_genus = [genus_to_id[id_to_species[i].split("_")[0]] for i in range(len(species_list))]

    return {
        "id_to_species": id_to_species,
        "species_to_id": species_to_id,
        "genus_to_id": genus_to_id,
        "class_to_genus": class_to_genus
    }


def compute_quartile_tiers(annotation_csv: str, subset: str = "Train"):
    """
    Computes quartile tiers based on the official training chunk distribution:
    Tier 1 (Head, n=114, >600 chunks)
    Tier 2 (Mid,  n=115, 251-600 chunks)
    Tier 3 (Few,  n=115, 119-250 chunks)
    Tier 4 (Tail, n=115, <=118 chunks)
    """
    df = pd.read_csv(annotation_csv)
    if "subset" in df.columns:
        train_df = df[df["subset"] == subset]
    else:
        train_df = df

    counts = train_df["species_name"].value_counts().to_dict()
    species_list = sorted(df["species_name"].unique())

    tier_mapping = {}
    tier_species = {"tier_1": [], "tier_2": [], "tier_3": [], "tier_4": []}

    for i, sp in enumerate(species_list):
        # Count approximate chunks: either explicit chunks or duration-based
        cnt = counts.get(sp, 0)
        # Check standard tier cutoffs
        if cnt > 600:
            t = "tier_1"
        elif cnt > 250:
            t = "tier_2"
        elif cnt > 118:
            t = "tier_3"
        else:
            t = "tier_4"

        tier_mapping[i] = t
        tier_species[t].append(i)

    return tier_mapping, tier_species


class SpecAugment(nn.Module):
    """Zero-cost spectral data augmentation baseline: Frequency and Time Masking."""
    def __init__(self, freq_mask_param: int = 16, time_mask_param: int = 20):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.time_mask(self.freq_mask(spec))

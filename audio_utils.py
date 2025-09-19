"""
Audio utility funkce, data struktury a základní audio processing.
Obsahuje filtry s dynamickým sample rate a audio handling.
"""

import numpy as np
import resampy
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PitchAnalysisResult:
    """Výsledek pitch analýzy s detailními metrikami"""
    fundamental_freq: Optional[float]
    confidence: float
    harmonics: List[float]
    method_used: str
    spectral_centroid: float
    spectral_rolloff: float
    spectrum: Optional[np.ndarray] = None  # Cached spectrum pro opakované použití
    freqs: Optional[np.ndarray] = None    # Cached frequencies

    @property
    def midi_note(self) -> Optional[int]:
        if self.fundamental_freq is None:
            return None
        return AudioUtils.freq_to_midi(self.fundamental_freq)


@dataclass
class VelocityAnalysisResult:
    """Výsledek velocity analýzy"""
    rms_db: float
    peak_db: float
    attack_peak_db: float
    attack_rms_db: float
    dynamic_range: float
    stereo_width: Optional[float] = None  # Pro stereo vzorky


@dataclass
class AudioSampleData:
    """Kompletní data audio vzorku"""
    filepath: Path
    waveform: np.ndarray
    sample_rate: int
    duration: float
    pitch_analysis: Optional[PitchAnalysisResult]
    velocity_analysis: Optional[VelocityAnalysisResult]
    assigned_velocity: Optional[int] = None
    target_midi_note: Optional[int] = None
    pitch_correction_semitones: Optional[float] = None


class AudioUtils:
    """Audio utility funkce s filtry"""

    MIDI_TO_NOTE = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }

    @staticmethod
    def freq_to_midi(freq: float) -> int:
        """Převod frekvence na MIDI číslo s validací"""
        if freq <= 0:
            raise ValueError("Frekvence musí být kladná")
        midi_float = 12 * np.log2(freq / 440.0) + 69
        return int(np.round(midi_float))

    @staticmethod
    def midi_to_freq(midi: int) -> float:
        """Převod MIDI čísla na frekvenci"""
        return 440.0 * 2 ** ((midi - 69) / 12)

    @staticmethod
    def midi_to_note_name(midi: int) -> str:
        """Převod MIDI čísla na název noty"""
        if not (0 <= midi <= 127):
            raise ValueError(f"MIDI číslo {midi} je mimo rozsah 0-127")

        octave = (midi // 12) - 1
        note_idx = midi % 12
        note = AudioUtils.MIDI_TO_NOTE[note_idx]
        return f"{note}{octave}"

    @staticmethod
    def normalize_audio(waveform: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalizace audio na cílovou úroveň"""
        if len(waveform.shape) > 1:
            rms = np.sqrt(np.mean(waveform.flatten() ** 2))
        else:
            rms = np.sqrt(np.mean(waveform ** 2))

        if rms == 0:
            return waveform

        target_rms = 10 ** (target_db / 20)
        return waveform * (target_rms / rms)

    @staticmethod
    def to_mono(waveform: np.ndarray, method: str = "mean") -> np.ndarray:
        """
        Převod na mono s volbou metody.

        Args:
            waveform: Input audio
            method: 'mean', 'left', 'right', 'mid_side'
        """
        if len(waveform.shape) <= 1:
            return waveform

        if waveform.shape[1] == 1:
            return waveform[:, 0]

        if method == "mean":
            return np.mean(waveform, axis=1)
        elif method == "left":
            return waveform[:, 0]
        elif method == "right":
            return waveform[:, 1] if waveform.shape[1] > 1 else waveform[:, 0]
        elif method == "mid_side" and waveform.shape[1] >= 2:
            # Mid channel pro mono conversion
            return (waveform[:, 0] + waveform[:, 1]) / 2
        else:
            return np.mean(waveform, axis=1)

    @staticmethod
    def ensure_format(waveform: np.ndarray, target_channels: int = 1) -> np.ndarray:
        """Zajistí správný formát audio (mono/stereo)"""
        if target_channels == 1:
            return AudioUtils.to_mono(waveform)
        elif target_channels == 2:
            if len(waveform.shape) == 1:
                # Mono -> stereo (duplicate)
                return np.column_stack([waveform, waveform])
            elif waveform.shape[1] == 1:
                # Mono column -> stereo
                return np.column_stack([waveform[:, 0], waveform[:, 0]])
            else:
                return waveform
        else:
            raise ValueError(f"Unsupported channel count: {target_channels}")

    @staticmethod
    def compute_spectrum(waveform: np.ndarray, sr: int, n_fft: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """
        Výpočet spektra s caching možnostmi.
        """
        n_fft = min(n_fft, len(waveform))
        if n_fft < 64:  # Minimum pro smysluplnou analýzu
            return np.array([]), np.array([])

        # Window pro lepší spektrální rozlišení
        window = np.hanning(n_fft)
        windowed = waveform[:n_fft] * window

        spectrum = np.abs(np.fft.fft(windowed))
        freqs = np.fft.fftfreq(n_fft, 1/sr)

        # Pouze pozitivní frekvence
        half_len = n_fft // 2
        return spectrum[:half_len], freqs[:half_len]

    @staticmethod
    def spectral_features(waveform: np.ndarray, sr: int) -> Tuple[float, float]:
        """Výpočet spektrálních charakteristik"""
        spectrum, freqs = AudioUtils.compute_spectrum(waveform, sr)

        if len(spectrum) == 0:
            return 0.0, sr/2

        # Spektrální centroid
        total_energy = np.sum(spectrum)
        if total_energy > 0:
            centroid = np.sum(freqs * spectrum) / total_energy
        else:
            centroid = 0.0

        # Spektrální rolloff (85% energie)
        cumsum_spectrum = np.cumsum(spectrum)
        if cumsum_spectrum[-1] > 0:
            rolloff_threshold = 0.85 * cumsum_spectrum[-1]
            rolloff_idx = np.where(cumsum_spectrum >= rolloff_threshold)[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else sr/2
        else:
            rolloff = sr/2

        return centroid, rolloff


class OptimizedFilters:
    """
    Filtry s dynamickým sample rate.
    Používá skutečný sample rate místo hardcoded hodnot.
    """

    @staticmethod
    def apply_processing_filters(audio: np.ndarray, sr: int) -> np.ndarray:
        """Aplikuje kompletnú sadu filtrů pro pitch detekci"""

        # 1. High-pass filter - odstranění DC a velmi nízkých frekvencí
        audio = OptimizedFilters.highpass_filter(audio, sr, cutoff_hz=40.0)

        # 2. Notch filter pro síťové brumání (50/60 Hz)
        audio = OptimizedFilters.notch_filter(audio, sr, freq=50.0, quality=10.0)
        audio = OptimizedFilters.notch_filter(audio, sr, freq=60.0, quality=10.0)

        # 3. Gentle low-pass pro redukci vysokofrekvenčního šumu
        audio = OptimizedFilters.lowpass_filter(audio, sr, cutoff_hz=8000.0)

        return audio

    @staticmethod
    def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
        """High-pass filter s dynamickým sample rate"""
        if len(audio) < 10:
            return audio

        # RC high-pass filter s cutoff frekvencí
        dt = 1.0 / sr
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        alpha = rc / (rc + dt)

        # Aplikace filtru
        filtered = np.zeros_like(audio)
        filtered[0] = audio[0]

        for i in range(1, len(audio)):
            filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1])

        return filtered

    @staticmethod
    def lowpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
        """Low-pass filter s dynamickým sample rate"""
        if len(audio) < 10:
            return audio

        # RC low-pass filter
        dt = 1.0 / sr
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        alpha = dt / (rc + dt)

        # Aplikace filtru
        filtered = np.zeros_like(audio)
        filtered[0] = audio[0]

        for i in range(1, len(audio)):
            filtered[i] = filtered[i-1] + alpha * (audio[i] - filtered[i-1])

        return filtered

    @staticmethod
    def notch_filter(audio: np.ndarray, sr: int, freq: float, quality: float = 10.0) -> np.ndarray:
        """Notch filter s dynamickým sample rate"""
        if len(audio) < 10:
            return audio

        # Vytvoř sinusovou vlnu na cílové frekvenci
        t = np.arange(len(audio)) / sr

        # Detekce amplitudy na cílové frekvenci pomocí korelace
        test_sin = np.sin(2 * np.pi * freq * t)
        test_cos = np.cos(2 * np.pi * freq * t)

        # Projekce signálu na sin/cos komponenty
        sin_coeff = np.dot(audio, test_sin) / len(audio)
        cos_coeff = np.dot(audio, test_cos) / len(audio)

        # Rekonstrukce komponenty na cílové frekvenci
        target_component = sin_coeff * test_sin + cos_coeff * test_cos

        # Odečtení (s faktorem útlumu)
        attenuation = 1.0 / quality  # Vyšší Q = větší útlum
        filtered = audio - attenuation * target_component

        return filtered


class AudioProcessor:
    """
    Unified audio processor pro stereo/mono handling a sample rate conversion.
    """

    @staticmethod
    def resample_audio(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Univerzální resampling pro mono i stereo"""
        if from_sr == to_sr:
            return audio

        try:
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                # Multi-channel
                resampled_channels = []
                for ch in range(audio.shape[1]):
                    resampled = resampy.resample(audio[:, ch], from_sr, to_sr)
                    resampled_channels.append(resampled)
                return np.column_stack(resampled_channels)
            else:
                # Mono
                audio_1d = audio.flatten() if len(audio.shape) > 1 else audio
                resampled = resampy.resample(audio_1d, from_sr, to_sr)
                return resampled[:, np.newaxis] if len(audio.shape) > 1 else resampled

        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return audio

    @staticmethod
    def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> Tuple[np.ndarray, int]:
        """Pitch shift s unified handling"""
        if abs(semitones) < 0.01:
            return audio, sr

        factor = 2 ** (semitones / 12)
        new_sr = int(sr * factor)

        try:
            return AudioProcessor.resample_audio(audio, sr, new_sr), new_sr
        except Exception as e:
            logger.error(f"Pitch shift error: {e}")
            return audio, sr

    @staticmethod
    def calculate_stereo_width(waveform: np.ndarray) -> Optional[float]:
        """Výpočet stereo width pro velocity analýzu"""
        if len(waveform.shape) < 2 or waveform.shape[1] < 2:
            return None

        left = waveform[:, 0]
        right = waveform[:, 1]

        # Korelace mezi kanály (1.0 = mono, 0.0 = nezávislé, -1.0 = opačné)
        correlation = np.corrcoef(left, right)[0, 1]

        # Stereo width (vyšší = více stereo)
        return 1.0 - abs(correlation)
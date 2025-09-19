"""
Pitch detection s hybridními algoritmy.
Zachovává modulární strukturu s osvědčenými detekčními metodami.
"""

import numpy as np
import logging
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
from audio_utils import PitchAnalysisResult, AudioUtils

logger = logging.getLogger(__name__)


class PitchDetector(ABC):
    """Abstraktní base class pro pitch detektory"""

    @abstractmethod
    def detect(self, waveform: np.ndarray, sr: int) -> PitchAnalysisResult:
        pass


class OriginalHybridPitchDetector(PitchDetector):
    """
    Hybridní pitch detector s kombinací spektrální analýzy a YIN algoritmu.
    """

    def __init__(self, fmin: float = 27.5, fmax: float = 4186.0):
        self.fmin = fmin
        self.fmax = fmax
        self.frame_length = 4096
        self.hop_length = 512

    def detect(self, waveform: np.ndarray, sr: int) -> PitchAnalysisResult:
        """Hlavní detekční metoda s kombinovanými algoritmy"""

        # Preprocessing
        audio = self._preprocess(waveform, sr)

        # 1. Spektrální analýza pro kandidáty
        spectral_candidates = self._spectral_peak_detection(audio, sr)

        # 2. YIN analýza pro potvrzení
        yin_candidates = self._yin_analysis(audio, sr)

        # 3. Kombinace výsledků
        fundamental_freq = self._combine_candidates(spectral_candidates, yin_candidates)

        # 4. Harmonická analýza
        harmonics = self._detect_harmonics(audio, sr, fundamental_freq) if fundamental_freq else []

        # 5. Confidence score
        confidence = self._calculate_confidence(fundamental_freq, harmonics, audio, sr)

        # 6. Spektrální charakteristiky
        centroid, rolloff = AudioUtils.spectral_features(audio, sr)

        return PitchAnalysisResult(
            fundamental_freq=fundamental_freq,
            confidence=confidence,
            harmonics=harmonics,
            method_used="hybrid_spectral_yin",
            spectral_centroid=centroid,
            spectral_rolloff=rolloff
        )

    def _preprocess(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Preprocessing bez agresivní filtrace"""
        audio = AudioUtils.to_mono(waveform)
        audio = AudioUtils.normalize_audio(audio, target_db=-20.0)

        # Základní filtrace
        audio = self._apply_basic_filters(audio, sr)

        return audio

    def _apply_basic_filters(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Základní filtrace bez přílišné agresivity"""

        # Pouze základní high-pass pro DC removal
        audio = self._simple_highpass_filter(audio, sr, cutoff_hz=30.0)

        # Gentle notch filter pouze pro 50 Hz
        audio = self._simple_notch_filter(audio, sr, freq=50.0, quality=5.0)

        return audio

    def _simple_highpass_filter(self, audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
        """Jednoduchý, méně agresivní high-pass filter"""
        if len(audio) < 10:
            return audio

        dt = 1.0 / sr
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        alpha = rc / (rc + dt)

        # Aplikace filtru
        filtered = np.zeros_like(audio)
        filtered[0] = audio[0]

        for i in range(1, len(audio)):
            filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1])

        return filtered

    def _simple_notch_filter(self, audio: np.ndarray, sr: int, freq: float, quality: float = 5.0) -> np.ndarray:
        """Jednoduchý, méně agresivní notch filter"""
        if len(audio) < 10:
            return audio

        t = np.arange(len(audio)) / sr

        # Detekce amplitudy na cílové frekvenci
        test_sin = np.sin(2 * np.pi * freq * t)
        test_cos = np.cos(2 * np.pi * freq * t)

        # Projekce signálu
        sin_coeff = np.dot(audio, test_sin) / len(audio)
        cos_coeff = np.dot(audio, test_cos) / len(audio)

        # Rekonstrukce komponenty
        target_component = sin_coeff * test_sin + cos_coeff * test_cos

        # Méně agresivní útlum
        attenuation = 1.0 / quality
        filtered = audio - attenuation * target_component

        return filtered

    def _spectral_peak_detection(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Spektrální detekce peaků"""
        candidates = []

        # Windowed analýza
        for start in range(0, len(audio) - self.frame_length, self.hop_length):
            frame = audio[start:start + self.frame_length]

            # Windowing function
            window = np.hanning(len(frame))
            windowed_frame = frame * window

            # FFT
            spectrum = np.abs(np.fft.fft(windowed_frame))
            freqs = np.fft.fftfreq(len(windowed_frame), 1/sr)

            # Pouze pozitivní frekvence
            half_spectrum = spectrum[:len(spectrum)//2]
            half_freqs = freqs[:len(freqs)//2]

            # Najdi lokální maxima
            peaks = self._find_spectral_peaks(half_spectrum, half_freqs)
            candidates.extend(peaks)

        # Seskupení
        return self._group_frequency_candidates(candidates)

    def _find_spectral_peaks(self, spectrum: np.ndarray, freqs: np.ndarray) -> List[Tuple[float, float]]:
        """Algoritmus pro hledání spektrálních peaků"""
        peaks = []

        # Práh (relativně k max spektra)
        min_peak_height = 0.1 * np.max(spectrum)

        for i in range(2, len(spectrum) - 2):
            if (spectrum[i] > spectrum[i-1] and
                spectrum[i] > spectrum[i+1] and
                spectrum[i] > spectrum[i-2] and
                spectrum[i] > spectrum[i+2] and
                spectrum[i] > min_peak_height and
                self.fmin <= freqs[i] <= self.fmax):

                peaks.append((freqs[i], spectrum[i]))

        return peaks

    def _yin_analysis(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """YIN analýza"""
        candidates = []

        for start in range(0, len(audio) - self.frame_length, self.hop_length):
            frame = audio[start:start + self.frame_length]
            pitch = self._yin_single_frame(frame, sr)

            if pitch and self.fmin <= pitch <= self.fmax:
                confidence = 0.8
                candidates.append((pitch, confidence))

        return self._group_frequency_candidates(candidates)

    def _yin_single_frame(self, frame: np.ndarray, sr: int) -> Optional[float]:
        """YIN implementace s opravami division by zero"""
        if len(frame) < 100:
            return None

        # YIN parameters
        max_tau = min(len(frame) // 2, int(sr / self.fmin))
        min_tau = max(1, int(sr / self.fmax))

        if max_tau <= min_tau:
            return None

        # Difference function
        diff = np.zeros(max_tau)
        for tau in range(1, max_tau):
            available_length = len(frame) - tau
            if available_length > 0:
                diff[tau] = np.sum((frame[:-tau] - frame[tau:]) ** 2)

        # Cumulative mean normalized difference
        cmndf = np.zeros(max_tau)
        cmndf[0] = 1.0

        for tau in range(1, max_tau):
            if tau <= 0:
                continue

            mean_diff = np.sum(diff[1:tau+1]) / tau

            if mean_diff > 1e-10:  # Ochrana proti division by zero
                cmndf[tau] = diff[tau] / mean_diff
            else:
                cmndf[tau] = 1.0

        # Threshold
        threshold = 0.3
        for tau in range(min_tau, max_tau):
            if cmndf[tau] < threshold:
                return sr / tau

        return None

    def _group_frequency_candidates(self, candidates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Skupinování kandidátů"""
        if not candidates:
            return []

        # Seřaď podle frekvence
        candidates.sort(key=lambda x: x[0])

        grouped = []
        current_group = [candidates[0]]

        for freq, confidence in candidates[1:]:
            # 5% tolerance
            if abs(freq - current_group[-1][0]) / current_group[-1][0] < 0.05:
                current_group.append((freq, confidence))
            else:
                # Uzavři aktuální skupinu
                if current_group:
                    avg_freq = np.mean([f for f, c in current_group])
                    total_confidence = np.sum([c for f, c in current_group])
                    grouped.append((avg_freq, total_confidence))
                current_group = [(freq, confidence)]

        # Poslední skupina
        if current_group:
            avg_freq = np.mean([f for f, c in current_group])
            total_confidence = np.sum([c for f, c in current_group])
            grouped.append((avg_freq, total_confidence))

        return grouped

    def _combine_candidates(self, spectral_candidates: List[Tuple[float, float]],
                          yin_candidates: List[Tuple[float, float]]) -> Optional[float]:
        """Kombinace kandidátů"""

        all_candidates = []

        # Váhy
        for freq, confidence in spectral_candidates:
            all_candidates.append((freq, confidence * 0.6, "spectral"))

        for freq, confidence in yin_candidates:
            all_candidates.append((freq, confidence * 0.8, "yin"))

        if not all_candidates:
            return None

        # Seskupení velmi podobných kandidátů
        final_candidates = []

        for freq, confidence, method in all_candidates:
            # Najdi podobné kandidáty
            similar_found = False
            for i, (existing_freq, existing_conf, existing_methods) in enumerate(final_candidates):
                if abs(freq - existing_freq) / existing_freq < 0.03:  # 3% tolerance
                    # Aktualizuj existující kandidát
                    new_freq = (existing_freq * existing_conf + freq * confidence) / (existing_conf + confidence)
                    new_conf = existing_conf + confidence
                    new_methods = existing_methods + [method]
                    final_candidates[i] = (new_freq, new_conf, new_methods)
                    similar_found = True
                    break

            if not similar_found:
                final_candidates.append((freq, confidence, [method]))

        if not final_candidates:
            return None

        # Výběr kandidáta
        best_candidate = None
        best_score = 0

        for freq, confidence, methods in final_candidates:
            score = confidence
            if len(set(methods)) > 1:  # Potvrzeno více metodami
                score *= 1.5

            # Penalizace pro vysoké frekvence
            if freq > 1500:
                score *= 0.3
            elif freq > 800:
                score *= 0.7

            if score > best_score:
                best_score = score
                best_candidate = freq

        return best_candidate

    def _detect_harmonics(self, audio: np.ndarray, sr: int, fundamental: Optional[float]) -> List[float]:
        """Detekce harmonických"""
        if fundamental is None:
            return []

        harmonics = []

        # Hledej harmonické až do 8. harmonické
        for harmonic_num in range(2, 9):
            harmonic_freq = fundamental * harmonic_num

            if harmonic_freq > sr / 2:  # Nyquist limit
                break

            # Ověření přítomnosti harmonické
            if self._verify_harmonic_presence(audio, sr, harmonic_freq):
                harmonics.append(harmonic_freq)

        return harmonics

    def _verify_harmonic_presence(self, audio: np.ndarray, sr: int, target_freq: float) -> bool:
        """Verifikace harmonických"""
        # Spektrální analýza
        n_fft = min(4096, len(audio))
        spectrum = np.abs(np.fft.fft(audio[:n_fft]))
        freqs = np.fft.fftfreq(n_fft, 1/sr)[:n_fft//2]
        spectrum = spectrum[:n_fft//2]

        # Najdi nearest frequency bin
        freq_idx = np.argmin(np.abs(freqs - target_freq))

        # Tolerance window
        window_size = max(1, int(0.02 * len(freqs)))  # 2% tolerance
        start_idx = max(0, freq_idx - window_size)
        end_idx = min(len(spectrum), freq_idx + window_size + 1)

        # Peak v tolerance window
        local_max = np.max(spectrum[start_idx:end_idx])
        overall_mean = np.mean(spectrum)

        # Práh
        return local_max > 3 * overall_mean

    def _calculate_confidence(self, fundamental: Optional[float], harmonics: List[float],
                            audio: np.ndarray, sr: int) -> float:
        """Výpočet confidence"""
        if fundamental is None:
            return 0.0

        confidence = 0.5  # Base confidence

        # Bonus za harmonické
        if len(harmonics) >= 2:
            confidence += 0.3
        elif len(harmonics) >= 1:
            confidence += 0.2

        # Bonus za frekvenci v hudebním rozsahu
        if 80 <= fundamental <= 1000:
            confidence += 0.2
        elif 1000 < fundamental <= 2000:
            confidence += 0.1

        return min(1.0, confidence)


class SimplePitchDetector(PitchDetector):
    """
    Jednoduchý pitch detector pouze s spektrální analýzou.
    """

    def __init__(self, fmin: float = 27.5, fmax: float = 4186.0):
        self.fmin = fmin
        self.fmax = fmax

    def detect(self, waveform: np.ndarray, sr: int) -> PitchAnalysisResult:
        """Rychlá spektrální detekce"""
        audio = AudioUtils.to_mono(waveform)
        audio = AudioUtils.normalize_audio(audio, target_db=-20.0)

        # Jednoduchá spektrální analýza
        spectrum, freqs = AudioUtils.compute_spectrum(audio, sr, n_fft=4096)

        if len(spectrum) == 0:
            return PitchAnalysisResult(
                fundamental_freq=None,
                confidence=0.0,
                harmonics=[],
                method_used="simple_spectral",
                spectral_centroid=0.0,
                spectral_rolloff=sr/2
            )

        # Najdi peak v hudebním rozsahu
        valid_indices = np.where((freqs >= self.fmin) & (freqs <= self.fmax))[0]

        if len(valid_indices) == 0:
            fundamental_freq = None
            confidence = 0.0
        else:
            valid_spectrum = spectrum[valid_indices]
            valid_freqs = freqs[valid_indices]

            # Najdi maximum
            max_idx = np.argmax(valid_spectrum)
            fundamental_freq = valid_freqs[max_idx]

            # Confidence založená na poměru peak/mean
            mean_power = np.mean(valid_spectrum)
            peak_power = valid_spectrum[max_idx]
            confidence = min(1.0, (peak_power / mean_power) / 10.0) if mean_power > 0 else 0.0

        # Spektrální charakteristiky
        centroid, rolloff = AudioUtils.spectral_features(audio, sr)

        return PitchAnalysisResult(
            fundamental_freq=fundamental_freq,
            confidence=confidence,
            harmonics=[],  # Nedetekuje harmonické
            method_used="simple_spectral",
            spectral_centroid=centroid,
            spectral_rolloff=rolloff
        )


# Aliasy pro zpětnou kompatibilitu
OptimizedHybridPitchDetector = OriginalHybridPitchDetector
AdaptivePitchDetector = OriginalHybridPitchDetector
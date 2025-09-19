"""
Specializovaný pitch detector pro piano s hybridními algoritmy.
Zachovává modulární strukturu s osvědčenými detekčními metodami.
"""

import numpy as np
import logging
from typing import Optional, List, Tuple, Dict
from abc import ABC, abstractmethod
from scipy import signal
from scipy.signal import find_peaks, stft

# Předpokládáme, že máte audio_utils modul
try:
    from audio_utils import PitchAnalysisResult, AudioUtils
except ImportError:
    # Fallback implementace pokud audio_utils neexistuje
    class PitchAnalysisResult:
        def __init__(self, fundamental_freq=None, confidence=0.0, harmonics=None,
                     method_used="unknown", spectral_centroid=0.0, spectral_rolloff=0.0):
            self.fundamental_freq = fundamental_freq
            self.confidence = confidence
            self.harmonics = harmonics or []
            self.method_used = method_used
            self.spectral_centroid = spectral_centroid
            self.spectral_rolloff = spectral_rolloff
            self.midi_note = self._freq_to_midi(fundamental_freq) if fundamental_freq else None

        def _freq_to_midi(self, freq):
            return int(round(12 * np.log2(freq / 440.0) + 69))

    class AudioUtils:
        @staticmethod
        def to_mono(waveform):
            if len(waveform.shape) == 2:
                return np.mean(waveform, axis=1)
            return waveform

        @staticmethod
        def normalize_audio(audio, target_db=-20.0):
            current_db = 20 * np.log10(np.max(np.abs(audio)))
            gain_db = target_db - current_db
            gain_linear = 10 ** (gain_db / 20.0)
            return audio * gain_linear

        @staticmethod
        def spectral_features(audio, sr):
            spectrum = np.abs(np.fft.fft(audio))
            freqs = np.fft.fftfreq(len(audio), 1/sr)
            half_len = len(spectrum) // 2
            spectrum = spectrum[:half_len]
            freqs = freqs[:half_len]

            # Spectral centroid
            centroid = np.sum(freqs * spectrum) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0

            # Spectral rolloff
            cumsum = np.cumsum(spectrum)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else sr/2

            return centroid, rolloff

logger = logging.getLogger(__name__)


class PitchDetector(ABC):
    """Abstraktní base class pro pitch detektory"""

    @abstractmethod
    def detect(self, waveform: np.ndarray, sr: int) -> PitchAnalysisResult:
        pass


class PianoPitchDetector(PitchDetector):
    """
    Specializovaný pitch detector pro piano s kombinací více metod:
    1. Harmonická template matching
    2. Cepstral analysis
    3. Subharmonic summation
    4. Piano-specific frequency mapping
    """

    def __init__(self, fmin: float = 27.5, fmax: float = 4186.0):
        self.fmin = fmin
        self.fmax = fmax
        self.piano_frequencies = self._generate_piano_frequencies()

        # Piano-specific parametry - optimalizované pro rychlost
        self.frame_length = 4096  # Zmenšeno z 8192 pro rychlejší zpracování
        self.hop_length = 1024   # Zmenšeno z 2048
        self.overlap = 0.75

    def _generate_piano_frequencies(self) -> np.ndarray:
        """Generuje frekvence klavírních tónů (A0 až C8)"""
        # A4 = 440 Hz jako reference
        A4_freq = 440.0
        A4_key = 49  # A4 je 49. klávesa na klavíru

        frequencies = []
        for key in range(1, 89):  # 88 kláves
            # Výpočet frekvence: f = A4 * 2^((n-49)/12)
            freq = A4_freq * (2 ** ((key - A4_key) / 12))
            if self.fmin <= freq <= self.fmax:
                frequencies.append(freq)

        return np.array(frequencies)

    def detect(self, waveform: np.ndarray, sr: int) -> PitchAnalysisResult:
        """Hlavní detekční metoda s omezením na první 3.5 sekundy"""

        logger.debug(f"Piano detector starting - original length: {len(waveform)} samples, SR: {sr} Hz")

        # Preprocessing specifický pro piano
        audio = self._piano_preprocess(waveform, sr)

        # OMEZENÍ NA PRVNÍ 3.5 SEKUNDY
        max_samples = int(3.5 * sr)  # 3.5 sekundy
        original_length = len(audio)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            logger.debug(f"Trimmed audio: {original_length} → {max_samples} samples ({len(audio)/sr:.2f}s)")
        else:
            logger.debug(f"Using full audio: {len(audio)} samples ({len(audio)/sr:.2f}s)")

        if len(audio) < self.frame_length:
            logger.debug(f"Audio too short for analysis: {len(audio)} < {self.frame_length}")
            return self._empty_result()

        logger.debug(f"Starting multi-method analysis with {len(self.piano_frequencies)} piano frequencies")

        # Kombinace metod
        methods_results = {}

        # 1. Harmonická template matching
        logger.debug("Running harmonic template matching...")
        methods_results['harmonic_template'] = self._harmonic_template_matching(audio, sr)
        self._log_method_results('harmonic_template', methods_results['harmonic_template'])

        # 2. Cepstral analysis
        logger.debug("Running cepstral analysis...")
        methods_results['cepstral'] = self._cepstral_analysis(audio, sr)
        self._log_method_results('cepstral', methods_results['cepstral'])

        # 3. Subharmonic summation spectrum
        logger.debug("Running subharmonic summation...")
        methods_results['subharmonic'] = self._subharmonic_summation(audio, sr)
        self._log_method_results('subharmonic', methods_results['subharmonic'])

        # 4. Autocorrelation s piano bias
        logger.debug("Running piano autocorrelation...")
        methods_results['autocorr_piano'] = self._piano_autocorrelation(audio, sr)
        self._log_method_results('autocorr_piano', methods_results['autocorr_piano'])

        # 5. Multi-pitch detection pro akordy
        logger.debug("Running multi-pitch detection...")
        methods_results['multi_pitch'] = self._multi_pitch_detection(audio, sr)
        self._log_method_results('multi_pitch', methods_results['multi_pitch'])

        # Kombinace výsledků
        logger.debug("Combining results from all methods...")
        fundamental_freq = self._combine_piano_results(methods_results)

        if fundamental_freq:
            logger.debug(f"Combined result: {fundamental_freq:.2f} Hz")
        else:
            logger.debug("No reliable frequency detected by any method")

        # Detekce harmonických
        harmonics = self._detect_piano_harmonics(audio, sr, fundamental_freq) if fundamental_freq else []
        logger.debug(f"Detected {len(harmonics)} harmonics: {[f'{h:.1f}' for h in harmonics[:5]]}")

        # Confidence score
        confidence = self._calculate_piano_confidence(fundamental_freq, harmonics, methods_results, audio, sr)
        logger.debug(f"Final confidence: {confidence:.3f}")

        # Spektrální charakteristiky
        centroid, rolloff = AudioUtils.spectral_features(audio, sr)
        logger.debug(f"Spectral features - Centroid: {centroid:.1f} Hz, Rolloff: {rolloff:.1f} Hz")

        # SPRÁVNÝ RETURN ZDE – po všech výpočtech
        return PitchAnalysisResult(
            fundamental_freq=fundamental_freq,
            confidence=confidence,
            harmonics=harmonics,
            method_used="piano_specialized",
            spectral_centroid=centroid,
            spectral_rolloff=rolloff
        )

    def _log_method_results(self, method_name: str, results: Dict) -> None:
        """Loguje výsledky jednotlivých metod pro debug – pouze logování, bez returnu"""
        candidates = results.get('candidates', [])
        confidence = results.get('method_confidence', 0.0)

        logger.debug(f"  {method_name}: {len(candidates)} candidates, confidence={confidence:.2f}")
        for i, (freq, score) in enumerate(candidates[:3]):  # Top 3 kandidáti
            closest_piano = self._find_closest_piano_freq(freq)
            cents_off = 1200 * np.log2(freq / closest_piano) if closest_piano else 0
            logger.debug(f"    #{i+1}: {freq:.2f} Hz (piano: {closest_piano:.2f} Hz, {cents_off:+.1f} cents) score={score:.3f}")

    def _find_closest_piano_freq(self, freq: float) -> float:
        """Najde nejbližší piano frekvenci"""
        distances = np.abs(self.piano_frequencies - freq)
        min_idx = np.argmin(distances)
        return self.piano_frequencies[min_idx]

    def _piano_preprocess(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Piano-specific preprocessing s adaptivní normalizací"""

        logger.debug(f"Preprocessing - Input shape: {waveform.shape}, SR: {sr}")

        # Explicitní stereo-to-mono převod
        if len(waveform.shape) == 2:
            # Sčítání kanálů s mírným upřednostněním levého kanálu (typické pro piano)
            audio = 0.55 * waveform[:, 0] + 0.45 * waveform[:, 1]
            logger.debug("Converted stereo to mono (55% L + 45% R)")
        else:
            audio = waveform.flatten()
            logger.debug("Using mono audio as-is")

        # ADAPTIVNÍ NORMALIZACE - pokud je signál slabší než -6 dB, zesil na -6 dB
        orig_max = np.max(np.abs(audio))
        if orig_max > 0:
            current_db = 20 * np.log10(orig_max)
            target_db = -6.0

            # Pouze pokud je signál slabší než -6 dB, normalizuj
            if current_db < target_db:
                gain_db = target_db - current_db
                gain_linear = 10 ** (gain_db / 20.0)
                audio = audio * gain_linear
                new_max = np.max(np.abs(audio))
                logger.debug(f"Adaptive normalization: {current_db:.1f} dB → {target_db:.1f} dB (gain: +{gain_db:.1f} dB)")
                logger.debug(f"Amplitude: {orig_max:.4f} → {new_max:.4f}")
            else:
                logger.debug(f"No normalization needed: signal already at {current_db:.1f} dB (≥ -6 dB)")
        else:
            logger.warning("Zero amplitude signal - cannot normalize")

        # Piano-specific filtrace
        audio = self._piano_filter(audio, sr)

        # Emphasis pro nižší frekvence (piano má silné nižší harmonické)
        audio = self._apply_low_frequency_emphasis(audio, sr)

        final_max = np.max(np.abs(audio))
        final_db = 20 * np.log10(final_max) if final_max > 0 else -float('inf')
        logger.debug(f"After filtering: max amplitude = {final_max:.4f} ({final_db:.1f} dB)")

        return audio

    def _piano_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Filtrace zaměřená na piano charakteristiky"""

        # Velmi jemný high-pass (piano má důležité nižší složky)
        nyquist = sr / 2
        low_cutoff = 20.0 / nyquist
        b, a = signal.butter(2, low_cutoff, btype='high')
        audio = signal.filtfilt(b, a, audio)

        # Notch pro síťové brumění (50/60 Hz a harmonické)
        for freq in [50, 60, 100, 120]:
            if freq < sr / 2:
                audio = self._notch_filter(audio, sr, freq, quality=10)

        return audio

    def _apply_low_frequency_emphasis(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Zdůraznění nízkých frekvencí pro lepší detekci fundamentálů"""

        # Pre-emphasis filter pro piano
        # Boost nízkých frekvencí kde jsou fundamentály
        nyquist = sr / 2
        low_freq = 200.0 / nyquist
        high_freq = 2000.0 / nyquist

        # Bandpass pro piano rozsah s mírným boostem
        b, a = signal.butter(2, [low_freq, high_freq], btype='band')
        emphasized = signal.filtfilt(b, a, audio)

        # Kombinace original + emphasized
        return 0.7 * audio + 0.3 * emphasized

    def _notch_filter(self, audio: np.ndarray, sr: int, freq: float, quality: float = 10.0) -> np.ndarray:
        """Notch filter pro piano preprocessing"""
        nyquist = sr / 2
        normalized_freq = freq / nyquist
        b, a = signal.iirnotch(normalized_freq, quality)
        return signal.filtfilt(b, a, audio)

    def _harmonic_template_matching(self, audio: np.ndarray, sr: int) -> Dict:
        """Harmonická template matching optimalizovaná pro piano"""
        # STFT pro frequency analysis
        f, t, Zxx = stft(audio, fs=sr, nperseg=self.frame_length, noverlap=int(self.frame_length * self.overlap))
        magnitude = np.abs(Zxx)

        # Průměrování přes čas pro stabilní spektrum
        avg_magnitude = np.mean(magnitude, axis=1)

        candidates = []
        for piano_freq in self.piano_frequencies:
            # Template: fundamental + 4 harmonické
            template_peaks = [piano_freq * h for h in [1, 2, 3, 4, 5]]
            template_score = 0.0
            for peak in template_peaks:
                if peak > sr / 2:
                    break
                idx = np.argmin(np.abs(f - peak))
                tolerance = max(1, int(len(f) * 0.01))  # 1% tolerance
                local_max = np.max(avg_magnitude[max(0, idx - tolerance):idx + tolerance + 1])
                template_score += local_max

            candidates.append((piano_freq, template_score))

        # Seřaď podle score
        candidates.sort(key=lambda x: x[1], reverse=True)

        return {
            'candidates': candidates[:3],
            'method_confidence': 0.8 if candidates else 0.0
        }

    def _cepstral_analysis(self, audio: np.ndarray, sr: int) -> Dict:
        """Cepstral analysis pro fundamental frequency detekci.

        Vypočítá real cepstrum z log-spektra a hledá quefrency peaks v rozsahu piano frekvencí.
        Ošetřuje okrajové případy (krátký signál, chybějící rozsah).
        """
        # Spectrum a log-spektrum (s epsilon pro stabilní log)
        spectrum = np.abs(np.fft.fft(audio))
        log_spectrum = np.log(spectrum + 1e-10)

        # Inverzní FFT pro real cepstrum
        cepstrum = np.fft.ifft(log_spectrum).real
        cepstrum = np.abs(cepstrum[:len(cepstrum) // 2])  # Pouze pozitivní část

        # Přepočet quefrencies do sekund
        quefrencies = np.arange(len(cepstrum)) / sr
        min_quefrency = 1 / self.fmax
        max_quefrency = min(len(cepstrum) / sr, 1 / self.fmin)

        # Debug log rozsahu
        logger.debug(
            f"Cepstrum: {len(cepstrum)} bins, quefrency range: {quefrencies[0]:.6f} - {quefrencies[-1]:.6f}s, "
            f"valid range: {min_quefrency:.6f}-{max_quefrency:.6f}s"
        )

        # Filtrace validního rozsahu
        mask = (quefrencies >= min_quefrency) & (quefrencies <= max_quefrency)
        valid_quefrency = quefrencies[mask]
        valid_cepstrum = cepstrum[mask]

        logger.debug(f"Valid quefrency bins: {len(valid_cepstrum)}")

        if len(valid_cepstrum) == 0:
            logger.debug("Cepstral analysis: No valid quefrency range, skipping peaks")
            return {
                'candidates': [],
                'method_confidence': 0.0
            }

        # Najdi quefrency peaks v validním rozsahu
        peaks, properties = find_peaks(
            valid_cepstrum,
            height=np.max(valid_cepstrum) * 0.4  # 40% maximální hodnoty jako práh
        )

        candidates = []
        for peak_idx in peaks:
            if peak_idx < len(valid_quefrency):
                freq = 1.0 / valid_quefrency[peak_idx]
                score = valid_cepstrum[peak_idx]
                candidates.append((freq, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Cepstral peaks found: {len(candidates)}")
        if candidates:
            logger.debug(f"Top candidate: {candidates[0][0]:.2f} Hz (score={candidates[0][1]:.3f})")

        return {
            'candidates': candidates[:3],
            'method_confidence': 0.7 if candidates else 0.0
        }


    def _subharmonic_summation(self, audio: np.ndarray, sr: int) -> Dict:
        """Subharmonic summation spectrum pro robustní detekci"""
        # STFT
        f, t, Zxx = stft(audio, fs=sr, nperseg=self.frame_length, noverlap=int(self.frame_length * self.overlap))
        magnitude = np.abs(Zxx)

        # Průměrování přes čas
        avg_magnitude = np.mean(magnitude, axis=1)
        freqs = f

        # Subharmonic summation
        summation_spectrum = np.zeros_like(avg_magnitude)
        for i in range(len(freqs)):
            freq = freqs[i]
            if freq < self.fmin:
                continue
            harmonic_sum = 0.0
            harmonic_count = 0

            for harmonic in range(1, 16):  # až 15. harmonická
                harmonic_freq = freq * harmonic
                if harmonic_freq >= freqs[-1]:
                    break

                # Najdi nejbližší bin
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))

                # Weight podle harmonické (piano má specifický profil)
                weight = 1.0 / (harmonic ** 0.5)  # Piano charakteristika
                harmonic_sum += avg_magnitude[harmonic_idx] * weight
                harmonic_count += 1

            if harmonic_count > 0:
                summation_spectrum[i] = harmonic_sum / harmonic_count

        # Najdi peaks v summation spectrum
        peaks, _ = find_peaks(
            summation_spectrum,
            height=np.max(summation_spectrum) * 0.2,
            distance=int(len(summation_spectrum) * 0.01)
        )

        candidates = []
        for peak_idx in peaks:
            freq = freqs[peak_idx]
            power = summation_spectrum[peak_idx]
            candidates.append((freq, power))

        # Seřaď podle power
        candidates.sort(key=lambda x: x[1], reverse=True)

        return {
            'candidates': candidates[:3],
            'method_confidence': 0.6 if candidates else 0.0
        }

    def _piano_autocorrelation(self, audio: np.ndarray, sr: int) -> Dict:
        """Autocorrelation s piano-specific optimalizacemi"""

        # Preprocessing pro autocorrelaci
        processed_audio = self._preprocess_for_autocorr(audio, sr)

        # Autocorrelation
        correlation = np.correlate(processed_audio, processed_audio, mode='full')
        correlation = correlation[correlation.size // 2:]

        # Lags odpovídající piano frekvencím
        max_lag = int(sr / self.fmin)
        min_lag = int(sr / self.fmax)

        if max_lag >= len(correlation):
            max_lag = len(correlation) - 1

        valid_correlation = correlation[min_lag:max_lag]
        valid_lags = np.arange(min_lag, max_lag)

        # Najdi peaks
        peaks, _ = find_peaks(
            valid_correlation,
            height=np.max(valid_correlation) * 0.3,
            distance=int(sr * 0.01)  # Min 10ms mezi peaky
        )

        candidates = []
        for peak_idx in peaks:
            if peak_idx < len(valid_lags):
                lag = valid_lags[peak_idx]
                freq = sr / lag
                confidence = valid_correlation[peak_idx]
                candidates.append((freq, confidence))

        # Piano bias - upřednostni klavírní frekvence
        biased_candidates = []
        for freq, confidence in candidates:
            # Najdi nejbližší klavírní frekvenci
            piano_distances = np.abs(self.piano_frequencies - freq)
            min_distance = np.min(piano_distances)

            # Bias based on distance to piano frequency
            if min_distance < freq * 0.02:  # 2% tolerance
                bias_factor = 1.5
            elif min_distance < freq * 0.05:  # 5% tolerance
                bias_factor = 1.2
            else:
                bias_factor = 0.8

            biased_candidates.append((freq, confidence * bias_factor))

        biased_candidates.sort(key=lambda x: x[1], reverse=True)

        return {
            'candidates': biased_candidates[:3],
            'method_confidence': 0.7 if biased_candidates else 0.0
        }

    def _preprocess_for_autocorr(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Preprocessing specifický pro autocorrelaci"""

        # Center clipping pro zdůraznění periodicity
        threshold = 0.3 * np.std(audio)
        processed = np.copy(audio)
        processed[np.abs(processed) < threshold] = 0
        processed[processed > threshold] = processed[processed > threshold] - threshold
        processed[processed < -threshold] = processed[processed < -threshold] + threshold

        return processed

    def _multi_pitch_detection(self, audio: np.ndarray, sr: int) -> Dict:
        """Multi-pitch detection pro detekci akordů"""

        # STFT pro time-frequency analýzu
        f, t, Zxx = stft(audio, sr, nperseg=2048, noverlap=1536)
        magnitude = np.abs(Zxx)

        # Průměrování přes čas
        avg_magnitude = np.mean(magnitude, axis=1)

        # Najdi více peaks současně
        peaks, properties = find_peaks(
            avg_magnitude,
            height=np.max(avg_magnitude) * 0.15,
            distance=int(len(avg_magnitude) * 0.02),
            prominence=np.max(avg_magnitude) * 0.1
        )

        candidates = []
        for peak_idx in peaks:
            freq = f[peak_idx]
            if self.fmin <= freq <= self.fmax:
                power = avg_magnitude[peak_idx]
                candidates.append((freq, power))

        # Seřaď podle power
        candidates.sort(key=lambda x: x[1], reverse=True)

        return {
            'candidates': candidates[:5],  # Až 5 současných tónů
            'method_confidence': 0.5 if candidates else 0.0
        }

    def _combine_piano_results(self, methods_results: Dict) -> Optional[float]:
        """Kombinace výsledků všech metod s robustním rozhodováním o oktávě.

        Tento refaktor používá hlasování napříč metodami (harmonic_template, cepstral,
        subharmonic, autocorr, multi_pitch) a porovnává tři varianty každého kandidáta
        (f/2, f, 2f). Každé variantě se spočítá agregované skóre složené z:
          - původního "group" skóre (z předchozí kombinace)
          - podpory od jednotlivých metod (pokud metoda má kandidát blízko testované frekvence)
          - podpory z harmonic_template pro více harmonických

        Varianty jsou vybírány pouze pokud jsou v rozsahu [fmin, fmax] a posun o oktávu
        se uplatní pouze pokud je pro něj výrazná evidence (víc než drobné zlepšení).
        """

        all_candidates = []

        # Váhy pro jednotlivé metody (stejné jako jinde, slouží také pro hlasování)
        method_weights = {
            'harmonic_template': 1.0,
            'cepstral': 0.8,
            'subharmonic': 0.9,
            'autocorr_piano': 0.7,
            'multi_pitch': 0.6
        }

        # Sběr všech kandidátů
        for method, results in methods_results.items():
            weight = method_weights.get(method, 0.5)
            method_conf = results.get('method_confidence', 0.0)

            for freq, confidence in results.get('candidates', []):
                final_confidence = confidence * weight * method_conf
                all_candidates.append((freq, final_confidence, method))

        if not all_candidates:
            return None

        grouped_candidates = self._group_similar_frequencies(all_candidates)
        if not grouped_candidates:
            return None

        # Pomocné: spočti jakou podporu má konkrétní frekvence v jednotlivých metodách
        def _method_support(freq_test: float, tol: float = 0.015) -> float:
            """Vrátí souhrnnou "hlasovou" podporu napříč metodami pro danou frekvenci.

            Tolerační práh defaultně 1.5% (dostatečné pro jemné odchylky klavírních frekvencí).
            """
            support = 0.0
            for method, results in methods_results.items():
                m_weight = method_weights.get(method, 0.5)
                m_conf = results.get('method_confidence', 0.0)
                for f, _ in results.get('candidates', []):
                    rel = abs(f - freq_test) / freq_test
                    if rel <= tol:
                        support += m_weight * m_conf
            return support

        # Pomocné: spočti podporu od harmonic_template pro více harmonických (2x,3x...)
        def _harmonic_template_hints(freq_test: float) -> float:
            hint = 0.0
            ht = methods_results.get('harmonic_template', {}).get('candidates', [])
            for f, score in ht:
                for h in (1, 2, 3, 4, 5):
                    target = freq_test * h
                    if abs(f - target) / target < 0.02:
                        # přičti vážené skóre (score z template je už spektrální energia)
                        hint += score
            return hint

        # Hledání nejlepší varianty mezi skupinami a jejich oktávami
        best_overall = None  # (chosen_freq, total_score, details)

        for group_freq, group_score, group_methods in grouped_candidates:
            variants = []
            for factor in (0.5, 1.0, 2.0):
                test_freq = group_freq * factor
                if not (self.fmin <= test_freq <= self.fmax):
                    continue

                # Skóre z metod (hlasování)
                vote_support = _method_support(test_freq)

                # Podpora z harmonic template (pokud nějaké tunele indikují harmoniky)
                ht_hint = _harmonic_template_hints(test_freq)

                # Agregované skóre: základní group score + váhy podpory
                total = group_score + 1.0 * vote_support + 0.8 * ht_hint

                variants.append((test_freq, total, group_score, vote_support, ht_hint, factor))

            if not variants:
                continue

            # Vyber nejlepší variantu pro tuto skupinu
            best_variant = max(variants, key=lambda x: x[1])

            # Pokud zatím nic převyšujícího, nastav první
            if best_overall is None or best_variant[1] > best_overall[1]:
                best_overall = best_variant

        if best_overall is None:
            return None

        chosen_freq, chosen_score, base_group_score, vote_support, ht_hint, factor = best_overall

        # Safety pravidlo: posun o oktávu se aplikuje jen pokud je zlepšení dostatečně velké.
        # Tedy pokud vybraná varianta je jiná než původní group_freq, musí mít >20% improvement.
        # Pokud zlepšení je malé, preferuj původní (méně riskantní) frekvenci.
        # Najdi původní group_freq (ten, který dal vznik candidate)
        # (pokud existuje přesná shoda, použij jí, jinak nech chosen_freq)

        # Logging pro diagnostiku
        logger.debug(
            f"Oktave-resolve: chosen={chosen_freq:.2f}Hz (factor={factor}), "
            f"score={chosen_score:.3f}, group={base_group_score:.3f}, votes={vote_support:.3f}, ht_hint={ht_hint:.3f}"
        )

        return chosen_freq

    def _group_similar_frequencies(self, candidates: List[Tuple[float, float, str]]) -> List[Tuple[float, float, List[str]]]:
        """Seskupování podobných frekvencí"""

        if not candidates:
            return []

        # Seřaď podle frekvence
        candidates.sort(key=lambda x: x[0])

        grouped = []
        current_group = [candidates[0]]

        for freq, confidence, method in candidates[1:]:
            # Tolerance 2% pro seskupování
            last_freq = current_group[-1][0]
            if abs(freq - last_freq) / last_freq < 0.02:
                current_group.append((freq, confidence, method))
            else:
                # Uzavři současnou skupinu
                if current_group:
                    group_freq, group_confidence, group_methods = self._finalize_group(current_group)
                    grouped.append((group_freq, group_confidence, group_methods))
                current_group = [(freq, confidence, method)]

        # Poslední skupina
        if current_group:
            group_freq, group_confidence, group_methods = self._finalize_group(current_group)
            grouped.append((group_freq, group_confidence, group_methods))

        return grouped

    def _finalize_group(self, group: List[Tuple[float, float, str]]) -> Tuple[float, float, List[str]]:
        """Finalizace skupiny kandidátů"""

        # Vážený průměr frekvencí
        total_weight = sum(conf for _, conf, _ in group)
        if total_weight > 0:
            weighted_freq = sum(freq * conf for freq, conf, _ in group) / total_weight
        else:
            weighted_freq = np.mean([freq for freq, _, _ in group])

        # Celková confidence
        total_confidence = sum(conf for _, conf, _ in group)

        # Bonus za více metod
        unique_methods = list(set(method for _, _, method in group))
        if len(unique_methods) > 1:
            total_confidence *= 1.3

        return weighted_freq, total_confidence, unique_methods

    def _detect_piano_harmonics(self, audio: np.ndarray, sr: int, fundamental: Optional[float]) -> List[float]:
        """Detekce harmonických specifická pro piano"""

        if fundamental is None:
            return []

        harmonics = []

        # Power spectrum
        spectrum = np.abs(np.fft.fft(audio, n=len(audio))) ** 2
        freqs = np.fft.fftfreq(len(audio), 1/sr)[:len(spectrum)//2]
        spectrum = spectrum[:len(spectrum)//2]

        # Piano má typicky silné harmonické 2, 3, slabší vyšší
        for harmonic_num in range(2, 12):  # až 11. harmonická
            harmonic_freq = fundamental * harmonic_num

            if harmonic_freq >= sr / 2:
                break

            # Ověř přítomnost harmonické
            if self._verify_piano_harmonic(spectrum, freqs, harmonic_freq, harmonic_num):
                harmonics.append(harmonic_freq)

        return harmonics

    def _verify_piano_harmonic(self, spectrum: np.ndarray, freqs: np.ndarray,
                              target_freq: float, harmonic_num: int) -> bool:
        """Verifikace harmonické pro piano"""

        # Najdi frequency bin
        freq_idx = np.argmin(np.abs(freqs - target_freq))

        # Tolerance window
        tolerance = max(1, int(0.03 * len(freqs)))  # 3% tolerance
        start_idx = max(0, freq_idx - tolerance)
        end_idx = min(len(spectrum), freq_idx + tolerance + 1)

        # Peak power v okně
        local_max = np.max(spectrum[start_idx:end_idx])

        # Background estimation
        background_power = np.median(spectrum)

        # Práh závislý na čísle harmonické (piano má klesající harmonické)
        expected_ratio = 1.0 / (harmonic_num ** 0.7)  # Piano charakteristika
        threshold = background_power * (3.0 + expected_ratio * 5.0)

        return local_max > threshold

    def _calculate_piano_confidence(self, fundamental: Optional[float], harmonics: List[float],
                                  methods_results: Dict, audio: np.ndarray, sr: int) -> float:
        """Výpočet confidence specifický pro piano"""

        if fundamental is None:
            return 0.0

        confidence = 0.3  # Base confidence

        # Bonus za harmonické
        if len(harmonics) >= 3:
            confidence += 0.4
        elif len(harmonics) >= 2:
            confidence += 0.3
        elif len(harmonics) >= 1:
            confidence += 0.2

        # Bonus za shodu s klavírními frekvencemi
        piano_distances = np.abs(self.piano_frequencies - fundamental)
        min_distance = np.min(piano_distances)

        if min_distance < fundamental * 0.01:  # 1% tolerance
            confidence += 0.2
        elif min_distance < fundamental * 0.02:  # 2% tolerance
            confidence += 0.1

        # Bonus za více metod
        method_count = sum(1 for results in methods_results.values()
                          if results.get('candidates', []))

        if method_count >= 4:
            confidence += 0.1
        elif method_count >= 3:
            confidence += 0.05

        # Piano range bonus
        if 80 <= fundamental <= 1000:  # Hlavní piano range
            confidence += 0.1
        elif 27.5 <= fundamental <= 2000:  # Rozšířený range
            confidence += 0.05

        return min(1.0, confidence)

    def _empty_result(self) -> PitchAnalysisResult:
        """Prázdný výsledek"""
        return PitchAnalysisResult(
            fundamental_freq=None,
            confidence=0.0,
            harmonics=[],
            method_used="piano_specialized",
            spectral_centroid=0.0,
            spectral_rolloff=0.0
        )
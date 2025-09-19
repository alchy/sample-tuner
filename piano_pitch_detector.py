"""
Hybridní pitch detector kombinující CREPE s tradičními algoritmy.
CREPE poskytuje state-of-the-art pitch detection, ostatní algoritmy validaci.
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
from audio_utils import PitchAnalysisResult, AudioUtils

logger = logging.getLogger(__name__)

# Dostupné knihovny
LIBRARIES = {
    'crepe': False,
    'aubio': False,
    'librosa': False
}

try:
    import crepe
    LIBRARIES['crepe'] = True
    logger.info("CREPE loaded successfully")
except ImportError as e:
    logger.warning(f"CREPE not available: {e}")

try:
    import aubio
    LIBRARIES['aubio'] = True
except ImportError:
    pass

try:
    import librosa
    LIBRARIES['librosa'] = True
except ImportError:
    pass


class CrepeHybridDetector:
    """
    Hybridní detector s CREPE jako primary + tradiční algoritmy pro validaci.

    Strategie:
    1. CREPE jako hlavní detector (nejvyšší přesnost)
    2. Tradiční algoritmy pro cross-validation
    3. Ensemble decision s váženým hlasováním
    4. Kontextová filtrace a oktáva korekce
    """

    def __init__(self, fmin: float = 27.5, fmax: float = 4186.0):
        self.fmin = fmin
        self.fmax = fmax
        self.piano_frequencies = self._generate_piano_frequencies()

        # Konfigurace CREPE
        self.crepe_model = 'large'  # 'tiny', 'small', 'medium', 'large', 'full'
        self.crepe_step_size = 10   # ms, trade-off rychlost/přesnost

        # Historie pro smoothing
        self.detection_history = []
        self.max_history = 3

        # Inicializace backup detektorů
        self.backup_detectors = self._initialize_backup_detectors()

        available = [lib for lib, avail in LIBRARIES.items() if avail]
        logger.info(f"Initialized CREPE hybrid detector with: {available}")

    def _generate_piano_frequencies(self) -> np.ndarray:
        """Generuje piano frekvence"""
        A4_freq = 440.0
        frequencies = []
        for midi in range(21, 109):  # A0 to C8
            freq = A4_freq * (2 ** ((midi - 69) / 12))
            if self.fmin <= freq <= self.fmax:
                frequencies.append(freq)
        return np.array(frequencies)

    def _initialize_backup_detectors(self) -> Dict:
        """Inicializuje backup detektory pro situace kdy CREPE selže"""
        detectors = {}

        # Aubio YIN jako nejspolehlivější backup
        if LIBRARIES['aubio']:
            try:
                detectors['aubio_yin'] = aubio.pitch("yin", 4096, 1024, 44100)
                detectors['aubio_yin'].set_unit("Hz")
                detectors['aubio_yin'].set_tolerance(0.8)
            except Exception as e:
                logger.warning(f"Aubio YIN init failed: {e}")

        # Vlastní HPS detector
        detectors['hps'] = self._hps_detector

        return detectors

    def detect(self, waveform: np.ndarray, sr: int) -> PitchAnalysisResult:
        """
        Hlavní detekční metoda s CREPE primary detection
        """
        # Preprocessing
        audio = self._preprocess_audio(waveform, sr)

        if len(audio) < 1024:
            return self._empty_result()

        # Primary detection s CREPE
        crepe_result = self._crepe_detection(audio, sr)

        # Backup detection pokud CREPE selže nebo má nízkou confidence
        backup_results = []
        if (not crepe_result['frequency'] or
            crepe_result['confidence'] < 0.5):

            logger.debug("Running backup detectors...")
            backup_results = self._run_backup_detectors(audio, sr)

        # Ensemble decision
        fundamental_freq = self._ensemble_decision(crepe_result, backup_results, audio, sr)

        # Aktualizuj historii pro temporal smoothing
        if fundamental_freq:
            self.detection_history.append(fundamental_freq)
            if len(self.detection_history) > self.max_history:
                self.detection_history.pop(0)

        # Post-processing s temporal smoothing
        if fundamental_freq and len(self.detection_history) > 1:
            fundamental_freq = self._temporal_smoothing(fundamental_freq)

        # Výpočet confidence
        confidence = self._calculate_confidence(fundamental_freq, crepe_result, backup_results)

        # Detekce harmonických
        harmonics = self._detect_harmonics(audio, sr, fundamental_freq) if fundamental_freq else []

        # Spektrální features
        centroid, rolloff = AudioUtils.spectral_features(audio, sr)

        # Logování pro debugging
        self._log_detection_details(crepe_result, backup_results, fundamental_freq, confidence)

        return PitchAnalysisResult(
            fundamental_freq=fundamental_freq,
            confidence=confidence,
            harmonics=harmonics,
            method_used="crepe_hybrid",
            spectral_centroid=centroid,
            spectral_rolloff=rolloff
        )

    def _crepe_detection(self, audio: np.ndarray, sr: int) -> Dict:
        """
        CREPE pitch detection s rozšířeným error handling a debugging
        """
        if not LIBRARIES['crepe']:
            return {'frequency': None, 'confidence': 0.0, 'method': 'crepe_unavailable'}

        try:
            # CREPE očekává audio jako 1D array
            if len(audio.shape) > 1:
                audio = audio.flatten()

            # Debugging info
            logger.debug(f"CREPE input: length={len(audio)}, sr={sr}, dtype={audio.dtype}")
            logger.debug(f"Audio stats: min={np.min(audio):.6f}, max={np.max(audio):.6f}, rms={np.sqrt(np.mean(audio**2)):.6f}")

            # CREPE potřebuje alespoň 1024 samples
            if len(audio) < 1024:
                logger.debug("Audio too short for CREPE")
                return {'frequency': None, 'confidence': 0.0, 'method': 'crepe_too_short'}

            # Zkrácení audio pokud je příliš dlouhé (CREPE může být pomalé)
            max_length = min(len(audio), sr * 10)  # Max 10 sekund
            if len(audio) > max_length:
                audio = audio[:max_length]
                logger.debug(f"Trimmed audio to {len(audio)} samples for CREPE")

            # CREPE detection s více parametry
            time_stamps, frequencies, confidences, _ = crepe.predict(
                audio,
                sr,
                model_capacity=self.crepe_model,
                step_size=self.crepe_step_size,
                verbose=1,  # Zapni verbose pro debugging
                center=True,
                viterbi=True  # Temporal smoothing
            )

            logger.debug(f"CREPE raw results: {len(frequencies)} detections")
            logger.debug(f"Frequency range: {np.min(frequencies):.2f} - {np.max(frequencies):.2f} Hz")
            logger.debug(f"Confidence range: {np.min(confidences):.3f} - {np.max(confidences):.3f}")

            # Filtrace na piano rozsah s nižším confidence threshold
            valid_mask = (frequencies >= self.fmin) & (frequencies <= self.fmax) & (confidences > 0.05)  # Sníženo z 0.1

            logger.debug(f"Valid detections (piano range + conf>0.05): {np.sum(valid_mask)}")

            if not np.any(valid_mask):
                # Zkus ještě méně přísnou filtraci
                loose_mask = (frequencies > 20) & (frequencies < 5000) & (confidences > 0.01)
                if np.any(loose_mask):
                    logger.debug(f"Loose filter found {np.sum(loose_mask)} detections:")
                    for i, (f, c) in enumerate(zip(frequencies[loose_mask][:5], confidences[loose_mask][:5])):
                        logger.debug(f"  {i}: {f:.2f} Hz (conf: {c:.3f})")

                return {'frequency': None, 'confidence': 0.0, 'method': 'crepe_no_valid'}

            valid_frequencies = frequencies[valid_mask]
            valid_confidences = confidences[valid_mask]

            # Log prvních několik výsledků pro debugging
            for i in range(min(5, len(valid_frequencies))):
                logger.debug(f"  Valid {i}: {valid_frequencies[i]:.2f} Hz (conf: {valid_confidences[i]:.3f})")

            # Robustnější výpočet finální frekvence
            if len(valid_frequencies) > 0:
                # Použij median pro odstranění outliers
                median_freq = np.median(valid_frequencies)

                # Filtruj hodnoty blízko mediánu
                median_mask = np.abs(valid_frequencies - median_freq) < median_freq * 0.1  # 10% od mediánu
                final_frequencies = valid_frequencies[median_mask]
                final_confidences = valid_confidences[median_mask]

                if len(final_frequencies) > 0:
                    # Vážený průměr podle confidence
                    weights = final_confidences / np.sum(final_confidences)
                    final_frequency = np.sum(final_frequencies * weights)
                    final_confidence = np.mean(final_confidences)
                else:
                    # Fallback na median
                    final_frequency = median_freq
                    final_confidence = np.mean(valid_confidences)

                logger.debug(f"CREPE final: {final_frequency:.2f} Hz (conf: {final_confidence:.3f})")

                return {
                    'frequency': final_frequency,
                    'confidence': final_confidence,
                    'method': 'crepe',
                    'raw_detections': len(frequencies),
                    'valid_detections': len(valid_frequencies),
                    'final_detections': len(final_frequencies) if 'final_frequencies' in locals() else len(valid_frequencies)
                }
            else:
                return {'frequency': None, 'confidence': 0.0, 'method': 'crepe_filtered'}

        except Exception as e:
            logger.error(f"CREPE detection failed: {e}")
            import traceback
            logger.debug(f"CREPE traceback: {traceback.format_exc()}")
            return {'frequency': None, 'confidence': 0.0, 'method': 'crepe_error'}

    def _run_backup_detectors(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Spustí backup detektory"""
        results = []

        # Aubio YIN
        if 'aubio_yin' in self.backup_detectors:
            try:
                freq = self._run_aubio_yin(audio, sr)
                if freq and self.fmin <= freq <= self.fmax:
                    results.append({
                        'frequency': freq,
                        'confidence': self.backup_detectors['aubio_yin'].get_confidence(),
                        'method': 'aubio_yin_backup'
                    })
            except Exception as e:
                logger.debug(f"Aubio YIN backup failed: {e}")

        # HPS detector
        try:
            hps_result = self._hps_detector(audio, sr)
            if hps_result['frequency'] and self.fmin <= hps_result['frequency'] <= self.fmax:
                results.append({
                    'frequency': hps_result['frequency'],
                    'confidence': hps_result['confidence'],
                    'method': 'hps_backup'
                })
        except Exception as e:
            logger.debug(f"HPS backup failed: {e}")

        return results

    def _ensemble_decision(self, crepe_result: Dict, backup_results: List[Dict],
                          audio: np.ndarray, sr: int) -> Optional[float]:
        """
        Ensemble decision s preferencí pro CREPE
        """
        all_results = []

        # Přidej CREPE result s vyšší váhou
        if crepe_result['frequency']:
            all_results.append({
                'frequency': crepe_result['frequency'],
                'confidence': crepe_result['confidence'],
                'weight': 2.0,  # CREPE má dvojnásobnou váhu
                'method': crepe_result['method']
            })

        # Přidej backup results
        for result in backup_results:
            all_results.append({
                'frequency': result['frequency'],
                'confidence': result['confidence'],
                'weight': 1.0,
                'method': result['method']
            })

        if not all_results:
            return None

        # Pokud máme jen jeden výsledek, vrať ho
        if len(all_results) == 1:
            return all_results[0]['frequency']

        # Seskup výsledky podle podobnosti (řeší oktávové chyby)
        groups = self._group_similar_frequencies(all_results)

        # Najdi nejlepší skupinu
        best_group = self._select_best_group(groups)

        if not best_group:
            return all_results[0]['frequency']  # Fallback

        # Vážený průměr v rámci nejlepší skupiny
        frequencies = [r['frequency'] for r in best_group]
        weights = [r['confidence'] * r['weight'] for r in best_group]

        total_weight = sum(weights)
        if total_weight > 0:
            return sum(f * w for f, w in zip(frequencies, weights)) / total_weight
        else:
            return frequencies[0]

    def _group_similar_frequencies(self, results: List[Dict]) -> List[List[Dict]]:
        """Seskupí podobné frekvence s inteligentní oktáva korekcí"""
        groups = []
        tolerance = 0.05  # 5% tolerance

        for result in results:
            freq = result['frequency']
            placed = False

            # Zkus zařadit do existující skupiny
            for group in groups:
                group_freqs = [r['frequency'] for r in group]
                group_avg = np.mean(group_freqs)

                # Kontrola přímé podobnosti
                if abs(freq - group_avg) / group_avg < tolerance:
                    group.append(result)
                    placed = True
                    break

                # Kontrola oktávových vztahů (zdvojení/půlení frekvence)
                oktava_ratios = [0.5, 2.0, 0.25, 4.0]  # 1, 2, 3 oktávy dolů/nahoru
                for ratio in oktava_ratios:
                    target_freq = group_avg * ratio
                    if abs(freq - target_freq) / target_freq < tolerance:
                        # Detekována oktávová chyba - oprav frekvenci
                        logger.debug(f"Oktáva korekce: {freq:.2f} Hz -> {group_avg:.2f} Hz (ratio: {ratio})")

                        # Korekce frekvence v result
                        corrected_result = result.copy()
                        corrected_result['frequency'] = group_avg  # Použij average skupiny
                        corrected_result['oktava_corrected'] = True
                        corrected_result['original_frequency'] = freq

                        group.append(corrected_result)
                        placed = True
                        break

                if placed:
                    break

            # Vytvoř novou skupinu
            if not placed:
                groups.append([result])

        # Debug info o skupinách
        for i, group in enumerate(groups):
            freqs_str = ", ".join([f"{r['frequency']:.1f}Hz" for r in group])
            methods_str = ", ".join([r['method'] for r in group])
            logger.debug(f"Group {i}: [{freqs_str}] from [{methods_str}]")

        return groups

    def _select_best_group(self, groups: List[List[Dict]]) -> Optional[List[Dict]]:
        """Vybere nejlepší skupinu s preferencí pro CREPE a rozumné frekvence"""
        if not groups:
            return None

        best_group = None
        best_score = 0

        for group in groups:
            # Základní metriky
            total_weight = sum(r.get('weight', 1.0) for r in group)
            avg_confidence = np.mean([r['confidence'] for r in group])
            detector_count = len(group)

            # Silný bonus za CREPE v skupině
            crepe_bonus = 0
            has_crepe = any('crepe' in r['method'] for r in group)
            if has_crepe:
                crepe_bonus = 3.0  # Velmi silný bonus pro CREPE

            # Penalizace pro velmi nízké nebo vysoké frekvence (pravděpodobné chyby)
            freq_penalty = 0
            group_avg_freq = np.mean([r['frequency'] for r in group])

            # Piano má většinu energie mezi 80-2000 Hz
            if group_avg_freq < 80:
                freq_penalty = -2.0  # Penalizace pro příliš nízké frekvence
                logger.debug(f"Low frequency penalty for {group_avg_freq:.1f} Hz")
            elif group_avg_freq > 2000:
                freq_penalty = -1.0  # Menší penalizace pro vysoké frekvence
                logger.debug(f"High frequency penalty for {group_avg_freq:.1f} Hz")

            # Bonus pro "rozumné" piano frekvence (200-1000 Hz)
            if 200 <= group_avg_freq <= 1000:
                freq_penalty = 0.5  # Bonus pro střední frekvence

            # Kombinovaný score
            score = (
                total_weight * 0.3 +
                avg_confidence * 0.3 +
                detector_count * 0.2 +
                crepe_bonus +
                freq_penalty
            )

            # Debug info
            methods = [r['method'] for r in group]
            logger.debug(f"Group score: {score:.2f} (freq: {group_avg_freq:.1f}Hz, "
                        f"methods: {methods}, crepe: {has_crepe}, conf: {avg_confidence:.3f})")

            if score > best_score:
                best_score = score
                best_group = group

        return best_group

    def _temporal_smoothing(self, current_freq: float) -> float:
        """Temporal smoothing na základě historie"""
        if len(self.detection_history) < 2:
            return current_freq

        # Jednoduchý exponential smoothing
        alpha = 0.7  # Váha pro current detection
        smoothed = alpha * current_freq + (1 - alpha) * self.detection_history[-2]

        # Pokud je rozdíl příliš velký, preferuj current detection
        if abs(smoothed - current_freq) > current_freq * 0.1:  # 10% threshold
            return current_freq
        else:
            return smoothed

    def _calculate_confidence(self, freq: Optional[float], crepe_result: Dict,
                            backup_results: List[Dict]) -> float:
        """Vypočítá celkovou confidence"""
        if not freq:
            return 0.0

        confidence = 0.0

        # Base confidence z CREPE (má největší váhu)
        if crepe_result['frequency'] and abs(crepe_result['frequency'] - freq) < freq * 0.02:
            confidence += 0.7 * crepe_result['confidence']

        # Confidence z backup detektorů
        agreeing_backups = [r for r in backup_results
                          if abs(r['frequency'] - freq) < freq * 0.02]

        if agreeing_backups:
            backup_conf = np.mean([r['confidence'] for r in agreeing_backups])
            confidence += 0.3 * backup_conf

        # Bonus za více souhlasících detektorů
        total_agreeing = len(agreeing_backups)
        if crepe_result['frequency'] and abs(crepe_result['frequency'] - freq) < freq * 0.02:
            total_agreeing += 1

        agreement_bonus = min(0.2, total_agreeing * 0.1)
        confidence += agreement_bonus

        return min(1.0, confidence)

    def _detect_harmonics(self, audio: np.ndarray, sr: int, fundamental: float) -> List[float]:
        """Detekce harmonických s validací"""
        harmonics = []
        spectrum, freqs = AudioUtils.compute_spectrum(audio, sr, n_fft=8192)

        if len(spectrum) == 0:
            return harmonics

        fund_idx = np.argmin(np.abs(freqs - fundamental))
        fund_energy = spectrum[fund_idx]

        for h in range(2, 8):
            harmonic_freq = fundamental * h
            if harmonic_freq >= sr / 2:
                break

            harm_idx = np.argmin(np.abs(freqs - harmonic_freq))
            harm_energy = spectrum[harm_idx]

            # Dynamický threshold podle harmonické
            threshold = 0.15 / h  # Vyšší harmonické mají nižší threshold

            if harm_energy > threshold * fund_energy:
                harmonics.append(harmonic_freq)

        return harmonics

    def _hps_detector(self, audio: np.ndarray, sr: int) -> Dict:
        """Harmonic Product Spectrum detector jako backup"""
        n_fft = min(8192, len(audio))
        windowed = audio[:n_fft] * np.hanning(n_fft)

        spectrum = np.abs(np.fft.fft(windowed))
        freqs = np.fft.fftfreq(n_fft, 1/sr)

        half_len = n_fft // 2
        spectrum = spectrum[:half_len]
        freqs = freqs[:half_len]

        # HPS calculation
        hps = spectrum.copy()
        for h in range(2, 6):
            downsampled = spectrum[::h]
            min_len = min(len(hps), len(downsampled))
            hps[:min_len] *= downsampled[:min_len]

        # Najdi peak v piano rozsahu
        valid_mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        if not np.any(valid_mask):
            return {'frequency': None, 'confidence': 0.0}

        valid_hps = hps[valid_mask]
        valid_freqs = freqs[valid_mask]

        peak_idx = np.argmax(valid_hps)
        frequency = valid_freqs[peak_idx]

        # Confidence
        peak_height = valid_hps[peak_idx]
        mean_hps = np.mean(valid_hps)
        confidence = min(1.0, peak_height / (mean_hps * 3)) if mean_hps > 0 else 0

        return {'frequency': frequency, 'confidence': confidence}

    def _run_aubio_yin(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """Spustí Aubio YIN"""
        detector = self.backup_detectors['aubio_yin']

        if sr != 44100:
            # Resample
            ratio = 44100 / sr
            new_length = int(len(audio) * ratio)
            audio = np.interp(np.linspace(0, len(audio)-1, new_length),
                            np.arange(len(audio)), audio)

        audio = audio.astype(np.float32)

        hop_size = 1024
        frequencies = []

        for i in range(0, len(audio), hop_size):
            window = audio[i:i + hop_size]
            if len(window) < hop_size:
                window = np.pad(window, (0, hop_size - len(window)), 'constant')

            freq = detector(window)[0]
            if freq > 0:
                frequencies.append(freq)

        if frequencies:
            return np.median(frequencies)
        else:
            return None

    def _preprocess_audio(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Audio preprocessing optimalizované pro CREPE"""
        audio = AudioUtils.to_mono(waveform).astype(np.float32)

        # CREPE pracuje nejlépe s normalizovaným audio
        audio = AudioUtils.normalize_audio(audio, target_db=-20.0).astype(np.float32)

        # Gentle high-pass filter pro odstranění DC
        if len(audio) > 100:
            alpha = 0.98
            filtered = np.zeros_like(audio)
            filtered[0] = audio[0]
            for i in range(1, len(audio)):
                filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1])
            audio = filtered.astype(np.float32)

        return audio

    def _log_detection_details(self, crepe_result: Dict, backup_results: List[Dict],
                             final_freq: Optional[float], confidence: float) -> None:
        """Logování pro debugging"""
        if logger.level <= logging.DEBUG:
            logger.debug(f"CREPE: {crepe_result['frequency']:.2f} Hz (conf: {crepe_result['confidence']:.3f})"
                        if crepe_result['frequency'] else "CREPE: No detection")

            for result in backup_results:
                logger.debug(f"{result['method']}: {result['frequency']:.2f} Hz (conf: {result['confidence']:.3f})")

            logger.debug(f"Final: {final_freq:.2f} Hz (conf: {confidence:.3f})" if final_freq else "Final: No detection")

    def _empty_result(self) -> PitchAnalysisResult:
        """Prázdný výsledek"""
        return PitchAnalysisResult(
            fundamental_freq=None,
            confidence=0.0,
            harmonics=[],
            method_used="crepe_hybrid_failed",
            spectral_centroid=0.0,
            spectral_rolloff=0.0
        )
"""
Velocity analýza a globální mapping s podporou stereo handling.
Obsahuje pokročilé metriky pro velocity detection.
"""

import numpy as np
import logging
from typing import List, Dict, Optional
from collections import defaultdict
from audio_utils import VelocityAnalysisResult, AudioSampleData, AudioProcessor

logger = logging.getLogger(__name__)


class AdvancedVelocityAnalyzer:
    """
    Pokročilý velocity analyzer s stereo podporou.

    Vylepšení:
    - Lepší handling stereo vzorků
    - Více metrik pro velocity detection
    - Vektorová matematika
    """

    def __init__(self, attack_duration: float = 0.5):
        self.attack_duration = attack_duration

    def analyze(self, waveform: np.ndarray, sr: int) -> VelocityAnalysisResult:
        """Kompletní velocity analýza"""

        # Zachovej stereo info, ale získej i mono verzi
        mono_audio = waveform.flatten() if len(waveform.shape) > 1 else waveform

        # Stereo width calculation
        stereo_width = AudioProcessor.calculate_stereo_width(waveform)

        # RMS metrics - použij mono pro konzistenci
        rms_db = self._calculate_rms_db(mono_audio)

        # Peak metrics
        peak_db = self._calculate_peak_db(mono_audio)

        # Attack phase metrics
        attack_samples = int(sr * self.attack_duration)
        attack_samples = min(attack_samples, len(mono_audio))

        if attack_samples > 0:
            attack_section = mono_audio[:attack_samples]
            attack_peak_db = self._calculate_peak_db(attack_section)
            attack_rms_db = self._calculate_rms_db(attack_section)
        else:
            attack_peak_db = peak_db
            attack_rms_db = rms_db

        # Dynamic range
        dynamic_range = peak_db - rms_db if rms_db != -np.inf else 0

        return VelocityAnalysisResult(
            rms_db=rms_db,
            peak_db=peak_db,
            attack_peak_db=attack_peak_db,
            attack_rms_db=attack_rms_db,
            dynamic_range=dynamic_range,
            stereo_width=stereo_width
        )

    def _calculate_rms_db(self, audio: np.ndarray) -> float:
        """RMS výpočet"""
        if len(audio) == 0:
            return -np.inf

        # Vectorized RMS calculation
        rms_squared = np.mean(audio ** 2)

        if rms_squared <= 1e-10:  # Ochrana proti velmi tichým signálům
            return -np.inf

        return 10 * np.log10(rms_squared)  # 10*log10 pro power, 20*log10 by bylo pro amplitudu

    def _calculate_peak_db(self, audio: np.ndarray) -> float:
        """Peak výpočet"""
        if len(audio) == 0:
            return -np.inf

        # Vectorized peak calculation
        peak = np.max(np.abs(audio))

        if peak <= 1e-10:  # Ochrana proti velmi tichým signálům
            return -np.inf

        return 20 * np.log10(peak)  # 20*log10 pro amplitudu

    def analyze_attack_curve(self, waveform: np.ndarray, sr: int) -> Dict:
        """
        Pokročilá analýza attack křivky pro velocity detection.
        Vrací detailní metriky o nástupu zvuku.
        """
        mono_audio = waveform.flatten() if len(waveform.shape) > 1 else waveform

        # Attack window
        attack_samples = int(sr * self.attack_duration)
        attack_samples = min(attack_samples, len(mono_audio))

        if attack_samples < 10:
            return {"attack_slope": 0.0, "attack_time": 0.0, "attack_sharpness": 0.0}

        attack_section = mono_audio[:attack_samples]

        # Najdi skutečný začátek signálu (nad noise floor)
        abs_attack = np.abs(attack_section)
        noise_floor = np.percentile(abs_attack, 10)  # 10% percentil jako noise floor

        # Najdi první vzorky nad noise floor
        signal_start = np.where(abs_attack > noise_floor * 3)[0]
        if len(signal_start) == 0:
            return {"attack_slope": 0.0, "attack_time": 0.0, "attack_sharpness": 0.0}

        start_idx = signal_start[0]

        # Najdi peak
        peak_idx = np.argmax(abs_attack[start_idx:]) + start_idx

        if peak_idx <= start_idx:
            return {"attack_slope": 0.0, "attack_time": 0.0, "attack_sharpness": 0.0}

        # Attack time (čas do peaku)
        attack_time = (peak_idx - start_idx) / sr

        # Attack slope (strmost nárůstu)
        if attack_time > 0:
            peak_value = abs_attack[peak_idx]
            start_value = abs_attack[start_idx]
            attack_slope = (peak_value - start_value) / attack_time
        else:
            attack_slope = 0.0

        # Attack sharpness (druhá derivace)
        if peak_idx > start_idx + 1:
            attack_curve = abs_attack[start_idx:peak_idx+1]
            if len(attack_curve) > 2:
                # Druhá derivace jako míra "ostrosti" attack
                first_diff = np.diff(attack_curve)
                second_diff = np.diff(first_diff)
                attack_sharpness = np.mean(np.abs(second_diff))
            else:
                attack_sharpness = 0.0
        else:
            attack_sharpness = 0.0

        return {
            "attack_slope": attack_slope,
            "attack_time": attack_time,
            "attack_sharpness": attack_sharpness,
            "start_idx": start_idx,
            "peak_idx": peak_idx
        }


class OptimizedVelocityMapper:
    """
    Globální velocity mapper s pokročilými metrikami.

    Vylepšení:
    - Podporuje více metrik současně
    - Adaptivní thresholding
    - Lepší distribuce velocity hodnot
    """

    @staticmethod
    def create_advanced_mapping(samples: List[AudioSampleData],
                              primary_metric: str = "attack_peak_db",
                              secondary_metric: Optional[str] = "attack_slope",
                              num_velocities: int = 8) -> Dict:
        """
        Vytvoří pokročilý velocity mapping s multiple metriky.

        Args:
            samples: Seznam vzorků
            primary_metric: Hlavní metrika pro velocity
            secondary_metric: Sekundární metrika pro tie-breaking
            num_velocities: Počet velocity úrovní
        """

        # Extrakce primární metriky
        primary_values = []
        secondary_values = []

        # Pro advanced metriky budeme potřebovat analyzer
        velocity_analyzer = AdvancedVelocityAnalyzer()

        for sample in samples:
            if sample.velocity_analysis is None:
                continue

            # Primární metrika
            primary_value = OptimizedVelocityMapper._extract_metric_value(
                sample, primary_metric, velocity_analyzer
            )

            if primary_value is not None and primary_value != -np.inf:
                primary_values.append(primary_value)

                # Sekundární metrika pro tie-breaking
                if secondary_metric:
                    secondary_value = OptimizedVelocityMapper._extract_metric_value(
                        sample, secondary_metric, velocity_analyzer
                    )
                    secondary_values.append(secondary_value if secondary_value is not None else 0.0)
                else:
                    secondary_values.append(0.0)

        if len(primary_values) < 2:
            logger.warning("Nedostatek dat pro velocity mapping")
            return {
                "thresholds": [],
                "min_value": 0,
                "max_value": 0,
                "primary_metric": primary_metric,
                "secondary_metric": secondary_metric
            }

        # Adaptivní thresholding - kombinace percentilů a uniform distribution
        primary_values = np.array(primary_values)

        # Odstranění outliers (5% z každé strany)
        p5, p95 = np.percentile(primary_values, [5, 95])
        filtered_values = primary_values[(primary_values >= p5) & (primary_values <= p95)]

        if len(filtered_values) < 2:
            filtered_values = primary_values

        min_value = np.min(filtered_values)
        max_value = np.max(filtered_values)

        # Vytvoř thresholdy - kombinace uniform a percentile-based
        thresholds = []

        if num_velocities <= 1:
            thresholds = [min_value]
        else:
            # Hybrid approach: část uniform, část percentile-based
            uniform_thresholds = np.linspace(min_value, max_value, num_velocities)
            percentile_points = np.linspace(0, 100, num_velocities)
            percentile_thresholds = np.percentile(filtered_values, percentile_points)

            # Váha pro kombinaci (více uniform pro malé datasety)
            uniform_weight = 0.7 if len(filtered_values) < 50 else 0.3

            for i in range(num_velocities):
                threshold = (uniform_weight * uniform_thresholds[i] +
                           (1 - uniform_weight) * percentile_thresholds[i])
                thresholds.append(threshold)

        return {
            "thresholds": thresholds,
            "min_value": float(min_value),
            "max_value": float(max_value),
            "primary_metric": primary_metric,
            "secondary_metric": secondary_metric,
            "num_samples": len(primary_values),
            "outliers_removed": len(primary_values) - len(filtered_values)
        }

    @staticmethod
    def _extract_metric_value(sample: AudioSampleData, metric: str,
                            velocity_analyzer: AdvancedVelocityAnalyzer) -> Optional[float]:
        """Extrahuje hodnotu metriky ze vzorku"""

        if metric == "attack_peak_db":
            return sample.velocity_analysis.attack_peak_db
        elif metric == "peak_db":
            return sample.velocity_analysis.peak_db
        elif metric == "rms_db":
            return sample.velocity_analysis.rms_db
        elif metric == "dynamic_range":
            return sample.velocity_analysis.dynamic_range
        elif metric == "stereo_width":
            return sample.velocity_analysis.stereo_width
        elif metric in ["attack_slope", "attack_time", "attack_sharpness"]:
            # Pro advanced metriky potřebujeme přepočítat
            try:
                attack_analysis = velocity_analyzer.analyze_attack_curve(
                    sample.waveform, sample.sample_rate
                )
                return attack_analysis.get(metric, 0.0)
            except Exception as e:
                logger.warning(f"Failed to compute {metric}: {e}")
                return 0.0
        else:
            logger.warning(f"Unknown metric: {metric}")
            return None

    @staticmethod
    def assign_velocity_advanced(sample: AudioSampleData, mapping: Dict,
                               velocity_analyzer: Optional[AdvancedVelocityAnalyzer] = None) -> int:
        """
        Přiřazení velocity s multiple metrikami.
        """
        if not mapping["thresholds"]:
            return 0

        if velocity_analyzer is None:
            velocity_analyzer = AdvancedVelocityAnalyzer()

        # Primární metrika
        primary_value = OptimizedVelocityMapper._extract_metric_value(
            sample, mapping["primary_metric"], velocity_analyzer
        )

        if primary_value is None or primary_value == -np.inf:
            return 0

        # Najdi velocity podle primární metriky
        velocity = 0
        for i, threshold in enumerate(mapping["thresholds"]):
            if primary_value >= threshold:
                velocity = i

        # Sekundární metrika pro fine-tuning (pokud jsou dva vzorky blízko)
        if mapping.get("secondary_metric"):
            secondary_value = OptimizedVelocityMapper._extract_metric_value(
                sample, mapping["secondary_metric"], velocity_analyzer
            )

            if secondary_value is not None:
                # Pokud jsme na hranici mezi dvěma velocity, použij sekundární metriku
                threshold_tolerance = 0.05  # 5% tolerance

                if velocity < len(mapping["thresholds"]) - 1:
                    current_threshold = mapping["thresholds"][velocity]
                    next_threshold = mapping["thresholds"][velocity + 1]
                    threshold_range = next_threshold - current_threshold

                    # Pokud jsme blízko následujícímu prahu
                    if (next_threshold - primary_value) < (threshold_tolerance * threshold_range):
                        # Pokud sekundární metrika je vysoká, zvyš velocity
                        if secondary_value > 0.5:  # Arbitrary threshold
                            velocity = min(velocity + 1, len(mapping["thresholds"]) - 1)

        return min(velocity, len(mapping["thresholds"]) - 1)

    @staticmethod
    def analyze_velocity_distribution(samples: List[AudioSampleData], mapping: Dict) -> Dict:
        """Analýza distribuce velocity hodnot pro debugging"""

        velocity_counts = defaultdict(int)
        velocity_values = defaultdict(list)  # Pro statistiky hodnot v každé velocity

        velocity_analyzer = AdvancedVelocityAnalyzer()

        for sample in samples:
            if sample.velocity_analysis is None:
                continue

            velocity = OptimizedVelocityMapper.assign_velocity_advanced(
                sample, mapping, velocity_analyzer
            )

            primary_value = OptimizedVelocityMapper._extract_metric_value(
                sample, mapping["primary_metric"], velocity_analyzer
            )

            velocity_counts[velocity] += 1
            if primary_value is not None and primary_value != -np.inf:
                velocity_values[velocity].append(primary_value)

        # Statistiky pro každou velocity
        velocity_stats = {}
        for vel in range(len(mapping["thresholds"])):
            count = velocity_counts[vel]
            values = velocity_values[vel]

            if values:
                velocity_stats[vel] = {
                    "count": count,
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
            else:
                velocity_stats[vel] = {
                    "count": 0,
                    "mean": 0,
                    "std": 0,
                    "min": 0,
                    "max": 0
                }

        return {
            "velocity_counts": dict(velocity_counts),
            "velocity_stats": velocity_stats,
            "total_samples": len(samples),
            "mapping_info": mapping
        }
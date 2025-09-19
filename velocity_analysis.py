"""
Modul pro pokročilou velocity analýzu a mapping s podporou stereo.
Obsahuje analyzátor pro metriky jako RMS, peak a attack křivku, a mapper pro globální velocity úrovně.
"""

import numpy as np
import logging
from typing import List, Dict, Optional
from collections import defaultdict
from audio_utils import VelocityAnalysisResult, AudioSampleData, AudioProcessor

logger = logging.getLogger(__name__)

class AdvancedVelocityAnalyzer:
    """
    Pokročilý analyzer velocity s podporou stereo a attack fáze.
    Vypočítává metriky pro mapping, jako RMS, peak a dynamický rozsah.
    """
    def __init__(self, attack_duration: float = 0.5):
        self.attack_duration = attack_duration  # Délka attack fáze v sekundách

    def analyze(self, waveform: np.ndarray, sr: int) -> VelocityAnalysisResult:
        """
        Kompletní velocity analýza waveformu, včetně mono konverze pro konzistenci.
        """
        # Převedení na mono pro analýzu
        mono_audio = waveform.flatten() if len(waveform.shape) > 1 else waveform

        # Výpočet stereo šířky
        stereo_width = AudioProcessor.calculate_stereo_width(waveform)

        # RMS a peak metriky
        rms_db = self._calculate_rms_db(mono_audio)
        peak_db = self._calculate_peak_db(mono_audio)

        # Attack fáze metriky
        attack_samples = int(sr * self.attack_duration)
        attack_samples = min(attack_samples, len(mono_audio))
        if attack_samples > 0:
            attack_section = mono_audio[:attack_samples]
            attack_peak_db = self._calculate_peak_db(attack_section)
            attack_rms_db = self._calculate_rms_db(attack_section)
        else:
            attack_peak_db = peak_db
            attack_rms_db = rms_db

        # Dynamický rozsah
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
        """
        Vypočítá RMS v dB s ochranou proti nulovým signálům.
        """
        if len(audio) == 0:
            return -np.inf
        rms_squared = np.mean(audio ** 2)
        if rms_squared <= 1e-10:
            return -np.inf
        return 10 * np.log10(rms_squared)

    def _calculate_peak_db(self, audio: np.ndarray) -> float:
        """
        Vypočítá peak v dB s ochranou proti nulovým signálům.
        """
        if len(audio) == 0:
            return -np.inf
        peak = np.max(np.abs(audio))
        if peak <= 1e-10:
            return -np.inf
        return 20 * np.log10(peak)

    def analyze_attack_curve(self, waveform: np.ndarray, sr: int) -> Dict:
        """
        Pokročilá analýza attack křivky pro velocity, včetně slope a sharpness.
        """
        mono_audio = waveform.flatten() if len(waveform.shape) > 1 else waveform

        # Attack window
        attack_samples = int(sr * self.attack_duration)
        attack_samples = min(attack_samples, len(mono_audio))

        if attack_samples < 10:
            return {"attack_slope": 0.0, "attack_time": 0.0, "attack_sharpness": 0.0}

        attack_section = mono_audio[:attack_samples]

        # Najdi začátek signálu nad noise floor
        abs_attack = np.abs(attack_section)
        noise_floor = np.percentile(abs_attack, 10)
        signal_start = np.where(abs_attack > noise_floor * 3)[0]
        if len(signal_start) == 0:
            return {"attack_slope": 0.0, "attack_time": 0.0, "attack_sharpness": 0.0}

        start_idx = signal_start[0]
        peak_idx = np.argmax(abs_attack[start_idx:]) + start_idx

        if peak_idx <= start_idx:
            return {"attack_slope": 0.0, "attack_time": 0.0, "attack_sharpness": 0.0}

        # Attack time a slope
        attack_time = (peak_idx - start_idx) / sr
        peak_value = abs_attack[peak_idx]
        start_value = abs_attack[start_idx]
        attack_slope = (peak_value - start_value) / attack_time if attack_time > 0 else 0.0

        # Attack sharpness (druhá derivace)
        attack_sharpness = 0.0
        if peak_idx > start_idx + 1:
            attack_curve = abs_attack[start_idx:peak_idx+1]
            if len(attack_curve) > 2:
                first_diff = np.diff(attack_curve)
                second_diff = np.diff(first_diff)
                attack_sharpness = np.mean(np.abs(second_diff))

        return {
            "attack_slope": attack_slope,
            "attack_time": attack_time,
            "attack_sharpness": attack_sharpness,
            "start_idx": start_idx,
            "peak_idx": peak_idx
        }

class OptimizedVelocityMapper:
    """
    Globální mapper velocity s podporou multiple metrik a adaptivního thresholding.
    """
    @staticmethod
    def create_advanced_mapping(samples: List[AudioSampleData],
                              primary_metric: str = "attack_peak_db",
                              secondary_metric: Optional[str] = "attack_slope",
                              num_velocities: int = 8) -> Dict:
        """
        Vytvoří pokročilý velocity mapping s multiple metrikami a outlier removal.
        """
        primary_values = []
        secondary_values = []
        velocity_analyzer = AdvancedVelocityAnalyzer()

        for sample in samples:
            if sample.velocity_analysis is None:
                continue

            primary_value = OptimizedVelocityMapper._extract_metric_value(
                sample, primary_metric, velocity_analyzer
            )
            if primary_value is not None and primary_value != -np.inf:
                primary_values.append(primary_value)

                secondary_value = OptimizedVelocityMapper._extract_metric_value(
                    sample, secondary_metric, velocity_analyzer
                ) if secondary_metric else 0.0
                secondary_values.append(secondary_value if secondary_value is not None else 0.0)

        if len(primary_values) < 2:
            logger.warning("Nedostatek dat pro velocity mapping")
            return {"thresholds": [], "min_value": 0, "max_value": 0, "primary_metric": primary_metric, "secondary_metric": secondary_metric}

        primary_values = np.array(primary_values)

        # Odstranění outliers (5% z každé strany)
        p5, p95 = np.percentile(primary_values, [5, 95])
        filtered_values = primary_values[(primary_values >= p5) & (primary_values <= p95)]

        if len(filtered_values) < 2:
            filtered_values = primary_values

        min_value = float(np.min(filtered_values))
        max_value = float(np.max(filtered_values))

        # Hybridní thresholds (uniform + percentile)
        uniform_thresholds = np.linspace(min_value, max_value, num_velocities)
        percentile_points = np.linspace(0, 100, num_velocities)
        percentile_thresholds = np.percentile(filtered_values, percentile_points)

        uniform_weight = 0.7 if len(filtered_values) < 50 else 0.3
        thresholds = (uniform_weight * uniform_thresholds + (1 - uniform_weight) * percentile_thresholds)

        return {
            "thresholds": thresholds.tolist(),
            "min_value": min_value,
            "max_value": max_value,
            "primary_metric": primary_metric,
            "secondary_metric": secondary_metric,
            "num_samples": len(primary_values),
            "outliers_removed": len(primary_values) - len(filtered_values)
        }

    @staticmethod
    def _extract_metric_value(sample: AudioSampleData, metric: str,
                            velocity_analyzer: AdvancedVelocityAnalyzer) -> Optional[float]:
        """
        Extrahuje hodnotu metriky ze sample, včetně pokročilých attack metrik.
        """
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
            attack_analysis = velocity_analyzer.analyze_attack_curve(
                sample.waveform, sample.sample_rate
            )
            return attack_analysis.get(metric, 0.0)
        else:
            logger.warning(f"Unknown metric: {metric}")
            return None

    @staticmethod
    def assign_velocity_advanced(sample: AudioSampleData, mapping: Dict,
                               velocity_analyzer: Optional[AdvancedVelocityAnalyzer] = None) -> int:
        """
        Přiřadí velocity na základě primary a secondary metriky s fine-tuningem.
        """
        if not mapping["thresholds"]:
            return 0

        if velocity_analyzer is None:
            velocity_analyzer = AdvancedVelocityAnalyzer()

        primary_value = OptimizedVelocityMapper._extract_metric_value(
            sample, mapping["primary_metric"], velocity_analyzer
        )

        if primary_value is None or primary_value == -np.inf:
            return 0

        velocity = np.searchsorted(mapping["thresholds"], primary_value)

        # Fine-tuning s secondary metrikou
        if mapping.get("secondary_metric"):
            secondary_value = OptimizedVelocityMapper._extract_metric_value(
                sample, mapping["secondary_metric"], velocity_analyzer
            )
            if secondary_value is not None and velocity < len(mapping["thresholds"]) - 1:
                current_threshold = mapping["thresholds"][velocity]
                next_threshold = mapping["thresholds"][velocity + 1]
                threshold_range = next_threshold - current_threshold
                if (next_threshold - primary_value) < 0.05 * threshold_range and secondary_value > 0.5:
                    velocity += 1

        return min(velocity, len(mapping["thresholds"]) - 1)

    @staticmethod
    def analyze_velocity_distribution(samples: List[AudioSampleData], mapping: Dict) -> Dict:
        """
        Analýza distribuce velocity pro debugging a statistiky.
        """
        velocity_counts = defaultdict(int)
        velocity_values = defaultdict(list)

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

        velocity_stats = {}
        for vel in range(len(mapping["thresholds"])):
            values = velocity_values[vel]
            velocity_stats[vel] = {
                "count": velocity_counts[vel],
                "mean": np.mean(values) if values else 0,
                "std": np.std(values) if values else 0,
                "min": np.min(values) if values else 0,
                "max": np.max(values) if values else 0
            }

        return {
            "velocity_counts": dict(velocity_counts),
            "velocity_stats": velocity_stats,
            "total_samples": len(samples),
            "mapping_info": mapping
        }
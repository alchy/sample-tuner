"""
Hlavní orchestrace pitch correctoru s dvou-fázovým zpracováním.
Fáze 1: Rychlá analýza amplitud pro velocity mapping.
Fáze 2: Plná pitch detection a korekce.
Zpracovává soubory podle předchozí implementace: glob pro podporované formáty, validace metadat, rychlá amplitude analýza.
"""

import argparse
import soundfile as sf
import numpy as np
from collections import defaultdict
from pathlib import Path
import logging
import sys
from typing import Optional, List, Dict
from scipy.signal import resample  # Pro pitch shift resamplování

# Importy z modulů (předpokládám existenci)
from audio_utils import AudioSampleData, AudioUtils, AudioProcessor, DirectionalTuner, TuneDirection, KeyFilter
from piano_pitch_detector import CrepeHybridDetector
from velocity_analysis import AdvancedVelocityAnalyzer

# Default konfigurace
TUNE_DIRECTION = TuneDirection.NEAREST
KEY_FILTER = KeyFilter.ALL

# Nastavení loggingu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioSampleLite:
    """
    Zjednodušená struktura pro fázi 1: obsahuje jen základní info o souboru pro amplitude analýzu.
    """
    def __init__(self, filepath: Path, max_amplitude: float, rms_amplitude: float,
                 duration: float, sample_rate: int):
        self.filepath = filepath  # Cesta k souboru
        self.max_amplitude = max_amplitude  # Maximální amplitude
        self.rms_amplitude = rms_amplitude  # RMS amplitude
        self.duration = duration  # Délka souboru
        self.sample_rate = sample_rate  # Sample rate
        self.assigned_velocity = None  # Přiřazená velocity (nastaví se později)

class TwoPhaseRefactoredPitchCorrector:
    """
    Hlavní třída pro dvoufázové zpracování: fáze 1 (amplitude), fáze 2 (pitch + export).
    Pracuje se soubory podobně jako v předchozí implementaci: glob pro vyhledání, validace metadat, rychlá analýza.
    """
    def __init__(self, input_dir: str, output_dir: str,
                 attack_duration: float = 0.5,
                 min_correction_cents: float = 33.33,
                 tune_direction: TuneDirection = TUNE_DIRECTION,
                 key_filter: KeyFilter = KEY_FILTER,
                 verbose: bool = False):
        """
        Inicializace: validace cest, komponenty (pitch detektor, velocity analyzer).
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.attack_duration = attack_duration
        self.min_correction_cents = min_correction_cents
        self.tune_direction = tune_direction
        self.key_filter = key_filter
        self.verbose = verbose

        # Validace vstupu
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Inicializace komponent
        self.pitch_detector = CrepeHybridDetector()
        self.velocity_analyzer = AdvancedVelocityAnalyzer(attack_duration)

        if verbose:
            logger.setLevel(logging.DEBUG)

    def process_all(self) -> None:
        """
        Hlavní proces: fáze 1 + fáze 2, s logováním a statistikami.
        """
        logger.info("=== TWO-PHASE PITCH CORRECTOR ===")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Minimum correction threshold: {self.min_correction_cents:.1f} cents")

        # Fáze 1
        logger.info("\n=== PHASE 1: AMPLITUDE ANALYSIS ===")
        lite_samples = self._phase1_amplitude_analysis()
        if not lite_samples:
            logger.error("No samples found in Phase 1")
            return

        velocity_mapping = self._create_velocity_mapping_from_amplitudes(lite_samples)

        # Fáze 2
        logger.info("\n=== PHASE 2: PITCH DETECTION & IMMEDIATE EXPORT ===")
        self._phase2_full_processing(lite_samples, velocity_mapping)

        # Finální statistiky
        self._print_final_statistics(lite_samples, velocity_mapping)

        logger.info("=== PROCESSING COMPLETED SUCCESSFULLY ===")

    def _phase1_amplitude_analysis(self) -> List[AudioSampleLite]:
        """
        Fáze 1: Rychlá analýza amplitud – načítání souborů pomocí glob, validace, amplitude výpočet (jako v předchozí implementaci).
        """
        logger.info("Scanning files for amplitude analysis...")
        supported_extensions = ['*.wav', '*.WAV', '*.flac', '*.FLAC']  # Podpora formátů jako v předchozí
        audio_files = []
        for ext in supported_extensions:
            audio_files.extend(list(self.input_dir.glob(ext)))
        if not audio_files:
            logger.error("No supported audio files found")
            return []

        logger.info(f"Found {len(audio_files)} audio files")

        lite_samples = []
        successful = 0
        failed = 0

        for i, filepath in enumerate(audio_files, 1):
            logger.info(f"[{i}/{len(audio_files)}] Analyzing: {filepath.name}")
            try:
                amplitude_data = self._quick_amplitude_analysis(filepath)
                if amplitude_data is None:
                    logger.warning(f"Skipping {filepath.name}: amplitude analysis failed")
                    failed += 1
                    continue

                lite_sample = AudioSampleLite(
                    filepath=filepath,
                    max_amplitude=amplitude_data["max_amplitude"],
                    rms_amplitude=amplitude_data["rms_amplitude"],
                    duration=amplitude_data["duration"],
                    sample_rate=amplitude_data["sample_rate"]
                )
                lite_samples.append(lite_sample)
                successful += 1

                # Log základních info (jako v předchozí)
                logger.info(f" Max amplitude: {amplitude_data['max_amplitude']:.4f}")
                logger.info(f" RMS amplitude: {amplitude_data['rms_amplitude']:.4f}")
                logger.info(f" Duration: {amplitude_data['duration']:.2f}s")

            except Exception as e:
                logger.error(f"Error analyzing {filepath.name}: {e}")
                failed += 1

        logger.info(f"Phase 1 complete: {successful} successful, {failed} failed")
        return lite_samples

    def _quick_amplitude_analysis(self, filepath: Path) -> Optional[Dict]:
        """
        Rychlá analýza amplitudy: validace metadat, načtení waveformu, výpočet max/RMS (jako v předchozí).
        """
        try:
            info = sf.info(str(filepath))

            # Validace (jako v předchozí)
            if info.samplerate not in [44100, 48000, 96000]:
                logger.warning(f"Unsupported sample rate {info.samplerate} Hz")
                return None
            if info.duration < 0.05:
                logger.warning(f"File too short ({info.duration:.2f}s)")
                return None
            if info.duration > 300.0:
                logger.warning(f"File too long ({info.duration:.2f}s)")
                return None

            waveform, sr = sf.read(str(filepath))
            if len(waveform.shape) == 2:
                mono_waveform = np.mean(waveform, axis=1)
            else:
                mono_waveform = waveform

            max_amplitude = np.max(np.abs(mono_waveform))
            rms_amplitude = np.sqrt(np.mean(mono_waveform ** 2))

            if max_amplitude < 1e-6:
                logger.warning("Audio level too low")
                return None

            return {
                "max_amplitude": max_amplitude,
                "rms_amplitude": rms_amplitude,
                "duration": info.duration,
                "sample_rate": sr
            }

        except Exception as e:
            logger.error(f"Quick analysis failed for {filepath}: {e}")
            return None

    def _create_velocity_mapping_from_amplitudes(self, lite_samples: List[AudioSampleLite]) -> Dict:
        """
        Vytvoření velocity mappingu: outlier removal, thresholds, přiřazení velocity (jako v předchozí).
        """
        logger.info("Creating velocity mapping from amplitudes...")

        if not lite_samples:
            logger.error("No samples for velocity mapping")
            return {"thresholds": [], "num_samples": 0}

        rms_amplitudes = [s.rms_amplitude for s in lite_samples]

        # Odstranění outliers (jako v předchozí)
        q1, q3 = np.percentile(rms_amplitudes, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_values = [v for v in rms_amplitudes if lower_bound <= v <= upper_bound]
        outliers_removed = len(rms_amplitudes) - len(filtered_values)
        if outliers_removed > 0:
            logger.info(f"Removed {outliers_removed} outliers from amplitude data")

        # Thresholds pro 8 úrovní (jako v předchozí)
        num_velocities = 8
        if len(filtered_values) >= num_velocities:
            thresholds = [np.percentile(filtered_values, ((i + 1) * 100) / num_velocities) for i in range(num_velocities - 1)]
        else:
            min_val, max_val = min(filtered_values), max(filtered_values)
            thresholds = [min_val + ((i + 1) * (max_val - min_val)) / num_velocities for i in range(num_velocities - 1)]

        # Přiřazení velocity
        for sample in lite_samples:
            sample.assigned_velocity = self._assign_velocity_from_amplitude(sample.rms_amplitude, thresholds)

        mapping = {
            "thresholds": thresholds,
            "primary_metric": "rms_amplitude",
            "min_value": min(filtered_values) if filtered_values else 0,
            "max_value": max(filtered_values) if filtered_values else 1,
            "num_samples": len(lite_samples),
            "outliers_removed": outliers_removed
        }

        self._log_velocity_distribution(lite_samples)
        return mapping

    def _assign_velocity_from_amplitude(self, amplitude: float, thresholds: List[float]) -> int:
        """
        Přiřazení velocity na základě thresholds (jako v předchozí).
        """
        for velocity, threshold in enumerate(thresholds):
            if amplitude <= threshold:
                return velocity
        return len(thresholds)  # Nejvyšší velocity

    def _log_velocity_distribution(self, lite_samples: List[AudioSampleLite]) -> None:
        """
        Logování distribuce velocity (jako v předchozí).
        """
        velocity_counts = defaultdict(int)
        for sample in lite_samples:
            if sample.assigned_velocity is not None:
                velocity_counts[sample.assigned_velocity] += 1

        logger.info("Velocity distribution from amplitudes:")
        total_samples = len(lite_samples)
        for vel in range(8):
            count = velocity_counts[vel]
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            logger.info(f" Velocity {vel}: {count} samples ({percentage:.1f}%)")

    def _phase2_full_processing(self, lite_samples: List[AudioSampleLite], velocity_mapping: Dict) -> None:
        """
        Fáze 2: Plná zpracování, pitch detekce, korekce a export (jako v předchozí, s loggingem).
        """
        logger.info(f"Processing {len(lite_samples)} samples with immediate export...")

        sample_counters = defaultdict(int)

        for i, lite_sample in enumerate(lite_samples, 1):
            logger.info(f"[{i}/{len(lite_samples)}] Processing & exporting: {lite_sample.filepath.name}")
            try:
                waveform, sr = sf.read(str(lite_sample.filepath))
                waveform = waveform.astype(np.float32)
                if len(waveform.shape) == 1:
                    waveform = waveform[:, np.newaxis]

                pitch_analysis = self.pitch_detector.detect(waveform, sr)
                if pitch_analysis.fundamental_freq is None:
                    logger.warning(f"Pitch detection failed for {lite_sample.filepath.name}")
                    continue

                velocity_analysis = self.velocity_analyzer.analyze(waveform, sr)

                sample = AudioSampleData(
                    filepath=lite_sample.filepath,
                    waveform=waveform,
                    sample_rate=sr,
                    duration=lite_sample.duration,
                    pitch_analysis=pitch_analysis,
                    velocity_analysis=velocity_analysis,
                    assigned_velocity=lite_sample.assigned_velocity
                )

                target_midi = DirectionalTuner.find_target_midi_directional(
                    pitch_analysis.fundamental_freq, self.tune_direction,
                    self.pitch_detector.piano_frequencies, self.key_filter
                )

                if target_midi is None or not DirectionalTuner.validate_directional_tuning(
                        pitch_analysis.fundamental_freq, target_midi, self.tune_direction, self.key_filter
                ):
                    logger.warning("Directional tuning failed, skipping")
                    continue

                sample.target_midi_note = target_midi

                rr_key = (target_midi, sample.assigned_velocity)
                rr_index = sample_counters[rr_key]
                sample_counters[rr_key] += 1

                cents_correction = 1200 * np.log2(AudioUtils.midi_to_freq(target_midi) / pitch_analysis.fundamental_freq)
                if abs(cents_correction) < self.min_correction_cents:
                    corrected_waveform = waveform
                    logger.info(f" Skipping small correction: {cents_correction:.2f} cents")
                else:
                    shift_factor = AudioUtils.midi_to_freq(target_midi) / pitch_analysis.fundamental_freq
                    new_length = int(len(waveform) / shift_factor)
                    corrected_waveform = resample(waveform, new_length)
                    logger.info(f" Applied pitch correction: {cents_correction:+.2f} cents")

                self._export_sample(sample, corrected_waveform, sr, rr_index)

            except Exception as e:
                logger.error(f"Error processing {lite_sample.filepath.name}: {e}")

    def _export_sample(self, sample: AudioSampleData, corrected_waveform: np.ndarray, sr: int, rr_index: int) -> None:
        """
        Export do dvou formátů (44.1 a 48 kHz) s názvy jako v README (jako v předchozí).
        """
        midi_str = f"m{sample.target_midi_note:03d}"
        vel_str = f"vel{sample.assigned_velocity}"
        rr_str = f"-rr{rr_index}" if rr_index > 0 else ""

        for target_sr, sr_suffix in [(44100, 'f44'), (48000, 'f48')]:
            final_audio = resample(corrected_waveform, int(len(corrected_waveform) * target_sr / sr)) if sr != target_sr else corrected_waveform
            output_filename = f"{midi_str}-{vel_str}-{sr_suffix}{rr_str}.wav"
            output_path = self.output_dir / output_filename
            sf.write(str(output_path), final_audio, target_sr)
            logger.info(f" Saved: {output_path.name}")

    def _print_final_statistics(self, lite_samples: List[AudioSampleLite], velocity_mapping: Dict) -> None:
        """
        Finální statistiky (jako v předchozí).
        """
        logger.info("\n=== FINAL STATISTICS ===")
        logger.info(f"Phase 1 (Amplitude): {len(lite_samples)} samples analyzed")
        # Další statistiky by mohly být přidány, ale nejsou v předchozí implementaci
def parse_arguments():
    """
    Parsování argumentů (jako v předchozí, s povinnými --input-dir a --output-dir).
    """
    parser = argparse.ArgumentParser(description="Two-Phase Pitch Corrector pro piano sample.")
    parser.add_argument('--input-dir', required=True, help='Vstupní adresář')
    parser.add_argument('--output-dir', required=True, help='Výstupní adresář')
    parser.add_argument('--attack-duration', type=float, default=0.5)
    parser.add_argument('--min-correction-cents', type=float, default=33.33)
    direction_group = parser.add_mutually_exclusive_group()
    direction_group.add_argument('--tune-direction-up', action='store_true')
    direction_group.add_argument('--tune-direction-down', action='store_true')
    direction_group.add_argument('--tune-direction-nearest', action='store_true')
    key_group = parser.add_mutually_exclusive_group()
    key_group.add_argument('--only-white-keys', action='store_true')
    key_group.add_argument('--only-black-keys', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    tune_direction = TuneDirection.UP if args.tune_direction_up else TuneDirection.DOWN if args.tune_direction_down else TuneDirection.NEAREST
    key_filter = KeyFilter.WHITE_ONLY if args.only_white_keys else KeyFilter.BLACK_ONLY if args.only_black_keys else KeyFilter.ALL
    corrector = TwoPhaseRefactoredPitchCorrector(
        args.input_dir, args.output_dir, args.attack_duration, args.min_correction_cents, tune_direction, key_filter, args.verbose
    )
    corrector.process_all()
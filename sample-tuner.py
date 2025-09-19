"""
Hlavní orchestrace pitch correctoru s dvou-fázovým zpracováním.
Fáze 1: Rychlá analýza amplitud pro velocity mapping
Fáze 2: Plná pitch detection a korekce
"""

import argparse
import soundfile as sf
import numpy as np
from collections import defaultdict
from pathlib import Path
import logging
import sys
from typing import List, Dict, Optional, Tuple

# Import našich modulů
from audio_utils import AudioSampleData, AudioUtils, AudioProcessor
from piano_pitch_detector import PianoPitchDetector
from velocity_analysis import AdvancedVelocityAnalyzer, OptimizedVelocityMapper

# Konfigurace loggingu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioSampleLite:
    """Zjednodušená verze AudioSampleData pro první fázi"""
    def __init__(self, filepath: Path, max_amplitude: float, rms_amplitude: float,
                 duration: float, sample_rate: int):
        self.filepath = filepath
        self.max_amplitude = max_amplitude
        self.rms_amplitude = rms_amplitude
        self.duration = duration
        self.sample_rate = sample_rate
        self.assigned_velocity = None


class TwoPhaseRefactoredPitchCorrector:
    """
    Dvou-fázový pitch corrector s optimalizovaným zpracováním.

    Fáze 1: Rychlá analýza všech souborů pro amplitude/velocity mapping
    Fáze 2: Plná pitch detection a korekce pouze pro vybrané soubory
    """

    def __init__(self, input_dir: str, output_dir: str,
                 attack_duration: float = 0.5,
                 min_correction_cents: float = 33.33,
                 verbose: bool = False):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.attack_duration = attack_duration
        self.min_correction_cents = min_correction_cents
        self.verbose = verbose

        # Validace
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Inicializace komponent - pouze piano detector
        self.pitch_detector_type = "piano"  # Fixed - pouze piano
        self.pitch_detector = None
        self.velocity_analyzer = AdvancedVelocityAnalyzer(attack_duration)

        # Konfigurace loggingu
        if verbose:
            logger.setLevel(logging.DEBUG)
            logging.getLogger('audio_utils').setLevel(logging.DEBUG)
            logging.getLogger('pitch_detection').setLevel(logging.DEBUG)
            logging.getLogger('piano_pitch_detector').setLevel(logging.DEBUG)
            logging.getLogger('velocity_analysis').setLevel(logging.DEBUG)

    def process_all(self) -> None:
        """Hlavní dvou-fázové zpracování"""
        try:
            logger.info("=== TWO-PHASE PITCH CORRECTOR ===")
            logger.info(f"Input: {self.input_dir}")
            logger.info(f"Output: {self.output_dir}")
            logger.info(f"Planned pitch detector: piano (only option)")
            logger.info(f"Minimum correction threshold: {self.min_correction_cents:.1f} cents")

            # === FÁZE 1: RYCHLÁ ANALÝZA AMPLITUD ===
            logger.info("\n=== PHASE 1: AMPLITUDE ANALYSIS ===")
            lite_samples = self._phase1_amplitude_analysis()

            if not lite_samples:
                logger.error("No samples found in Phase 1")
                return

            # Vytvoření velocity mappingu na základě amplitud
            velocity_mapping = self._create_velocity_mapping_from_amplitudes(lite_samples)

            # === FÁZE 2: PLNÁ PITCH DETECTION A OKAMŽITÝ EXPORT ===
            logger.info("\n=== PHASE 2: PITCH DETECTION & IMMEDIATE EXPORT ===")

            # Inicializace pitch detektoru až nyní
            self._initialize_pitch_detector()

            # Plné zpracování s okamžitým ukládáním
            processed_samples = self._phase2_full_processing(lite_samples, velocity_mapping)

            # === FINÁLNÍ STATISTIKY ===
            self._print_final_statistics(lite_samples, processed_samples, velocity_mapping)

            logger.info("=== PROCESSING COMPLETED SUCCESSFULLY ===")

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            if self.verbose:
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _phase1_amplitude_analysis(self) -> List[AudioSampleLite]:
        """Fáze 1: Rychlá analýza amplitud všech souborů"""
        logger.info("Scanning files for amplitude analysis...")

        # Najdi všechny audio soubory
        supported_extensions = ['*.wav', '*.WAV', '*.flac', '*.FLAC']
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
                # Rychlé načtení pouze pro amplitudu
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

                # Log základních informací
                logger.info(f"  Max amplitude: {amplitude_data['max_amplitude']:.4f}")
                logger.info(f"  RMS amplitude: {amplitude_data['rms_amplitude']:.4f}")
                logger.info(f"  Duration: {amplitude_data['duration']:.2f}s")

            except Exception as e:
                logger.error(f"Error analyzing {filepath.name}: {e}")
                if self.verbose:
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                failed += 1
                continue

        logger.info(f"Phase 1 complete: {successful} successful, {failed} failed")
        return lite_samples

    def _quick_amplitude_analysis(self, filepath: Path) -> Optional[Dict]:
        """Rychlá analýza amplitudy bez plného načtení"""
        try:
            # Načti metadata nejdříve
            info = sf.info(str(filepath))

            # Základní validace
            if info.samplerate not in [44100, 48000, 96000]:
                logger.warning(f"Unsupported sample rate {info.samplerate} Hz")
                return None

            if info.duration < 0.05:
                logger.warning(f"File too short ({info.duration:.2f}s)")
                return None

            if info.duration > 300.0:
                logger.warning(f"File too long ({info.duration:.2f}s)")
                return None

            # Načti audio pro amplitude analýzu
            waveform, sr = sf.read(str(filepath))

            # Ensure mono for amplitude analysis
            if len(waveform.shape) == 2:
                mono_waveform = np.mean(waveform, axis=1)
            else:
                mono_waveform = waveform

            # Základní amplitude metriky
            max_amplitude = np.max(np.abs(mono_waveform))
            rms_amplitude = np.sqrt(np.mean(mono_waveform ** 2))

            # Validace úrovně
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
        """Vytvoří velocity mapping na základě amplitud"""
        logger.info("Creating velocity mapping from amplitudes...")

        if not lite_samples:
            logger.error("No samples for velocity mapping")
            return {"thresholds": [], "num_samples": 0}

        # Extrakce amplitud pro mapping
        max_amplitudes = [s.max_amplitude for s in lite_samples]
        rms_amplitudes = [s.rms_amplitude for s in lite_samples]

        # Použij RMS jako primární metriku (stabilnější než max)
        primary_values = rms_amplitudes

        # Odstranění outliers
        q1, q3 = np.percentile(primary_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_values = [v for v in primary_values if lower_bound <= v <= upper_bound]
        outliers_removed = len(primary_values) - len(filtered_values)

        if outliers_removed > 0:
            logger.info(f"Removed {outliers_removed} outliers from amplitude data")

        # Vytvoření thresholds pro 8 velocity úrovní
        num_velocities = 8
        if len(filtered_values) >= num_velocities:
            thresholds = []
            for i in range(num_velocities - 1):
                percentile = ((i + 1) * 100) / num_velocities
                threshold = np.percentile(filtered_values, percentile)
                thresholds.append(threshold)
        else:
            # Fallback pro málo samples
            min_val, max_val = min(filtered_values), max(filtered_values)
            thresholds = []
            for i in range(num_velocities - 1):
                threshold = min_val + ((i + 1) * (max_val - min_val)) / num_velocities
                thresholds.append(threshold)

        # Přiřazení velocity každému sample
        for sample in lite_samples:
            sample.assigned_velocity = self._assign_velocity_from_amplitude(
                sample.rms_amplitude, thresholds
            )

        mapping = {
            "thresholds": thresholds,
            "primary_metric": "rms_amplitude",
            "min_value": min(filtered_values) if filtered_values else 0,
            "max_value": max(filtered_values) if filtered_values else 1,
            "num_samples": len(lite_samples),
            "outliers_removed": outliers_removed
        }

        # Logování distribuce
        self._log_velocity_distribution(lite_samples)

        return mapping

    def _assign_velocity_from_amplitude(self, amplitude: float, thresholds: List[float]) -> int:
        """Přiřadí velocity na základě amplitudy"""
        for velocity, threshold in enumerate(thresholds):
            if amplitude <= threshold:
                return velocity
        return len(thresholds)  # Nejvyšší velocity

    def _log_velocity_distribution(self, lite_samples: List[AudioSampleLite]) -> None:
        """Loguje distribuci velocity"""
        velocity_counts = defaultdict(int)
        for sample in lite_samples:
            if sample.assigned_velocity is not None:
                velocity_counts[sample.assigned_velocity] += 1

        logger.info("Velocity distribution from amplitudes:")
        total_samples = len(lite_samples)
        for vel in range(8):
            count = velocity_counts[vel]
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            logger.info(f"  Velocity {vel}: {count} samples ({percentage:.1f}%)")

    def _initialize_pitch_detector(self) -> None:
        """Inicializuje piano pitch detector"""
        # Pouze piano detector - ostatní odstraněny
        self.pitch_detector = PianoPitchDetector()
        logger.info("Initialized piano pitch detector")
        logger.info(f"Piano frequency range: {self.pitch_detector.fmin:.1f} - {self.pitch_detector.fmax:.1f} Hz")
        logger.info(f"Piano keys available: {len(self.pitch_detector.piano_frequencies)}")

    def _phase2_full_processing(self, lite_samples: List[AudioSampleLite],
                               velocity_mapping: Dict) -> List[AudioSampleData]:
        """Fáze 2: Plná pitch detection a OKAMŽITÉ uložení"""
        logger.info(f"Processing {len(lite_samples)} samples with immediate export...")

        processed_samples = []
        successful = 0
        failed = 0
        total_exported = 0
        sample_counters = defaultdict(int)

        for i, lite_sample in enumerate(lite_samples, 1):
            logger.info(f"[{i}/{len(lite_samples)}] Processing & exporting: {lite_sample.filepath.name}")

            try:
                # Plné načtení souboru pro pitch detection
                waveform, sr = sf.read(str(lite_sample.filepath))

                # Ensure 2D format
                if len(waveform.shape) == 1:
                    waveform = waveform[:, np.newaxis]

                # Pitch analýza
                mono_audio = AudioUtils.to_mono(waveform)
                pitch_analysis = self._perform_pitch_analysis(mono_audio, sr)

                if pitch_analysis.fundamental_freq is None:
                    logger.warning(f"Pitch detection failed for {lite_sample.filepath.name}")
                    failed += 1
                    continue

                # Vytvoř plný AudioSampleData
                duration = len(waveform) / sr
                sample = AudioSampleData(
                    filepath=lite_sample.filepath,
                    waveform=waveform,
                    sample_rate=sr,
                    duration=duration,
                    pitch_analysis=pitch_analysis,
                    velocity_analysis=None,  # Nepotřebujeme pro tuto fázi
                    assigned_velocity=lite_sample.assigned_velocity
                )

                processed_samples.append(sample)

                # Validace pro piano
                validation = self._validate_piano_detection(pitch_analysis, lite_sample.filepath.name)
                if not validation["valid"]:
                    logger.warning(f"  {validation['reason']}")
                    continue

                # Log analýzy
                self._log_analysis_results(sample)

                # Určení cílové MIDI noty
                target_midi = self._find_target_midi_note(pitch_analysis.fundamental_freq)
                if target_midi is None:
                    logger.warning("Cannot determine target MIDI note, skipping")
                    failed += 1
                    continue

                sample.target_midi_note = target_midi

                # Výpočet korekce
                correction_result = self._calculate_pitch_correction(
                    sample.pitch_analysis.fundamental_freq, target_midi
                )

                sample.pitch_correction_semitones = correction_result["semitones"]

                # Kontrola, zda přeskočit malou korekci
                skip_correction = self._should_skip_correction(correction_result["cents"])

                # Round-robin tracking
                rr_key = (target_midi, sample.assigned_velocity)
                rr_index = sample_counters[rr_key]
                sample_counters[rr_key] += 1

                # Logování
                self._log_correction_info(sample, correction_result, rr_index, skip_correction)

                # Aplikace korekce
                if skip_correction:
                    corrected_audio, corrected_sr = sample.waveform, sample.sample_rate
                    logger.info(f"  Using original audio without correction")
                else:
                    corrected_audio, corrected_sr = self._apply_pitch_correction(
                        sample, correction_result["semitones"]
                    )

                # OKAMŽITÝ EXPORT
                export_count = self._export_sample_multiple_formats(
                    sample, corrected_audio, corrected_sr, rr_index
                )
                total_exported += export_count

                successful += 1

            except Exception as e:
                logger.error(f"Error processing {lite_sample.filepath.name}: {e}")
                if self.verbose:
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                failed += 1
                continue

        logger.info(f"Phase 2 complete: {successful} successful, {failed} failed, {total_exported} files exported")

        # Log round-robin statistik
        self._log_round_robin_statistics(sample_counters)

        return processed_samples

    def _perform_pitch_analysis(self, mono_audio: np.ndarray, sr: int):
        """Provede pitch analýzu s error handling"""
        try:
            return self.pitch_detector.detect(mono_audio, sr)
        except Exception as e:
            logger.error(f"Pitch detection error: {e}")
            from audio_utils import PitchAnalysisResult
            return PitchAnalysisResult(
                fundamental_freq=None,
                confidence=0.0,
                harmonics=[],
                method_used=f"{type(self.pitch_detector).__name__}_failed",
                spectral_centroid=0.0,
                spectral_rolloff=sr/2
            )

    def _validate_piano_detection(self, pitch_analysis, filename: str) -> Dict[str, any]:
        """Validace specifická pro piano detection"""
        freq = pitch_analysis.fundamental_freq

        if freq is None:
            return {"valid": False, "reason": "No fundamental frequency detected"}

        if not (27.5 <= freq <= 4186.0):
            return {"valid": False, "reason": f"Frequency {freq:.2f} Hz outside piano range"}

        piano_frequencies = self.pitch_detector.piano_frequencies
        distances = np.abs(piano_frequencies - freq)
        min_distance = np.min(distances)

        if min_distance > freq * 0.03:
            logger.warning(f"Detected frequency {freq:.2f} Hz is not close to any piano key")

        return {"valid": True, "reason": "OK"}

    def _log_analysis_results(self, sample: AudioSampleData) -> None:
        """Logování výsledků analýzy"""
        pitch = sample.pitch_analysis
        freq = pitch.fundamental_freq
        midi = pitch.midi_note
        note_name = AudioUtils.midi_to_note_name(midi) if midi else "N/A"

        logger.info(f"  Detected frequency: {freq:.2f} Hz -> MIDI {midi} ({note_name})")
        logger.info(f"  Assigned velocity: {sample.assigned_velocity} (from amplitude analysis)")
        logger.info(f"  Confidence: {pitch.confidence:.3f}, Method: {pitch.method_used}")

        # Piano-specific info
        if isinstance(self.pitch_detector, PianoPitchDetector) and freq:
            piano_distances = np.abs(self.pitch_detector.piano_frequencies - freq)
            min_idx = np.argmin(piano_distances)
            closest_piano_freq = self.pitch_detector.piano_frequencies[min_idx]
            distance_cents = 1200 * np.log2(freq / closest_piano_freq)
            logger.info(f"  Closest piano frequency: {closest_piano_freq:.2f} Hz ({distance_cents:+.1f} cents)")

    def _find_target_midi_note(self, freq: float) -> Optional[int]:
        """Najde nejbližší MIDI notu"""
        try:
            if isinstance(self.pitch_detector, PianoPitchDetector):
                piano_distances = np.abs(self.pitch_detector.piano_frequencies - freq)
                min_idx = np.argmin(piano_distances)
                closest_piano_freq = self.pitch_detector.piano_frequencies[min_idx]
                midi_float = 12 * np.log2(closest_piano_freq / 440.0) + 69
                midi_note = int(np.round(midi_float))
            else:
                midi_float = 12 * np.log2(freq / 440.0) + 69
                midi_note = int(np.round(midi_float))

            return midi_note if 0 <= midi_note <= 127 else None

        except (ValueError, OverflowError) as e:
            logger.error(f"Error calculating MIDI note for frequency {freq:.2f} Hz: {e}")
            return None

    def _calculate_pitch_correction(self, detected_freq: float, target_midi: int) -> Dict:
        """Výpočet pitch korekce"""
        target_freq = AudioUtils.midi_to_freq(target_midi)
        semitone_correction = 12 * np.log2(target_freq / detected_freq)

        return {
            "semitones": semitone_correction,
            "detected_freq": detected_freq,
            "target_freq": target_freq,
            "cents": semitone_correction * 100,
            "is_large_correction": abs(semitone_correction) > 1.0
        }

    def _should_skip_correction(self, cents: float) -> bool:
        """Rozhodne, zda přeskočit korekci"""
        return abs(cents) < self.min_correction_cents

    def _log_correction_info(self, sample: AudioSampleData, correction: Dict,
                            rr_index: int, skip_correction: bool) -> None:
        """Logování informací o korekci"""
        note_name = AudioUtils.midi_to_note_name(sample.target_midi_note)

        if rr_index == 0:
            logger.info(f"  MIDI {sample.target_midi_note} ({note_name}) -> velocity {sample.assigned_velocity}")
        else:
            logger.info(f"  MIDI {sample.target_midi_note} ({note_name}) -> velocity {sample.assigned_velocity} [RR {rr_index+1}]")

        logger.info(f"  Pitch correction: {correction['semitones']:+.3f} semitones ({correction['cents']:+.1f} cents)")

        if skip_correction:
            logger.info(f"  Correction skipped (below {self.min_correction_cents:.1f} cent threshold)")

    def _apply_pitch_correction(self, sample: AudioSampleData, semitones: float) -> tuple:
        """Aplikace pitch korekce"""
        if abs(semitones) > 0.01:
            return AudioProcessor.pitch_shift(sample.waveform, sample.sample_rate, semitones)
        else:
            return sample.waveform, sample.sample_rate

    def _export_sample_multiple_formats(self, sample: AudioSampleData, audio: np.ndarray,
                                      sr: int, rr_index: int) -> int:
        """Export vzorku ve více formátech"""
        target_rates = [(44100, 'f44'), (48000, 'f48')]
        exported_count = 0

        for target_sr, sr_suffix in target_rates:
            try:
                final_audio = AudioProcessor.resample_audio(audio, sr, target_sr)
                output_path = self._generate_filename(
                    sample.target_midi_note, sample.assigned_velocity, sr_suffix, rr_index
                )
                sf.write(str(output_path), final_audio, target_sr)
                logger.info(f"  Saved: {output_path.name}")
                exported_count += 1

            except Exception as e:
                logger.error(f"Export error for {sr_suffix}: {e}")

        return exported_count

    def _generate_filename(self, midi: int, velocity: int, sr_suffix: str, sample_index: int = 0) -> Path:
        """Generuje název souboru"""
        if sample_index == 0:
            base_name = f"m{midi:03d}-vel{velocity}-{sr_suffix}"
        else:
            base_name = f"m{midi:03d}-vel{velocity}-{sr_suffix}-rr{sample_index+1}"
        return self.output_dir / f"{base_name}.wav"

    def _log_round_robin_statistics(self, sample_counters: defaultdict) -> None:
        """
        Loguje statistiky round-robin distribuce (počty vzorků pro každou kombinaci MIDI + velocity).
        Pomáhá ověřit rovnoměrnou distribuci vzorků.
        """
        logger.info("\n=== ROUND-ROBIN STATISTICS ===")
        total_rr_samples = sum(sample_counters.values())
        logger.info(f"Total round-robin samples: {total_rr_samples}")

        if not sample_counters:
            logger.warning("No round-robin data available")
            return

        # Seřazení podle klíčů pro lepší čitelnost
        sorted_keys = sorted(sample_counters.keys())
        for (midi, vel), count in sorted_keys:
            percentage = (count / total_rr_samples * 100) if total_rr_samples > 0 else 0
            note_name = AudioUtils.midi_to_note_name(midi)
            logger.info(f"  MIDI {midi} ({note_name}) + Velocity {vel}: {count} samples ({percentage:.1f}%)")

        # Průměrný počet na kombinaci
        avg_per_combination = total_rr_samples / len(sample_counters)
        logger.info(f"Average samples per MIDI+velocity combination: {avg_per_combination:.1f}")

    def _print_final_statistics(self, lite_samples: List[AudioSampleLite],
                               processed_samples: List[AudioSampleData],
                               velocity_mapping: Dict) -> None:
        """Finální statistiky"""
        logger.info("\n=== FINAL STATISTICS ===")
        logger.info(f"Phase 1 (Amplitude): {len(lite_samples)} samples analyzed")
        logger.info(f"Phase 2 (Pitch): {len(processed_samples)} samples processed")

        # Velocity distribuce
        velocity_counts = defaultdict(int)
        for sample in lite_samples:
            if sample.assigned_velocity is not None:
                velocity_counts[sample.assigned_velocity] += 1

        logger.info("Final velocity distribution:")
        total = len(lite_samples)
        for vel in range(8):
            count = velocity_counts[vel]
            percentage = (count / total * 100) if total > 0 else 0
            logger.info(f"  Velocity {vel}: {count} samples ({percentage:.1f}%)")

        # Pitch detection úspěšnost
        if processed_samples:
            confidences = [s.pitch_analysis.confidence for s in processed_samples
                          if s.pitch_analysis.fundamental_freq is not None]
            if confidences:
                avg_confidence = np.mean(confidences)
                logger.info(f"Average pitch confidence: {avg_confidence:.3f}")


def parse_arguments():
    """Parsování argumentů"""
    parser = argparse.ArgumentParser(
        description="""
Two-Phase Pitch Corrector s optimalizovaným zpracováním

Fáze 1: Rychlá analýza amplitud pro velocity mapping
Fáze 2: Plná pitch detection a korekce pouze pro vybrané soubory

Výhody:
- Rychlejší zpracování velkých množství souborů
- Lepší velocity mapping na základě celé sady
- Optimalizace paměti a výkonu
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--input-dir', required=True,
                       help='Cesta k vstupnímu adresáři s audio soubory')
    parser.add_argument('--output-dir', required=True,
                       help='Cesta k výstupnímu adresáři')
    parser.add_argument('--attack-duration', type=float, default=0.5,
                       help='Délka attack fáze pro velocity detection (default: 0.5s)')
    parser.add_argument('--min-correction-cents', type=float, default=33.33,
                       help='Minimální korekce v centech (default: 33.33 = 1/3 pultonu)')
    parser.add_argument('--detector',
                       choices=['piano', 'hybrid', 'simple', 'adaptive'],
                       default='piano',
                       help='Typ pitch detectoru (default: piano)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging pro debugging')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    try:
        corrector = TwoPhaseRefactoredPitchCorrector(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            attack_duration=args.attack_duration,
            min_correction_cents=args.min_correction_cents,
            verbose=args.verbose
        )

        corrector.process_all()

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)
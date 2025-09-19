"""
Hlavní orchestrace pitch correctoru.
Integruje všechny komponenty a řídí celý proces.
"""

import argparse
import soundfile as sf
import numpy as np
from collections import defaultdict
from pathlib import Path
import logging
import sys
from typing import List, Dict, Optional

# Import našich modulů
from audio_utils import AudioSampleData, AudioUtils, AudioProcessor
from pitch_detection import OriginalHybridPitchDetector, SimplePitchDetector
from velocity_analysis import AdvancedVelocityAnalyzer, OptimizedVelocityMapper

# Konfigurace loggingu - OPRAVENO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RefactoredPitchCorrector:
    """
    Hlavní třída pitch correctoru.

    Klíčové vlastnosti:
    - Modulární architektura rozdělená do více souborů
    - Algoritmy s ochranou proti chybám
    - Pokročilé velocity mapping s multiple metrikami
    - Robustní error handling
    """

    def __init__(self, input_dir: str, output_dir: str,
                 attack_duration: float = 0.5,
                 pitch_detector_type: str = "hybrid",
                 min_correction_cents: float = 33.33,  # 1/3 pultonu = 33.33 centů
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

        # Inicializace komponent podle typu
        if pitch_detector_type == "hybrid":
            self.pitch_detector = OriginalHybridPitchDetector()
        elif pitch_detector_type == "simple":
            self.pitch_detector = SimplePitchDetector()
        elif pitch_detector_type == "adaptive":
            self.pitch_detector = OriginalHybridPitchDetector()
        else:
            raise ValueError(f"Unknown pitch detector type: {pitch_detector_type}")

        self.velocity_analyzer = AdvancedVelocityAnalyzer(attack_duration)

        # Konfigurace loggingu
        if verbose:
            logger.setLevel(logging.DEBUG)
            # Enable debug logging for all our modules
            logging.getLogger('audio_utils').setLevel(logging.DEBUG)
            logging.getLogger('pitch_detection').setLevel(logging.DEBUG)
            logging.getLogger('velocity_analysis').setLevel(logging.DEBUG)

    def process_all(self) -> None:
        """Hlavní processing pipeline s error handling"""
        try:
            logger.info("=== PITCH CORRECTOR ===")
            logger.info(f"Input: {self.input_dir}")
            logger.info(f"Output: {self.output_dir}")
            logger.info(f"Pitch detector: {type(self.pitch_detector).__name__}")
            logger.info(f"Minimum correction threshold: {self.min_correction_cents:.1f} cents")
            logger.info("Processing with multiple metrics")
            logger.info("No sample duration limits - preserving original lengths")

            # Fáze 1: Načtení a analýza
            samples = self._load_and_analyze()
            if not samples:
                logger.error("No samples to process")
                return

            # Fáze 2: Velocity mapping
            velocity_mapping = self._create_velocity_mapping(samples)

            # Fáze 3: Pitch korekce a export
            self._process_and_export(samples, velocity_mapping)

            # Fáze 4: Finální statistiky
            self._print_final_statistics(samples, velocity_mapping)

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

    def _load_and_analyze(self) -> List[AudioSampleData]:
        """Načtení a analýza souborů s robustním error handling"""
        logger.info("Phase 1: Loading and analyzing files")

        # Podporované formáty
        supported_extensions = ['*.wav', '*.WAV', '*.flac', '*.FLAC']
        audio_files = []

        for ext in supported_extensions:
            audio_files.extend(list(self.input_dir.glob(ext)))

        if not audio_files:
            logger.error("No supported audio files found")
            return []

        logger.info(f"Found {len(audio_files)} audio files")

        samples = []
        successful = 0
        failed = 0

        for i, filepath in enumerate(audio_files, 1):
            logger.info(f"[{i}/{len(audio_files)}] Processing: {filepath.name}")

            try:
                # Načtení souboru
                waveform, sr = sf.read(str(filepath))

                # Ensure 2D format pro konzistentní handling
                if len(waveform.shape) == 1:
                    waveform = waveform[:, np.newaxis]

                duration = len(waveform) / sr

                # Validace souboru
                validation_result = self._validate_audio_file(waveform, sr, duration, filepath)
                if not validation_result["valid"]:
                    logger.warning(f"Skipping {filepath.name}: {validation_result['reason']}")
                    failed += 1
                    continue

                # Pitch analýza
                mono_audio = AudioUtils.to_mono(waveform)
                pitch_analysis = self.pitch_detector.detect(mono_audio, sr)

                # Validace pitch detection
                if pitch_analysis.fundamental_freq is None:
                    logger.warning(f"Pitch detection failed for {filepath.name}")
                    failed += 1
                    continue

                if pitch_analysis.confidence < 0.2:  # Snížený práh pro více tolerance
                    logger.warning(f"Low confidence pitch detection ({pitch_analysis.confidence:.2f}) for {filepath.name}")
                    # Ale pokračujeme - možná je to stále použitelné

                # Velocity analýza
                velocity_analysis = self.velocity_analyzer.analyze(waveform, sr)

                # Vytvoření sample data
                sample = AudioSampleData(
                    filepath=filepath,
                    waveform=waveform,
                    sample_rate=sr,
                    duration=duration,
                    pitch_analysis=pitch_analysis,
                    velocity_analysis=velocity_analysis
                )

                samples.append(sample)
                successful += 1

                # Log analýzy
                self._log_analysis_results(sample)

            except Exception as e:
                logger.error(f"Error processing {filepath.name}: {e}")
                if self.verbose:
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                failed += 1
                continue

        logger.info(f"Analysis complete: {successful} successful, {failed} failed")
        return samples

    def _validate_audio_file(self, waveform: np.ndarray, sr: int, duration: float,
                           filepath: Path) -> Dict[str, any]:
        """Validace audio souboru"""

        # Sample rate check
        if sr not in [44100, 48000, 96000]:  # Rozšířeno o 96kHz
            return {"valid": False, "reason": f"Unsupported sample rate {sr} Hz"}

        # Duration check
        if duration < 0.05:  # Velmi krátké soubory
            return {"valid": False, "reason": f"File too short ({duration:.2f}s)"}

        # Zvýšen limit pro velmi dlouhé soubory nebo odstraněn úplně
        if duration > 300.0:  # 5 minut jako rozumná horní hranice pro processing
            return {"valid": False, "reason": f"File extremely long ({duration:.2f}s) - may cause memory issues"}

        # Audio level check
        max_amplitude = np.max(np.abs(waveform))
        if max_amplitude < 1e-6:  # Velmi tichý soubor
            return {"valid": False, "reason": "Audio level too low"}

        # Channel count check
        if len(waveform.shape) > 1 and waveform.shape[1] > 2:
            return {"valid": False, "reason": f"Too many channels ({waveform.shape[1]})"}

        return {"valid": True, "reason": "OK"}

    def _log_analysis_results(self, sample: AudioSampleData) -> None:
        """Logování výsledků analýzy"""
        pitch = sample.pitch_analysis
        velocity = sample.velocity_analysis

        freq = pitch.fundamental_freq
        midi = pitch.midi_note
        note_name = AudioUtils.midi_to_note_name(midi) if midi else "N/A"

        logger.info(f"  Detected frequency: {freq:.2f} Hz -> MIDI {midi} ({note_name})")

        # Vypočítej a zobraz target frekvenci a rozdíl
        if midi is not None:
            target_freq = AudioUtils.midi_to_freq(midi)
            freq_diff = freq - target_freq
            semitone_correction = 12 * np.log2(target_freq / freq)
            cents_correction = semitone_correction * 100

            logger.info(f"  Target frequency:   {target_freq:.2f} Hz")
            logger.info(f"  Frequency difference: {freq_diff:+.2f} Hz")
            logger.info(f"  Required correction: {semitone_correction:+.3f} semitones ({cents_correction:+.1f} cents)")

            # Informace o tom, zda bude korekce aplikována
            if abs(cents_correction) < self.min_correction_cents:
                logger.info(f"  → Correction will be SKIPPED (below {self.min_correction_cents:.1f} cent threshold)")
            else:
                logger.info(f"  → Correction will be APPLIED")

        logger.info(f"  Confidence: {pitch.confidence:.3f}, Method: {pitch.method_used}")
        logger.info(f"  Attack Peak: {velocity.attack_peak_db:.2f} dB")
        logger.info(f"  Harmonics: {len(pitch.harmonics)}")

        if velocity.stereo_width is not None:
            logger.info(f"  Stereo Width: {velocity.stereo_width:.3f}")

    def _create_velocity_mapping(self, samples: List[AudioSampleData]) -> Dict:
        """Vytvoření velocity mappingu"""
        logger.info("Phase 2: Creating velocity mapping")

        # Použij velocity mapping s multiple metrikami
        mapping = OptimizedVelocityMapper.create_advanced_mapping(
            samples,
            primary_metric="attack_peak_db",
            secondary_metric="attack_slope",  # Sekundární metrika pro tie-breaking
            num_velocities=8
        )

        if not mapping["thresholds"]:
            logger.error("Failed to create velocity mapping")
            return mapping

        # Přiřazení velocities s advanced algoritmem
        velocity_analyzer = AdvancedVelocityAnalyzer()

        for sample in samples:
            if sample.velocity_analysis:
                sample.assigned_velocity = OptimizedVelocityMapper.assign_velocity_advanced(
                    sample, mapping, velocity_analyzer
                )

        # Analýza distribuce
        distribution_analysis = OptimizedVelocityMapper.analyze_velocity_distribution(
            samples, mapping
        )

        # Logování statistik
        self._log_velocity_statistics(mapping, distribution_analysis)

        return mapping

    def _log_velocity_statistics(self, mapping: Dict, distribution: Dict) -> None:
        """Logování velocity statistik"""
        logger.info(f"Velocity mapping created:")
        logger.info(f"  Primary metric: {mapping['primary_metric']}")
        logger.info(f"  Secondary metric: {mapping.get('secondary_metric', 'None')}")
        logger.info(f"  Range: {mapping['min_value']:.2f} to {mapping['max_value']:.2f}")
        logger.info(f"  Samples processed: {mapping['num_samples']}")

        if mapping.get('outliers_removed', 0) > 0:
            logger.info(f"  Outliers removed: {mapping['outliers_removed']}")

        logger.info("Velocity distribution:")
        for vel, stats in distribution["velocity_stats"].items():
            if stats["count"] > 0:
                logger.info(f"  Velocity {vel}: {stats['count']} samples "
                          f"(mean: {stats['mean']:.2f}, std: {stats['std']:.2f})")

    def _process_and_export(self, samples: List[AudioSampleData], velocity_mapping: Dict) -> None:
        """Zpracování a export"""
        logger.info("Phase 3: Processing and exporting")

        total_exported = 0
        sample_counters = defaultdict(int)  # Round-robin tracking

        for i, sample in enumerate(samples, 1):
            if sample.assigned_velocity is None or sample.pitch_analysis.fundamental_freq is None:
                logger.warning(f"Skipping sample {i}: missing velocity or pitch data")
                continue

            logger.info(f"[{i}/{len(samples)}] Exporting: {sample.filepath.name}")

            try:
                # Určení cílové MIDI noty
                target_midi = self._find_target_midi_note(sample.pitch_analysis.fundamental_freq)

                if target_midi is None:
                    logger.warning("Cannot determine target MIDI note, skipping")
                    continue

                # Výpočet korekce
                correction_result = self._calculate_pitch_correction(
                    sample.pitch_analysis.fundamental_freq, target_midi
                )

                sample.target_midi_note = target_midi
                sample.pitch_correction_semitones = correction_result["semitones"]

                # Kontrola, zda přeskočit malou korekci
                skip_correction = self._should_skip_correction(correction_result["cents"])
                correction_result["skip_correction"] = skip_correction

                # Round-robin tracking
                rr_key = (target_midi, sample.assigned_velocity)
                rr_index = sample_counters[rr_key]
                sample_counters[rr_key] += 1

                # Logování korekce
                self._log_correction_info(sample, correction_result, rr_index)

                # Aplikace pitch korekce pouze pokud není přeskočena
                if skip_correction:
                    corrected_audio, corrected_sr = sample.waveform, sample.sample_rate
                    logger.info(f"  Using original audio without correction")
                else:
                    corrected_audio, corrected_sr = self._apply_pitch_correction(
                        sample, correction_result["semitones"]
                    )

                # Export pro target sample rates
                export_count = self._export_sample_multiple_formats(
                    sample, corrected_audio, corrected_sr, rr_index
                )

                total_exported += export_count

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                if self.verbose:
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                continue

        logger.info(f"Export complete: {total_exported} files exported")

        # Round-robin statistiky
        self._log_round_robin_statistics(sample_counters)

    def _find_target_midi_note(self, freq: float) -> Optional[int]:
        """Najde nejbližší MIDI notu s lepší validací"""
        try:
            midi_float = 12 * np.log2(freq / 440.0) + 69
            midi_note = int(np.round(midi_float))

            # Validace rozsahu
            if 0 <= midi_note <= 127:
                return midi_note
            else:
                logger.warning(f"MIDI note {midi_note} outside valid range for frequency {freq:.2f} Hz")
                return None

        except (ValueError, OverflowError) as e:
            logger.error(f"Error calculating MIDI note for frequency {freq:.2f} Hz: {e}")
            return None

    def _calculate_pitch_correction(self, detected_freq: float, target_midi: int) -> Dict:
        """Výpočet pitch korekce s dodatečnými informacemi"""
        target_freq = AudioUtils.midi_to_freq(target_midi)
        semitone_correction = 12 * np.log2(target_freq / detected_freq)

        return {
            "semitones": semitone_correction,
            "detected_freq": detected_freq,
            "target_freq": target_freq,
            "cents": semitone_correction * 100,  # Pro jemnější měření
            "is_large_correction": abs(semitone_correction) > 1.0
        }

    def _log_correction_info(self, sample: AudioSampleData, correction: Dict, rr_index: int) -> None:
        """Logování informací o korekci s detailními frekvencemi"""
        note_name = AudioUtils.midi_to_note_name(sample.target_midi_note)

        if rr_index == 0:
            logger.info(f"  MIDI {sample.target_midi_note} ({note_name}) -> velocity {sample.assigned_velocity}")
        else:
            logger.info(f"  MIDI {sample.target_midi_note} ({note_name}) -> velocity {sample.assigned_velocity} [RR {rr_index+1}]")

        # Detailní informace o frekvencích
        logger.info(f"  Detected frequency: {correction['detected_freq']:.2f} Hz")
        logger.info(f"  Target frequency:   {correction['target_freq']:.2f} Hz")
        logger.info(f"  Difference: {correction['detected_freq'] - correction['target_freq']:+.2f} Hz")
        logger.info(f"  Pitch correction: {correction['semitones']:+.3f} semitones ({correction['cents']:+.1f} cents)")

        # Upozornění na velké korekce
        if correction["is_large_correction"]:
            logger.warning(f"  Large correction detected - possible pitch detection error")

        # Info o přeskočení malých korekcí
        if correction.get("skip_correction", False):
            logger.info(f"  Correction skipped (below {self.min_correction_cents:.1f} cent threshold)")

    def _should_skip_correction(self, cents: float) -> bool:
        """Rozhodne, zda přeskočit korekci na základě prahu"""
        return abs(cents) < self.min_correction_cents

    def _apply_pitch_correction(self, sample: AudioSampleData, semitones: float) -> tuple:
        """Aplikace pitch korekce s error handling"""
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
                # Konverze sample rate
                final_audio = AudioProcessor.resample_audio(audio, sr, target_sr)

                # Generování názvu souboru
                output_path = self._generate_filename(
                    sample.target_midi_note, sample.assigned_velocity, sr_suffix, rr_index
                )

                # Export
                sf.write(str(output_path), final_audio, target_sr)
                logger.info(f"  Saved: {output_path.name}")
                exported_count += 1

                # Log duration info
                final_duration = len(final_audio) / target_sr
                logger.info(f"  Duration: {final_duration:.2f}s")

            except Exception as e:
                logger.error(f"Export error for {sr_suffix}: {e}")

        return exported_count

    def _generate_filename(self, midi: int, velocity: int, sr_suffix: str, sample_index: int = 0) -> Path:
        """Generuje strukturovaný název souboru"""
        if sample_index == 0:
            base_name = f"m{midi:03d}-vel{velocity}-{sr_suffix}"
        else:
            base_name = f"m{midi:03d}-vel{velocity}-{sr_suffix}-rr{sample_index+1}"

        return self.output_dir / f"{base_name}.wav"

    def _log_round_robin_statistics(self, sample_counters: Dict) -> None:
        """Logování round-robin statistik"""
        logger.info("\nRound-robin statistics:")
        multi_sample_count = 0

        for (midi, velocity), count in sorted(sample_counters.items()):
            if count > 1:
                note_name = AudioUtils.midi_to_note_name(midi)
                logger.info(f"  MIDI {midi} ({note_name}) velocity {velocity}: {count} samples")
                multi_sample_count += 1

        if multi_sample_count == 0:
            logger.info("  No round-robin samples generated (each note/velocity combination has only 1 sample)")

    def _print_final_statistics(self, samples: List[AudioSampleData], velocity_mapping: Dict) -> None:
        """Vytiskne finální statistiky"""
        logger.info("\n=== FINAL STATISTICS ===")

        # Základní statistiky
        total_samples = len(samples)
        successful_pitch = sum(1 for s in samples if s.pitch_analysis.fundamental_freq is not None)
        successful_velocity = sum(1 for s in samples if s.assigned_velocity is not None)

        logger.info(f"Total input samples: {total_samples}")
        logger.info(f"Successful pitch detection: {successful_pitch}")
        logger.info(f"Successful velocity assignment: {successful_velocity}")

        # Pitch detection statistiky
        if successful_pitch > 0:
            confidences = [s.pitch_analysis.confidence for s in samples if s.pitch_analysis.fundamental_freq is not None]
            avg_confidence = np.mean(confidences)
            logger.info(f"Average pitch confidence: {avg_confidence:.3f}")

            # MIDI range
            midi_notes = [s.pitch_analysis.midi_note for s in samples if s.pitch_analysis.midi_note is not None]
            if midi_notes:
                min_midi, max_midi = min(midi_notes), max(midi_notes)
                min_note = AudioUtils.midi_to_note_name(min_midi)
                max_note = AudioUtils.midi_to_note_name(max_midi)
                logger.info(f"MIDI range: {min_midi} ({min_note}) to {max_midi} ({max_note})")

        # Velocity statistiky
        velocity_counts = defaultdict(int)
        for sample in samples:
            if sample.assigned_velocity is not None:
                velocity_counts[sample.assigned_velocity] += 1

        logger.info("Final velocity distribution:")
        for vel in range(8):
            count = velocity_counts[vel]
            percentage = (count / successful_velocity * 100) if successful_velocity > 0 else 0
            logger.info(f"  Velocity {vel}: {count} samples ({percentage:.1f}%)")


def parse_arguments():
    """Parsování argumentů"""
    parser = argparse.ArgumentParser(
        description="""
Pitch Corrector s pokročilými funkcemi

Klíčové vlastnosti:
- Modulární architektura v několika souborech
- Algoritmy s ochranou proti chybám
- Pokročilé velocity mapping s multiple metrikami
- Robustní error handling a comprehensive logging

Příklad použití:
python sample-tuner.py --input-dir ./samples --output-dir ./output --verbose --detector hybrid
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
                       help='Minimální korekce v centech pro aplikaci ladění (default: 33.33 = 1/3 pultonu)')
    parser.add_argument('--detector', choices=['hybrid', 'simple', 'adaptive'], default='hybrid',
                       help='Typ pitch detectoru (default: hybrid)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging pro debugging')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    try:
        corrector = RefactoredPitchCorrector(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            attack_duration=args.attack_duration,
            pitch_detector_type=args.detector,
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
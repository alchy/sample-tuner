"""
Utility modul pro audio zpracování, včetně datových struktur a procesorů.
Tento soubor obsahuje společné třídy a funkce pro celý systém, jako je konverze MIDI, normalizace audio a směrové ladění.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from enum import Enum
import logging
from pathlib import Path  # Import pro práci s cestami k souborům

logger = logging.getLogger(__name__)

# Globální konfigurace směru ladění
class TuneDirection(Enum):
    """
    Enum pro směr ladění MIDI not.
    """
    UP = "up"        # Vždy ladit směrem nahoru k vyšší MIDI notě
    DOWN = "down"    # Vždy ladit směrem dolů k nižší MIDI notě
    NEAREST = "nearest"  # Ladit k nejbližší MIDI notě (default)

# Globální konfigurace filtru kláves
class KeyFilter(Enum):
    """
    Enum pro filtr kláves na piano.
    """
    ALL = "all"          # Všechny klávesy (bílé i černé)
    WHITE_ONLY = "white"  # Pouze bílé klávesy
    BLACK_ONLY = "black"  # Pouze černé klávesy

class PitchAnalysisResult:
    """
    Výsledek analýzy pitchu, včetně základní frekvence a dalších metrik.
    """
    def __init__(self, fundamental_freq: Optional[float], confidence: float,
                 harmonics: List[float], method_used: str,
                 spectral_centroid: float, spectral_rolloff: float):
        self.fundamental_freq = fundamental_freq  # Základní detekovaná frekvence
        self.confidence = confidence  # Míra spolehlivosti detekce
        self.harmonics = harmonics  # Seznam harmonických frekvencí
        self.method_used = method_used  # Použitá metoda detekce (např. 'crepe_hybrid')
        self.spectral_centroid = spectral_centroid  # Spektrální centroid
        self.spectral_rolloff = spectral_rolloff  # Spektrální rolloff

class VelocityAnalysisResult:
    """
    Výsledek analýzy velocity, včetně RMS, peak a dalších metrik.
    """
    def __init__(self, rms_db: float, peak_db: float, attack_peak_db: float,
                 attack_rms_db: float, dynamic_range: float, stereo_width: float):
        self.rms_db = rms_db  # RMS v dB
        self.peak_db = peak_db  # Peak v dB
        self.attack_peak_db = attack_peak_db  # Peak v attack fázi
        self.attack_rms_db = attack_rms_db  # RMS v attack fázi
        self.dynamic_range = dynamic_range  # Dynamický rozsah
        self.stereo_width = stereo_width  # Šířka sterea

class AudioSampleData:
    """
    Datová struktura pro audio sample, včetně waveformu, analýz a cílových hodnot.
    """
    def __init__(self, filepath: Path, waveform: np.ndarray, sample_rate: int, duration: float,
                 pitch_analysis: Optional[PitchAnalysisResult] = None,
                 velocity_analysis: Optional[VelocityAnalysisResult] = None,
                 assigned_velocity: Optional[int] = None,
                 target_midi_note: Optional[int] = None,
                 pitch_correction_semitones: Optional[float] = None):
        self.filepath = filepath  # Cesta k souboru
        self.waveform = waveform  # Audio data jako numpy array
        self.sample_rate = sample_rate  # Sample rate (Hz)
        self.duration = duration  # Délka v sekundách
        self.pitch_analysis = pitch_analysis  # Výsledek pitch analýzy
        self.velocity_analysis = velocity_analysis  # Výsledek velocity analýzy
        self.assigned_velocity = assigned_velocity  # Přiřazená velocity (0-7)
        self.target_midi_note = target_midi_note  # Cílová MIDI nota
        self.pitch_correction_semitones = pitch_correction_semitones  # Korekce v půltónech

class AudioUtils:
    """
    Utility funkce pro manipulaci s audio daty, jako je konverze, normalizace a MIDI převody.
    """
    @staticmethod
    def to_mono(waveform: np.ndarray) -> np.ndarray:
        """
        Převede stereo waveform na mono průměrováním kanálů.
        """
        if len(waveform.shape) == 2:
            return np.mean(waveform, axis=1)
        return waveform

    @staticmethod
    def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalizuje audio na cílovou úroveň dB pro konzistentní hlasitost.
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return audio
        gain_db = target_db - 20 * np.log10(rms)
        gain = 10 ** (gain_db / 20)
        gain = 10 ** (gain_db / 20)
        return audio * gain

    @staticmethod
    def midi_to_freq(midi_note: int) -> float:
        """
        Převede MIDI notu na frekvenci v Hz (A4 = 440 Hz).
        """
        return 440.0 * (2 ** ((midi_note - 69) / 12))

    @staticmethod
    def midi_to_note_name(midi_note: int) -> str:
        """
        Převede MIDI notu na jméno noty (např. 'C4').
        """
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note // 12) - 1
        note_name = notes[midi_note % 12]
        return f"{note_name}{octave}"

    @staticmethod
    def spectral_features(audio: np.ndarray, sr: int) -> Tuple[float, float]:
        """
        Vypočítá spektrální centroid a rolloff pro analýzu spektra.
        """
        # Dummy implementace – v reálu použij librosa nebo podobné
        centroid = np.mean(np.fft.fftfreq(len(audio), 1/sr) * np.abs(np.fft.fft(audio)))
        rolloff = centroid * 1.5  # Přibližný výpočet
        return centroid, rolloff

    @staticmethod
    def compute_spectrum(audio: np.ndarray, sr: int, n_fft: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vypočítá spektrum audio signálu pomocí FFT.
        """
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))
        spectrum = np.abs(np.fft.fft(audio[:n_fft]))
        freqs = np.fft.fftfreq(n_fft, 1/sr)
        return spectrum[:n_fft//2], freqs[:n_fft//2]

class AudioProcessor:
    """
    Procesor pro pokročilé operace s audio, jako je výpočet stereo šířky.
    """
    @staticmethod
    def calculate_stereo_width(waveform: np.ndarray) -> float:
        """
        Vypočítá šířku stereo signálu na základě mid-side analýzy.
        """
        if len(waveform.shape) != 2 or waveform.shape[1] != 2:
            return 0.0
        left, right = waveform[:, 0], waveform[:, 1]
        mid = (left + right) / 2
        side = (left - right) / 2
        mid_mean = np.mean(np.abs(mid))
        if mid_mean == 0:
            return 0.0
        return np.mean(np.abs(side)) / mid_mean

class DirectionalTuner:
    """
    Pomocná třída pro směrové ladění MIDI not s filtrem kláves podle specifikace v README.
    """
    # MIDI noty pro bílé klávesy (C, D, E, F, G, A, B)
    WHITE_KEY_NOTES = {0, 2, 4, 5, 7, 9, 11}  # Chromatic positions within octave
    # MIDI noty pro černé klávesy (C#, D#, F#, G#, A#)
    BLACK_KEY_NOTES = {1, 3, 6, 8, 10}  # Chromatic positions within octave

    @staticmethod
    def is_white_key(midi_note: int) -> bool:
        """
        Zkontroluje, zda je MIDI nota bílá klávesa.
        """
        return (midi_note % 12) in DirectionalTuner.WHITE_KEY_NOTES

    @staticmethod
    def is_black_key(midi_note: int) -> bool:
        """
        Zkontroluje, zda je MIDI nota černá klávesa.
        """
        return (midi_note % 12) in DirectionalTuner.BLACK_KEY_NOTES

    @staticmethod
    def matches_key_filter(midi_note: int, key_filter: KeyFilter) -> bool:
        """
        Zkontroluje, zda MIDI nota odpovídá zvolenému filtru kláves.
        """
        if key_filter == KeyFilter.ALL:
            return True
        elif key_filter == KeyFilter.WHITE_ONLY:
            return DirectionalTuner.is_white_key(midi_note)
        elif key_filter == KeyFilter.BLACK_ONLY:
            return DirectionalTuner.is_black_key(midi_note)
        return False

    @staticmethod
    def find_target_midi_directional(freq: float, direction: TuneDirection,
                                     piano_frequencies: np.ndarray,
                                     key_filter: KeyFilter = KeyFilter.ALL) -> Optional[int]:
        """
        Najde cílovou MIDI notu podle směru ladění a filtru kláves, s logováním pro debugging.
        """
        try:
            # Vypočet plovoucí MIDI hodnoty z frekvence
            midi_float = 12 * np.log2(freq / 440.0) + 69

            if direction == TuneDirection.NEAREST:
                target_midi = int(np.round(midi_float))
            elif direction == TuneDirection.UP:
                target_midi = int(np.ceil(midi_float))
            else:  # TuneDirection.DOWN
                target_midi = int(np.floor(midi_float))

            # Aplikace filtru kláves, pokud je specifikován
            if key_filter != KeyFilter.ALL:
                target_midi = DirectionalTuner._find_key_in_direction(
                    target_midi, direction, key_filter
                )
                if target_midi is None:
                    key_type = "white" if key_filter == KeyFilter.WHITE_ONLY else "black"
                    logger.warning(f"Cannot find {key_type} key in specified direction")
                    return None

            # Validace MIDI rozsahu (0-127, piano 21-108)
            if not (0 <= target_midi <= 127):
                logger.warning(f"Target MIDI {target_midi} outside valid range")
                return None
            if not (21 <= target_midi <= 108):
                logger.warning(f"Target MIDI {target_midi} outside piano range (21-108)")
                return None

            # Logování výsledku pro debugging
            target_freq = AudioUtils.midi_to_freq(target_midi)
            cents_correction = 1200 * np.log2(target_freq / freq)
            note_name = AudioUtils.midi_to_note_name(target_midi)
            direction_str = direction.value.upper()
            key_filter_str = " (WHITE KEYS ONLY)" if key_filter == KeyFilter.WHITE_ONLY else " (BLACK KEYS ONLY)" if key_filter == KeyFilter.BLACK_ONLY else ""
            key_type = "WHITE" if DirectionalTuner.is_white_key(target_midi) else "BLACK"

            logger.info(f"Directional tuning ({direction_str}{key_filter_str}): {freq:.2f} Hz -> MIDI {target_midi} ({note_name}) = {target_freq:.2f} Hz [{key_type} KEY]")
            logger.info(f"Required correction: {cents_correction:+.1f} cents")

            return target_midi

        except (ValueError, OverflowError) as e:
            logger.error(f"Error in directional MIDI calculation for {freq:.2f} Hz: {e}")
            return None

    @staticmethod
    def _find_key_in_direction(midi_note: int, direction: TuneDirection,
                               key_filter: KeyFilter) -> Optional[int]:
        """
        Najde nejbližší klávesu odpovídajícího typu ve specifikovaném směru, s limitem hledání.
        """
        if DirectionalTuner.matches_key_filter(midi_note, key_filter):
            return midi_note

        search_range = 6  # Maximálně 6 půltónů pro hledání

        if direction == TuneDirection.NEAREST:
            candidates = []

            # Hledání nahoru
            for offset in range(1, search_range + 1):
                candidate = midi_note + offset
                if candidate > 127:
                    break
                if DirectionalTuner.matches_key_filter(candidate, key_filter):
                    candidates.append((candidate, offset))

            # Hledání dolů
            for offset in range(1, search_range + 1):
                candidate = midi_note - offset
                if candidate < 0:
                    break
                if DirectionalTuner.matches_key_filter(candidate, key_filter):
                    candidates.append((candidate, offset))

            if candidates:
                return min(candidates, key=lambda x: x[1])[0]

        elif direction == TuneDirection.UP:
            for offset in range(1, search_range + 1):
                candidate = midi_note + offset
                if candidate > 127:
                    break
                if DirectionalTuner.matches_key_filter(candidate, key_filter):
                    return candidate

        else:  # TuneDirection.DOWN
            for offset in range(1, search_range + 1):
                candidate = midi_note - offset
                if candidate < 0:
                    break
                if DirectionalTuner.matches_key_filter(candidate, key_filter):
                    return candidate

        key_type = "white" if key_filter == KeyFilter.WHITE_ONLY else "black"
        logger.warning(f"Cannot find {key_type} key in direction {direction.value} from MIDI {midi_note}")
        return None

    @staticmethod
    def validate_directional_tuning(detected_freq: float, target_midi: int,
                                    direction: TuneDirection, key_filter: KeyFilter = KeyFilter.ALL) -> bool:
        """
        Validuje, zda směrové ladění a filtr kláves odpovídají požadavkům, s logováním chyb.
        """
        target_freq = AudioUtils.midi_to_freq(target_midi)

        # Validace směru frekvence (pro NEAREST se nevaliduje)
        freq_correct = True
        if direction == TuneDirection.UP:
            freq_correct = target_freq >= detected_freq
            if not freq_correct:
                logger.warning(f"Direction validation failed: UP but target {target_freq:.2f} Hz < detected {detected_freq:.2f} Hz")
        elif direction == TuneDirection.DOWN:
            freq_correct = target_freq <= detected_freq
            if not freq_correct:
                logger.warning(f"Direction validation failed: DOWN but target {target_freq:.2f} Hz > detected {detected_freq:.2f} Hz")

        # Validace filtru kláves
        key_correct = DirectionalTuner.matches_key_filter(target_midi, key_filter)
        if not key_correct:
            note_name = AudioUtils.midi_to_note_name(target_midi)
            if key_filter == KeyFilter.WHITE_ONLY:
                logger.warning(f"White key validation failed: MIDI {target_midi} ({note_name}) is not a white key")
            elif key_filter == KeyFilter.BLACK_ONLY:
                logger.warning(f"Black key validation failed: MIDI {target_midi} ({note_name}) is not a black key")

        return freq_correct and key_correct
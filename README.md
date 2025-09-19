# Two-Phase Piano Sample Tuner

Pokročilý nástroj pro automatickou analýzu, ladění a organizaci piano samplů s využitím moderních algoritmu pitch detection a velocity mappingu.

## Přehled

Program zpracovává kolekci audio samplů rozladěného piano a vytváří z nich konzistentně naladěnou sadu organizovanou podle MIDI not a velocity úrovní. Používá dvou-fázový přístup pro optimální rychlost a přesnost zpracování.

## Architektura systému

```
┌──────────────────┐    ┌───────────────────┐    ┌──────────────────┐
│   Input Audio    │ => │   Two-Phase       │ => │  Organized       │
│   Samples        │    │   Processing      │    │  MIDI Samples    │
│                  │    │                   │    │                  │
│ • Rozladěné      │    │ Phase 1: Amplitude│    │ • Naladěné       │
│ • Nesystematické │    │ Phase 2: Pitch    │    │ • MIDI mapping   │
│ • Různé úrovně   │    │                   │    │ • Velocity tiers │
└──────────────────┘    └───────────────────┘    └──────────────────┘
```

## Fáze zpracování

### Fáze 1: Rychlá analýza amplitud

**Účel:** Vytvoření velocity mappingu na základě celé sady vzorků

**Proces:**
1. **Skenování souborů** - Nalezení všech podporovaných audio formátů (.wav, .flac)
2. **Metadata validace** - Kontrola sample rate (44.1/48/96 kHz), délky (0.05-300s)
3. **Rychlá amplitude analýza** - Výpočet RMS a peak amplitud bez plného načtení
4. **Outlier removal** - Odstranění extrémních hodnot pomocí IQR filtrace
5. **Velocity threshold creation** - Vytvoření 8 velocity úrovní na základě percentilů
6. **Velocity assignment** - Přiřazení velocity každému samplu

**Výstupy Fáze 1:**
- Velocity mapping s threshold hodnotami
- Každý vzorek má přiřazenou velocity (0-7)
- Statistika distribuce velocity

### Fáze 2: Pitch detection a korekce

**Účel:** Přesná detekce frekvencí, cílových MIDI not a aplikace pitch korekcí

**Proces:**

#### 2.1 Inicializace detektoru
- **CREPE Hybrid Detector** - Kombinace neuronové sítě CREPE s tradičními algoritmy
- **Backup detektory** - Aubio YIN, HPS, Autocorrelation pro validaci
- **Piano frequency table** - 88 přesných piano frekvencí pro validaci

#### 2.2 Pro každý audio soubor:

**A. Načtení a preprocessing:**
```
Raw Audio → Float32 → Mono conversion → Normalization (-20dB) 
    ↓
High-pass filter (40Hz) → Notch filters (50/60Hz) → Low-pass (12kHz)
    ↓
Pre-emphasis filter → Final preprocessing
```

**B. Pitch detection (Multi-algorithm ensemble):**
1. **CREPE Neural Network** (Primary)
   - State-of-the-art přesnost
   - Temporal smoothing s Viterbi algoritmem
   - Filtrování na piano rozsah (27.5-4186 Hz)

2. **Backup detektory** (pokud CREPE selže nebo má nízkou confidence):
   - Aubio YIN - nejspolehlivější tradiční algoritmus
   - HPS (Harmonic Product Spectrum) - dobrý pro komplexní harmonické
   - Autocorrelation - backup pro edge cases

3. **Ensemble decision:**
   - Octave error correction - inteligentní detekce a oprava oktávových chyb
   - Spectral validation - ověření frekvencí proti audio spektru
   - Weighted voting podle confidence a reliability

**C. Target MIDI note determination:**
```
Detected Frequency → Find closest piano frequency → Calculate MIDI note
    ↓
Handle detuned piano (tolerance ±120 cents)
    ↓
Target MIDI note (21-108, A0-C8)
```

**D. Pitch correction calculation:**
```
Target Frequency = 440 * 2^((MIDI-69)/12)
Correction = 12 * log2(Target/Detected)
    ↓
Skip if correction < min_threshold (default 33.33 cents)
    ↓
Apply pitch shift using resampling
```

**E. Export processing:**
1. **Multi-format export** - 44.1 kHz a 48 kHz verze
2. **Round-robin tracking** - Více vzorků pro stejnou MIDI+velocity kombinaci
3. **Filename generation** - `mXXX-velY-fZZ[-rrN].wav`
   - `XXX` = MIDI note (000-127)
   - `Y` = velocity level (0-7)
   - `ZZ` = sample rate (f44/f48)
   - `N` = round-robin index (volitelné)

## Detaily algoritmů

### CREPE Neural Network
- **Model:** Large (nejvyšší přesnost)
- **Step size:** 10ms pro balance rychlost/přesnost
- **Temporal smoothing:** Viterbi algoritmus pro konzistenci
- **Confidence threshold:** Adaptivní podle piano charakteristik

### Octave Error Correction
```
For each detection result:
    Group similar frequencies (±5% tolerance)
    Check octave relationships (0.5x, 2x, 4x, 8x)
    Apply intelligent correction based on:
        - Spectral energy analysis
        - Piano frequency proximity
        - Detection confidence
        - Historical consistency
```

### Velocity Mapping
```
RMS Amplitude Distribution → Percentile Analysis → 8 Velocity Levels

Level 0: 0-12.5 percentile    (nejtiší)
Level 1: 12.5-25 percentile
Level 2: 25-37.5 percentile
Level 3: 37.5-50 percentile
Level 4: 50-62.5 percentile
Level 5: 62.5-75 percentile
Level 6: 75-87.5 percentile
Level 7: 87.5-100 percentile  (nejhlasitější)
```

## Instalace a spuštění

### Požadavky
```bash
pip install numpy scipy soundfile resampy aubio tensorflow crepe
```

### Základní použití
```bash
python sample-tuner.py \
    --input-dir "path/to/raw/samples" \
    --output-dir "path/to/output" \
    --verbose
```

### Pokročilé parametry
```bash
python sample-tuner.py \
    --input-dir "samples/" \
    --output-dir "tuned/" \
    --attack-duration 0.5 \
    --min-correction-cents 33.33 \
    --verbose
```

**Parametry:**
- `--attack-duration` - Délka attack fáze pro velocity analýzu (default: 0.5s)
- `--min-correction-cents` - Minimální korekce pro aplikaci (default: 33.33 = 1/3 půltónu)
- `--verbose` - Detailní logování pro debugging

## Výstupní struktura

```
output_dir/
├── m060-vel0-f44.wav          # C4, velocity 0, 44.1kHz
├── m060-vel0-f48.wav          # C4, velocity 0, 48kHz
├── m060-vel1-f44.wav          # C4, velocity 1, 44.1kHz
├── m060-vel1-f44-rr2.wav      # C4, velocity 1, round-robin 2
├── m072-vel3-f44.wav          # C5, velocity 3, 44.1kHz
└── ...
```

## Logování a monitoring

### Fáze 1 - Amplitude Analysis
```
[1/348] Analyzing: sample001.wav
  Max amplitude: 0.8245
  RMS amplitude: 0.1234
  Duration: 3.45s
```

### Fáze 2 - Pitch Detection
```
[1/348] Processing & exporting: sample001.wav
  Detected: 286.72 Hz, Closest piano: 293.66 Hz
  Piano detuning: -39.1 cents
  Validation: PASS (CREPE detection within ±120 cents)
  MIDI 62 (D4) -> velocity 3
  Pitch correction: +0.423 semitones (+39.1 cents)
  Applying pitch correction
  Saved: m062-vel3-f44.wav
  Saved: m062-vel3-f48.wav
```

### Finální statistiky
```
=== FINAL STATISTICS ===
Phase 1 (Amplitude): 348 samples analyzed
Phase 2 (Pitch): 312 samples processed
Success rate: 89.7%

Velocity distribution:
  Velocity 0: 43 samples (12.4%)
  Velocity 1: 41 samples (11.8%)
  ...

Average pitch confidence: 0.847
```

## Řešení problémů

### Časté problémy

**1. Nízká úspěšnost detection**
- Zkontrolujte kvalitu audio (noise, reverberation)
- Zvyšte `--min-correction-cents` pro tolerance rozladěných nástrojů
- Použijte `--verbose` pro analýzu failed detections

**2. TensorFlow/CREPE chyby**
```bash
pip install tensorflow
# nebo použijte CPU-only verzi:
pip install tensorflow-cpu
```

**3. Nesprávné oktávy**
- Program automaticky opravuje oktávové chyby
- Zkontrolujte piano frequency validation v logs
- CREPE by měl vyřešit většinu oktávových problémů

**4. Špatná velocity distribuce**
- Zkontrolujte amplitude range vstupních samplů
- Outlier removal může odstranit extrémní hodnoty
- Upravte attack_duration pro různé typy piano

### Debug režim
```bash
python sample-tuner.py --input-dir samples --output-dir out --verbose
```

Zobrazí detailní informace o:
- CREPE neural network detection
- Backup detector results
- Octave correction decisions
- Spectral analysis
- Confidence scoring

## Technické detaily

### Podporované formáty
- **Input:** WAV, FLAC (44.1/48/96 kHz)
- **Output:** WAV (44.1 kHz a 48 kHz)

### Frequency rozsah
- **Piano range:** 27.5 - 4186 Hz (A0 - C8)
- **88 piano keys** s přesnými temperovanými frekvencemi

### Performance
- **Fáze 1:** ~1-2 sekundy na vzorek (pouze metadata + rychlá amplitude)
- **Fáze 2:** ~10-15 sekund na vzorek (CREPE + full processing)
- **Memory:** ~2-4 GB pro CREPE model + audio buffer

### Precision
- **Pitch detection:** ±5-10 cents typicky s CREPE
- **Velocity mapping:** 8 úrovní s adaptivní distribucí
- **Detuning tolerance:** ±120 cents (1.2 půltónu)

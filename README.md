# Sample Tuner

Pokročilý pitch corrector pro audio vzorky s automatickým laděním a velocity mappingem.

## Funkce

- **Hybridní pitch detection** - kombinuje spektrální analýzu a YIN algoritmus pro přesnou detekci základní frekvence
- **Automatické ladění** - koriguje pitch na nejbližší MIDI notu
- **Konfigurovatelný práh** - možnost přeskočit malé korekce (např. menší než 1/3 pultonu)
- **Velocity mapping** - automatické přiřazení velocity hodnot podle attack charakteristik
- **Round-robin vzorky** - generuje více vzorků pro stejnou MIDI notu a velocity
- **Multiple sample rates** - export do 44.1 kHz a 48 kHz formátů
- **Robustní error handling** - pokračuje ve zpracování i při chybách jednotlivých souborů

## Instalace

### Požadavky

```bash
pip install numpy soundfile resampy
```

### Struktura souborů

```
sample-tuner/
├── audio_utils.py          # Základní audio utility funkce
├── pitch_detection.py      # Pitch detection algoritmy
├── velocity_analysis.py    # Velocity analýza a mapping
├── sample-tuner.py         # Hlavní program
└── README.md
```

## Použití

### Základní použití

```bash
python sample-tuner.py --input-dir ./samples --output-dir ./output
```

### Všechny parametry

```bash
python sample-tuner.py \
  --input-dir /path/to/input/samples \
  --output-dir /path/to/output \
  --attack-duration 0.5 \
  --min-correction-cents 33.33 \
  --detector hybrid \
  --verbose
```

### Parametry

| Parametr | Popis | Default | Možnosti |
|----------|-------|---------|-----------|
| `--input-dir` | Vstupní adresář s audio soubory | **povinný** | |
| `--output-dir` | Výstupní adresář | **povinný** | |
| `--attack-duration` | Délka attack fáze pro velocity detection (sekundy) | `0.5` | `0.1` - `2.0` |
| `--min-correction-cents` | Minimální korekce v centech pro aplikaci ladění | `33.33` | `0` - `100` |
| `--detector` | Typ pitch detectoru | `hybrid` | `hybrid`, `simple`, `adaptive` |
| `--verbose` | Detailní logování | `false` | |

### Podporované formáty

- **Vstup**: WAV, FLAC (mono i stereo)
- **Sample rates**: 44.1 kHz, 48 kHz, 96 kHz
- **Výstup**: WAV (44.1 kHz a 48 kHz verze)

## Příklady použití

### Standardní ladění s přeskočením malých korekcí

```bash
python sample-tuner.py --input-dir ./samples-in --output-dir ./samples-out
```
Přeskočí korekce menší než 33.33 centů (1/3 pultonu).

### Citlivé ladění

```bash
python sample-tuner.py --input-dir ./samples-in --output-dir ./samples-out --min-correction-cents 10
```
Aplikuje i malé korekce, přeskočí pouze korekce menší než 10 centů.

### Vždy ladit

```bash
python sample-tuner.py --input-dir ./samples-in --output-dir ./samples-out --min-correction-cents 0
```
Aplikuje všechny korekce bez ohledu na velikost.

### Rychlé zpracování

```bash
python sample-tuner.py --input-dir ./samples-in --output-dir ./samples-out --detector simple
```
Použije rychlejší, ale méně přesný algoritmus.

## Výstup

### Struktura výstupních souborů

```
output/
├── m048-vel0-f44.wav     # MIDI 48, velocity 0, 44.1kHz
├── m048-vel0-f48.wav     # MIDI 48, velocity 0, 48kHz
├── m048-vel1-f44.wav     # MIDI 48, velocity 1, 44.1kHz
├── m048-vel1-f48.wav     # MIDI 48, velocity 1, 48kHz
├── m048-vel1-f44-rr2.wav # Round-robin vzorek 2
└── ...
```

### Logování

Program vypíše detailní informace o každém vzorku:

```
[1/590] Processing: sample.wav
  Pitch: 128.91 Hz -> MIDI 48 (C3)
  Confidence: 1.000, Method: hybrid_spectral_yin
  Attack Peak: -15.65 dB
  Harmonics: 7
  Detected frequency: 128.91 Hz
  Target frequency:   130.81 Hz
  Difference: +1.90 Hz
  Pitch correction: +0.024 semitones (+2.4 cents)
  Correction skipped (below 33.3 cent threshold)
  Using original audio without correction
```

## Algoritmy

### Pitch Detection

**Hybrid detector** (doporučený):
- Kombinuje spektrální analýzu s YIN algoritmem
- Vysoká přesnost i pro složité harmonické struktury
- Automatická validace výsledků

**Simple detector**:
- Pouze spektrální analýza
- Rychlejší zpracování
- Vhodný pro jednoduché tóny

### Velocity Mapping

Program analyzuje attack charakteristiky každého vzorku:
- **Attack Peak** - maximální úroveň v attack fázi
- **Attack Slope** - strmost nárůstu
- **Dynamic Range** - rozdíl mezi peak a RMS

Vzorky jsou automaticky rozděleny do 8 velocity úrovní (0-7).

## Řešení problémů

### Program je pomalý
- Použijte `--detector simple` pro rychlejší zpracování
- Zvyšte `--min-correction-cents` pro přeskočení více korekcí

### Špatná pitch detekce
- Zkuste jiný detector: `--detector adaptive`
- Zkontrolujte kvalitu vstupních souborů
- Použijte `--verbose` pro detailní analýzu

### Chyby při zpracování
- Program pokračuje i při chybách jednotlivých souborů
- Zkontrolujte logy pro specifické chybové zprávy
- Ověřte, že vstupní soubory nejsou poškozené

### Neočekávané velocity přiřazení
- Použijte `--verbose` pro analýzu velocity distribuce
- Upravte `--attack-duration` podle typu materiálu

## Technické detaily

### Pitch Correction
- Používá time-stretching pro změnu pitch
- Zachovává původní délku vzorků
- Resample do cílových sample rates

### Memory Management
- Zpracování po jednom souboru
- Automatické uvolňování paměti
- Podpora velkých souborů

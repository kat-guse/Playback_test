# The following code takes details regarding the timings of playbacks (2025_PLAYBACKS_copy.csv), locates the corresponding file in the directory (synced_base_dir) 
#and detects 1000Hz square-wave harmonic ladder beeps. 

import os
import glob
import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf

# 1. File Paths
summary_csv_path = "/Users/katrine/Desktop/2025_PLAYBACKS_copy.csv"
synced_base_dir  = "/Volumes/BUworkspace/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/Synced_Files"
out_dir          = "/Volumes/BUworkspace/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/Playback_Analysis"

df_summary = pd.read_csv(summary_csv_path)

def resolve_wav_path(row, base_dir):
    grid     = str(row['grid']).strip()
    host_aru = str(row['host_ARU']).strip()
    fname    = str(row['filenames']).strip()
    base_nm  = fname[:-4] if fname.lower().endswith('.wav') else fname

    for ext in ['.WAV', '.wav']:
        candidate = os.path.join(base_dir, f"{grid}-Sync2", f"{grid}-{host_aru}", f"{base_nm}{ext}")
        if os.path.exists(candidate):
            return candidate

    matches = glob.glob(os.path.join(base_dir, f"{grid}*", "**", f"{base_nm}.*"), recursive=True)
    wav_matches = [m for m in matches if m.lower().endswith('.wav')]
    if wav_matches:
        return wav_matches[0]
    raise FileNotFoundError(f"Audio file not found: {base_nm}")

def detect_1khz_harmonic_beeps(audio_path, f0_hz=1000, min_dur_s=0.15, max_dur_s=1.5, comb_thresh_db=10.0):
    """
    Detects 1000 Hz square-wave harmonic ladder beeps (1, 2, ..., 10 kHz rungs vs 1.5, 2.5, ... valleys).
    """
    audio, fs = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    nperseg  = int(fs * 0.04)   # 40ms window
    noverlap = int(fs * 0.03)   # 10ms step (100 Hz temporal resolution)
    freqs, times, Sxx = signal.spectrogram(audio, fs=fs, nperseg=nperseg, noverlap=noverlap)

    harmonics = np.arange(f0_hz, min(10500, fs / 2 - 300), f0_hz)  # 1000, 2000, ..., 10000 Hz
    gaps      = harmonics + (f0_hz / 2.0)                          # 1500, 2500, ..., 9500 Hz

    rung_mask = np.zeros_like(freqs, dtype=bool)
    gap_mask  = np.zeros_like(freqs, dtype=bool)

    for h in harmonics:
        rung_mask |= (np.abs(freqs - h) <= 50)
    for g in gaps:
        gap_mask |= (np.abs(freqs - g) <= 150)

    rung_power  = np.mean(Sxx[rung_mask, :], axis=0)
    gap_power   = np.mean(Sxx[gap_mask, :], axis=0) + 1e-12
    comb_snr_db = 10 * np.log10(rung_power / gap_power)

    is_beep = comb_snr_db >= comb_thresh_db
    padded  = np.concatenate(([False], is_beep, [False]))
    starts  = np.where(padded[1:] & ~padded[:-1])[0]
    ends    = np.where(~padded[1:] & padded[:-1])[0]

    beeps = []
    for s_idx, e_idx in zip(starts, ends):
        start_s = times[s_idx]
        end_s   = times[min(e_idx, len(times) - 1)]
        dur_s   = end_s - start_s
        if min_dur_s <= dur_s <= max_dur_s:
            beeps.append({
                'beep_start_s': round(float(start_s), 3),
                'beep_end_s': round(float(end_s), 3),
                'duration_s': round(float(dur_s), 3),
                'comb_snr_db': round(float(np.max(comb_snr_db[s_idx:e_idx])), 2)
            })
    return beeps

# 2. Run Across All Host Files
file_summaries = []
individual_beeps = []

for idx, row in df_summary.iterrows():
    fname = row['filenames']
    try:
        wav_path = resolve_wav_path(row, synced_base_dir)
        beeps = detect_1khz_harmonic_beeps(wav_path)

        if len(beeps) == 0:
            print(f"[WARNING] Row {idx} ({fname}): No 1 kHz harmonic beeps detected above threshold.")
            file_summaries.append({
                'row_id': idx, 'grid': row['grid'], 'host_ARU': row['host_ARU'],
                'track': row['track'], 'filename': fname, 'n_beeps_found': 0,
                'first_beep_start_s': None, 'last_beep_end_s': None, 'status': 'NO_BEEPS_FOUND'
            })
        else:
            first_s = beeps[0]['beep_start_s']
            last_e  = beeps[-1]['beep_end_s']
            print(f"[SUCCESS] Row {idx} ({fname}): Found {len(beeps)} beeps | First beep = {first_s:.3f}s")

            file_summaries.append({
                'row_id': idx, 'grid': row['grid'], 'host_ARU': row['host_ARU'],
                'track': row['track'], 'filename': fname, 'n_beeps_found': len(beeps),
                'first_beep_start_s': first_s, 'last_beep_end_s': last_e, 'status': 'SUCCESS'
            })
            for b_i, b in enumerate(beeps, start=1):
                individual_beeps.append({
                    'row_id': idx, 'grid': row['grid'], 'host_ARU': row['host_ARU'],
                    'track': row['track'], 'filename': fname, 'beep_num': b_i, **b
                })

    except Exception as e:
        print(f"[ERROR] Row {idx} ({fname}): {e}")
        file_summaries.append({
            'row_id': idx, 'grid': row['grid'], 'host_ARU': row['host_ARU'],
            'track': row['track'], 'filename': fname, 'n_beeps_found': None,
            'first_beep_start_s': None, 'last_beep_end_s': None, 'status': f"ERROR: {e}"
        })

os.makedirs(out_dir, exist_ok=True)
pd.DataFrame(file_summaries).to_csv(os.path.join(out_dir, "detected_playback_beep_summary.csv"), index=False)
pd.DataFrame(individual_beeps).to_csv(os.path.join(out_dir, "detected_individual_beeps.csv"), index=False)
print("\nSaved both summary and individual beep timestamp CSVs.")

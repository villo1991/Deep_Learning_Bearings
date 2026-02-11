import numpy as np
from scipy.signal import spectrogram
import os
import gc

# =========================
# 1. Set parameters
# =========================
BASE_DIR = "....../paderborn_university/dataset"
BASE_DIR_S = "....../paderborn_university/dataset/spectrograms_FixedScale"
FS = 64000
N_FFT = 1080
HOP_LENGTH = 270
TARGET_SHAPE = (160, 112)

# --- FIXED SCALE PARAMETERS (SoX Style) ---
# Absolute Normalization: -90dB (Black) to +10dB (White/Red)
REF_DB_MAX = 10.0
REF_DB_MIN = -90.0

os.makedirs(BASE_DIR_S, exist_ok=True)

# =========================
# 2. Function Spectrograms (Fixed Scale)
# =========================

def array_to_spectrogram_fixed(signal: np.ndarray):
    """
    Generates a spectrogram with ABSOLUTE scale.
    Output shape: (160, 112, 1) normalized between 0 and 1.
    """
    # Flattening safe
    if signal.ndim != 1:
        signal = signal.flatten()

    f, t, Sxx = spectrogram(
        signal, fs=FS, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH,
        mode="psd", scaling="density", window="hann"
    )

    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)

    Sxx_db = np.clip(Sxx_db, REF_DB_MIN, REF_DB_MAX)

    Sxx_norm = (Sxx_db - REF_DB_MIN) / (REF_DB_MAX - REF_DB_MIN)


    Sxx_norm = Sxx_norm[:TARGET_SHAPE[0], :]

    if Sxx_norm.shape[1] > TARGET_SHAPE[1]:
        Sxx_norm = Sxx_norm[:, :TARGET_SHAPE[1]]
    elif Sxx_norm.shape[1] < TARGET_SHAPE[1]:
        pad_width = TARGET_SHAPE[1] - Sxx_norm.shape[1]
        Sxx_norm = np.pad(Sxx_norm, ((0, 0), (0, pad_width)), mode='constant')

    return Sxx_norm[..., np.newaxis].astype(np.float32)


# =========================
# 3. Process Batch
# =========================

def process_and_save(filename_in, filename_out, description):
    path_in = os.path.join(BASE_DIR, filename_in)
    path_out = os.path.join(BASE_DIR_S, filename_out)

    print(f"\n--- Elaboration: {description} ---")
    if not os.path.exists(path_in):
        print(f" File not found: {path_in}")
        return

    print(f"Loading raw data from {filename_in}...")

    try:
        data = np.load(path_in, mmap_mode='r')
    except Exception as e:
        print(f"Loading Error: {e}")
        return

    n_samples = len(data)
    print(f"Mapped Dataset. Samples: {n_samples}. Shape input: {data.shape}")

    output_shape = (n_samples, TARGET_SHAPE[0], TARGET_SHAPE[1], 1)

    try:
        X_spect = np.zeros(output_shape, dtype=np.float32)
    except MemoryError:
        print(" Error: Impossible allocate output array in RAM.")
        print("Tips: Usa chunks smoler.")
        return

    print("Generation Spectrograms (Fix scale)...")

    for i in range(n_samples):
        X_spect[i] = array_to_spectrogram_fixed(data[i])

        if (i + 1) % 2000 == 0:
            print(f"\r -> {i + 1}/{n_samples} complete", end="")

    print(f"\n Salving: {filename_out}")
    np.save(path_out, X_spect)

    # Clean
    del data
    del X_spect
    gc.collect()
    print(" Clean ram  .")


if __name__ == "__main__":

    # 1. Train Standard
    process_and_save(
        "X_train_dataset_vibration_raw_0.5.npy",
        "X_train_spect_fixed.npy",
        "Train Standard"
    )

    # 2. Test Standard
    process_and_save(
        "X_test_dataset_vibration_raw_0.5.npy",
        "X_test_spect_fixed.npy",
        "Test Standard"
    )

    # 3. Train Augmented
    process_and_save(
        "X_train_dataset_vibration_raw_aug_0.5s.npy",
        "X_train_spect_aug_fixed.npy",
        "Train Augmented"
    )

    print("\n Script complete.")
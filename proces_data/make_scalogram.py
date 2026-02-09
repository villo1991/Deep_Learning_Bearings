import numpy as np
import pywt
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# ===============================
# Funzione per generare scalogramma
# ===============================
def array_to_scalogram(signal, wavelet='morl', scales=np.arange(1, 129)):
    """
    Converte un segnale 1D in scalogramma 2D normalizzato.

    Output shape: (num_scales, len(signal), 1)
    """
    # CWT: coeff. complessi
    coeffs, _ = pywt.cwt(signal, scales, wavelet)

    # Magnitudine
    S = np.abs(coeffs)

    # Log-scale per stabilità numerica
    S = np.log10(S + 1e-10)

    # Normalizzazione 0–1 (min/max del singolo scalogramma)
    S_min, S_max = S.min(), S.max()
    S = (S - S_min) / (S_max - S_min + 1e-10)

    # Aggiungo canale per CNN
    return S.astype(np.float32)[..., np.newaxis]


# =====================================
# Preprocessing completo dataset scalogrammi
# =====================================
def preprocess_scalogram_dataset(X, Y, wavelet='morl', scales=np.arange(1,129),
                                 test_size=0.2, random_state=42):
    """
    Genera scalogrammi 2D tensoriali, split train/test e one-hot encoding.
    """
    scalograms = []

    for i in range(len(X)):
        S = array_to_scalogram(X[i], wavelet=wavelet, scales=scales)
        scalograms.append(S)

    scalograms = np.stack(scalograms)

    # Split stratificato
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        scalograms, Y, test_size=test_size,
        random_state=random_state, stratify=Y
    )

    # One-hot
    num_classes = len(np.unique(Y))
    y_train = keras.utils.to_categorical(y_train_int, num_classes)
    y_test = keras.utils.to_categorical(y_test_int, num_classes)

    return X_train, X_test, y_train, y_test


# ===============================
# Uso pratico
# ===============================
X = np.load("/home/villo/projects/bearings_project/paderborn_university/dataset/X_dataset_vibration_aug_2s.npy"
).astype(np.float32)

Y = np.load("/home/villo/projects/bearings_project/paderborn_university/dataset/Y_dataset_vibration_aug_2s.npy"
).astype(np.uint8).squeeze()

# Preprocessing
X_train, X_test, y_train, y_test = preprocess_scalogram_dataset(X, Y,
                                                                wavelet='morl',
                                                                scales=np.arange(1,129))

# Salvataggio
np.save("X_train_scalogram.npy", X_train)
np.save("X_test_scalogram.npy", X_test)
np.save("Y_train_scalogram.npy", y_train)
np.save("Y_test_scalogram.npy", y_test)

print("Scalogrammi generati e salvati!")

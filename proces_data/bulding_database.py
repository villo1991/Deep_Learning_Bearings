import numpy as np
import random
from collections import Counter

# ==========================================================
# SET UP
# ==========================================================
fixed_signal_length = 250600
slice_dimension = 5012
augmentation_enabled = False
p_distortion = 0.55

target_distribution = {
    0: 30000,
    1: 30000,
    2: 30000,
    3: 30000,
    4: 30000,
    5: 30000,
    6: 30000,
    7: 30000,
}

np.random.seed(42)
random.seed(42)

# ==========================================================
# AUGMENTATION
# ==========================================================
def distortion_raw_vibration(array: np.ndarray, p: float = 0.5):
    """
    :param array: Array to distortion
    :param p: probability distortion
    :return: distorted version of the array
    """
    d = array.astype(np.float32).copy()

    if np.random.rand() < p:
        d = np.roll(d, np.random.randint(0, len(d)))

    if np.random.rand() < p:
        noise = np.random.normal(0.0, 0.01 * np.std(d), d.shape)
        d += noise

    if np.random.rand() < p:
        d *= np.random.uniform(0.9, 1.1)

    return d


# ==========================================================
# Loading
# ==========================================================
dataset = np.load(
    "....../paderborn_university/dataset/test_dataset_vibration.npz",
    allow_pickle=True
)

X_v_raw = dataset["X_raw"]
Y_v_raw = dataset["y"]

print("Original distribution:", Counter(Y_v_raw))


# ==========================================================
# 1️⃣  TRONCAMENTO A LUNGHEZZA FISSA
# ==========================================================
X_fixed = []
Y_fixed = []

for signal, label in zip(X_v_raw, Y_v_raw):

    if len(signal) >= fixed_signal_length:
        truncated = signal[:fixed_signal_length]
        X_fixed.append(truncated)
        Y_fixed.append(label)

# conversione array
X_fixed = np.array(X_fixed, dtype=np.float32)
Y_fixed = np.array(Y_fixed, dtype=np.uint8)

print("Distribution after clipping:", Counter(Y_fixed))
print(f"Length used: {fixed_signal_length}")

# controllo divisibilità
if fixed_signal_length % slice_dimension != 0:
    raise ValueError(
        "fixed_signal_length has to be a multiple of slice dimension"
    )

n_slices_per_signal = fixed_signal_length // slice_dimension
print(f"Number of slice per signal: {n_slices_per_signal}")


# ==========================================================
# 2️⃣  CREAZIONE CHUNK
# ==========================================================
chunks = []
labels = []

for signal, label in zip(X_fixed, Y_fixed):

    for i in range(n_slices_per_signal):
        start = i * slice_dimension
        end = start + slice_dimension
        chunks.append(signal[start:end])
        labels.append(label)

chunks = np.array(chunks, dtype=np.float32)
labels = np.array(labels, dtype=np.uint8)

print("Distribution after slicing:", Counter(labels))


# ==========================================================
# 3️⃣  AUGMENTATION (OPZIONALE)
# ==========================================================
if augmentation_enabled:

    final_chunks = list(chunks)
    final_labels = list(labels)

    current_distribution = Counter(labels)

    for class_id, target_count in target_distribution.items():

        current_count = current_distribution[class_id]

        if current_count >= target_count:
            continue

        print(f"Augment class {class_id}: {current_count} → {target_count}")

        class_chunks = chunks[labels == class_id]

        while current_count < target_count:
            base_chunk = random.choice(class_chunks)
            aug_chunk = distortion_raw_vibration(base_chunk, p=p_distortion)

            final_chunks.append(aug_chunk)
            final_labels.append(class_id)

            current_count += 1

    final_chunks = np.array(final_chunks, dtype=np.float32)
    final_labels = np.array(final_labels, dtype=np.uint8)

else:
    print("Augmentation disable.")
    final_chunks = chunks
    final_labels = labels


print("Distribution final:", Counter(final_labels))


# ==========================================================
# SALVE
# ==========================================================
suffix = "aug" if augmentation_enabled else "noaug"

np.save(
    f"......./paderborn_university/dataset/X_test_raw_{slice_dimension}_{suffix}.npy",
    final_chunks
)

np.save(
    f"......./paderborn_university/dataset/Y_test_raw_{slice_dimension}_{suffix}.npy",
    final_labels
)

print("Dataset save correct.")

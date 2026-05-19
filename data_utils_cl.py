import os
from glob import glob
import numpy as np

# ---------------------------------------------------------------------------- #
# Mapeamento: pasta numérica (str) -> índice de classe (int)
# Pastas 1-25 = classes 0-24 (elementos da lista)
# Classe 25 = "unknown" (sem pasta de treino; usada apenas na inferência)
# ---------------------------------------------------------------------------- #

CLASS_NAMES = [
    "single door",      # 0  (pasta 1)
    "double door",      # 1  (pasta 2)
    "sliding door",     # 2  (pasta 3)
    "folding door",     # 3  (pasta 4)
    "window",           # 4  (pasta 5)
    "bay window",       # 5  (pasta 6)
    "blind window",     # 6  (pasta 7)
    "opening symbol",   # 7  (pasta 8)
    "sofa",             # 8  (pasta 9)
    "bed",              # 9  (pasta 10)
    "chair",            # 10 (pasta 11)
    "table",            # 11 (pasta 12)
    "TV cabinet",       # 12 (pasta 13)
    "wardrobe",         # 13 (pasta 14)
    "gas stove",        # 14 (pasta 15)
    "sink",             # 15 (pasta 16)
    "refrigerator",     # 16 (pasta 17)
    "air conditioner",  # 17 (pasta 18)
    "bath",             # 18 (pasta 19)
    "bath tub",         # 19 (pasta 20)
    "washing machine",  # 20 (pasta 21)
    "squat toilet",     # 21 (pasta 22)
    "urinal",           # 22 (pasta 23)
    "toilet",           # 23 (pasta 24)
    "stairs",           # 24 (pasta 25)
    "unknown",          # 25 (sem pasta – apenas inferência)
]

NUM_CLASSES = len(CLASS_NAMES)  # 26


def get_class_mapping():
    """Retorna dict {str(pasta): class_idx} para pastas 1-25."""
    return {str(i): i - 1 for i in range(1, 26)}


def load_folder_data(split, class_mapping, base_path):
    image_paths = []
    labels = []

    dataset_path = os.path.join(base_path, split)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Caminho não encontrado: {dataset_path}")

    for folder in sorted(os.listdir(dataset_path), key=lambda x: int(x) if x.isdigit() else -1):
        if not folder.isdigit():
            continue

        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        class_idx = class_mapping.get(folder)
        if class_idx is None:
            continue

        imgs = glob(os.path.join(folder_path, "*.*"))
        imgs = [p for p in imgs if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

        image_paths.extend(imgs)
        labels.extend([class_idx] * len(imgs))

    return np.array(image_paths), np.array(labels)


def load_dataset(base_path, seed=42):
    np.random.seed(seed)
    class_mapping = get_class_mapping()

    train_paths, train_labels = load_folder_data("train", class_mapping, base_path)
    val_paths, val_labels     = load_folder_data("val",   class_mapping, base_path)

    # Embaralha treino
    idx = np.random.permutation(len(train_paths))
    train_paths  = train_paths[idx]
    train_labels = train_labels[idx]

    return train_paths, train_labels, val_paths, val_labels

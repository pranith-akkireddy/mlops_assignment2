"""Cats vs Dogs preprocessing.

Images are decoded to the canonical 224x224 RGB representation required for
standard CNN pipelines. Because the baseline classifier is a logistic
regression, the 224x224x3 tensor (150,528 features) is downscaled to
MODEL_INPUT_SIZE before flattening -- otherwise the design matrix is
intractable for a linear model. Both stages are explicit and configurable.
"""

import io
import os
import random

import joblib
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(_BASE, "data", "raw")
PROC_DIR = os.path.join(_BASE, "data", "processed")

# Canonical decode size mandated for standard CNN inputs.
IMAGE_SIZE = 224
# Downscaled edge actually fed to the linear baseline.
MODEL_INPUT_SIZE = int(os.environ.get("MODEL_INPUT_SIZE", "32"))

CLASS_NAMES = ("cat", "dog")
CLASS_DIRS = {"Cat": 0, "Dog": 1}
_VALID_EXT = (".jpg", ".jpeg", ".png")


def discover_images(root=None, max_per_class=None, seed=42):
    """Return (paths, labels) for readable images under <root>/Cat and <root>/Dog.

    The Kaggle corpus ships a handful of truncated/zero-byte JPEGs, so every
    candidate is verified before inclusion.
    """
    root = root or os.path.join(RAW_DIR, "PetImages")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Image root not found at {root}. Run scripts/fetch_data.py first."
        )
    rng = random.Random(seed)
    paths, labels = [], []
    for class_dir, label in sorted(CLASS_DIRS.items()):
        class_path = os.path.join(root, class_dir)
        if not os.path.isdir(class_path):
            raise FileNotFoundError(f"Expected class directory {class_path}")
        names = sorted(
            n for n in os.listdir(class_path) if n.lower().endswith(_VALID_EXT)
        )
        kept = []
        for name in names:
            full = os.path.join(class_path, name)
            if _is_readable_image(full):
                kept.append(full)
        rng.shuffle(kept)
        if max_per_class:
            kept = kept[:max_per_class]
        paths.extend(kept)
        labels.extend([label] * len(kept))
    if not paths:
        raise RuntimeError(f"No readable images found under {root}")
    return paths, np.asarray(labels, dtype=np.int64)


def _is_readable_image(path):
    try:
        if os.path.getsize(path) == 0:
            return False
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_image(source, size=IMAGE_SIZE):
    """Decode any image source to a (size, size, 3) uint8 RGB array."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with Image.open(source) as img:
        img = img.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def image_to_features(img, model_input_size=None):
    """Downscale a 224x224x3 RGB array and flatten it to [0, 1] features."""
    model_input_size = model_input_size or MODEL_INPUT_SIZE
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB array, got shape {img.shape}")
    small = Image.fromarray(img).resize(
        (model_input_size, model_input_size), Image.BILINEAR
    )
    return (np.asarray(small, dtype=np.float32) / 255.0).ravel()


def augment_image(img, seed=None):
    """Return one randomly augmented copy of a 224x224x3 RGB array."""
    rng = np.random.default_rng(seed)
    pil = Image.fromarray(img)
    if rng.random() < 0.5:
        pil = ImageOps.mirror(pil)
    angle = float(rng.uniform(-15, 15))
    pil = pil.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    pil = ImageEnhance.Brightness(pil).enhance(float(rng.uniform(0.8, 1.2)))
    pil = ImageEnhance.Contrast(pil).enhance(float(rng.uniform(0.8, 1.2)))
    return np.asarray(pil, dtype=np.uint8)


def featurize_paths(paths, augment_copies=0, seed=42, model_input_size=None):
    """Featurize images, optionally appending augmented copies of each one."""
    features, origin = [], []
    for i, path in enumerate(paths):
        img = load_image(path)
        features.append(image_to_features(img, model_input_size))
        origin.append(i)
        for k in range(augment_copies):
            aug = augment_image(img, seed=(seed + i * 31 + k))
            features.append(image_to_features(aug, model_input_size))
            origin.append(i)
    return np.asarray(features, dtype=np.float32), np.asarray(origin, dtype=np.int64)


def split_paths(paths, labels, val_size=0.1, test_size=0.1, seed=42):
    """Stratified 80/10/10 train/validation/test split over file paths."""
    paths = np.asarray(paths)
    train_paths, hold_paths, train_y, hold_y = train_test_split(
        paths, labels, test_size=(val_size + test_size), random_state=seed,
        stratify=labels,
    )
    rel_test = test_size / (val_size + test_size)
    val_paths, test_paths, val_y, test_y = train_test_split(
        hold_paths, hold_y, test_size=rel_test, random_state=seed, stratify=hold_y,
    )
    return (train_paths, train_y), (val_paths, val_y), (test_paths, test_y)


def preprocess_data(
    paths,
    labels,
    val_size=0.1,
    test_size=0.1,
    seed=42,
    augment_copies=1,
    model_input_size=None,
    persist=True,
):
    """Split, augment (train only), scale, and persist the dataset."""
    (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = split_paths(
        paths, labels, val_size=val_size, test_size=test_size, seed=seed
    )

    X_train, origin = featurize_paths(
        tr_p, augment_copies=augment_copies, seed=seed,
        model_input_size=model_input_size,
    )
    y_train = tr_y[origin]
    X_val, _ = featurize_paths(va_p, model_input_size=model_input_size)
    X_test, _ = featurize_paths(te_p, model_input_size=model_input_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if persist:
        os.makedirs(PROC_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(PROC_DIR, "scaler.pkl"))
        np.savez_compressed(
            os.path.join(PROC_DIR, "cats_dogs_processed.npz"),
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=va_y,
            X_test=X_test, y_test=te_y,
            # Kept so post-deployment monitoring can score the *held-out*
            # split rather than images the model was trained on.
            test_paths=np.asarray(te_p, dtype=np.str_),
        )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": va_y,
        "X_test": X_test, "y_test": te_y,
        "scaler": scaler,
        "test_paths": te_p,
    }


def load_processed_data():
    path = os.path.join(PROC_DIR, "cats_dogs_processed.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Processed data not found. Run scripts/fetch_data.py first."
        )
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def load_scaler(path=None):
    path = path or os.path.join(PROC_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found at {path}")
    return joblib.load(path)

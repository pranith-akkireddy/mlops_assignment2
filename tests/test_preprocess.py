import io
import os

import numpy as np
import pytest
from PIL import Image

from src import preprocess


def _write_image(path, color, size=(300, 200)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


@pytest.fixture
def image_root(tmp_path):
    """A miniature PetImages tree: 50 cats, 50 dogs, plus two corrupt files."""
    root = tmp_path / "PetImages"
    for i in range(50):
        _write_image(str(root / "Cat" / f"cat_{i}.jpg"), (200, 120, 120))
        _write_image(str(root / "Dog" / f"dog_{i}.jpg"), (120, 120, 200))
    # Corrupt entries the real Kaggle corpus also contains.
    (root / "Cat" / "corrupt.jpg").write_bytes(b"")
    (root / "Dog" / "corrupt.jpg").write_bytes(b"not a jpeg at all")
    return root


def test_load_image_produces_224_rgb():
    buf = io.BytesIO()
    Image.new("RGB", (57, 311), (10, 20, 30)).save(buf, format="JPEG")
    img = preprocess.load_image(buf.getvalue())
    assert img.shape == (preprocess.IMAGE_SIZE, preprocess.IMAGE_SIZE, 3)
    assert img.dtype == np.uint8


def test_load_image_converts_grayscale_to_three_channels(tmp_path):
    path = tmp_path / "gray.jpg"
    Image.new("L", (120, 120), 128).save(path, format="JPEG")
    img = preprocess.load_image(str(path))
    assert img.shape[2] == 3


def test_discover_images_skips_corrupt_files(image_root):
    paths, labels = preprocess.discover_images(root=str(image_root))
    assert len(paths) == 100
    assert set(np.unique(labels)) == {0, 1}
    assert int((labels == 0).sum()) == 50
    assert int((labels == 1).sum()) == 50
    assert not any("corrupt" in p for p in paths)


def test_discover_images_respects_max_per_class(image_root):
    paths, labels = preprocess.discover_images(root=str(image_root), max_per_class=10)
    assert len(paths) == 20
    assert int((labels == 1).sum()) == 10


def test_image_to_features_shape_and_range():
    img = np.full((preprocess.IMAGE_SIZE, preprocess.IMAGE_SIZE, 3), 255, dtype=np.uint8)
    feats = preprocess.image_to_features(img, model_input_size=16)
    assert feats.shape == (16 * 16 * 3,)
    assert feats.min() >= 0.0 and feats.max() <= 1.0


def test_image_to_features_rejects_non_rgb():
    with pytest.raises(ValueError):
        preprocess.image_to_features(np.zeros((10, 10), dtype=np.uint8))


def test_augment_image_is_deterministic_and_changes_pixels():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (preprocess.IMAGE_SIZE, preprocess.IMAGE_SIZE, 3),
                       dtype=np.uint8)
    a = preprocess.augment_image(img, seed=123)
    b = preprocess.augment_image(img, seed=123)
    assert a.shape == img.shape
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, img)


def test_split_paths_is_80_10_10_and_disjoint(image_root):
    paths, labels = preprocess.discover_images(root=str(image_root))
    (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = preprocess.split_paths(
        paths, labels, seed=42
    )
    assert (len(tr_p), len(va_p), len(te_p)) == (80, 10, 10)
    assert not (set(tr_p) & set(va_p))
    assert not (set(tr_p) & set(te_p))
    assert not (set(va_p) & set(te_p))
    # Stratification keeps both classes present in every split.
    for y in (tr_y, va_y, te_y):
        assert set(np.unique(y)) == {0, 1}


def test_preprocess_data_augments_train_and_persists(image_root, tmp_path, monkeypatch):
    proc = tmp_path / "processed"
    monkeypatch.setattr(preprocess, "PROC_DIR", str(proc))
    paths, labels = preprocess.discover_images(root=str(image_root))

    data = preprocess.preprocess_data(
        paths, labels, seed=42, augment_copies=1, model_input_size=8, persist=True
    )

    # 80 originals + 80 augmented copies.
    assert data["X_train"].shape[0] == 160
    assert data["y_train"].shape[0] == 160
    assert data["X_val"].shape[0] == 10
    assert data["X_test"].shape[0] == 10
    assert data["X_train"].shape[1] == 8 * 8 * 3
    assert hasattr(data["scaler"], "scale_")
    assert os.path.exists(proc / "scaler.pkl")
    assert os.path.exists(proc / "cats_dogs_processed.npz")


def test_load_scaler_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess.load_scaler(str(tmp_path / "nope.pkl"))

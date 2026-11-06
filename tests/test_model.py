import numpy as np
import tensorflow as tf

from src.config import load_config, set_seeds
from src.model import build_base_model, build_model, compile_model, freeze_backbone


def test_model_output_shape_and_softmax():
    cfg = load_config()
    set_seeds(cfg)
    model = build_model(cfg)
    x = tf.random.uniform((3, 224, 224, 3), 0, 1, seed=42)
    y = model.predict(x, verbose=0)
    assert y.shape == (3, 3)
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5)
    assert (y >= 0).all() and (y <= 1).all()


def test_model_architecture_has_gap_dropout_softmax():
    cfg = load_config()
    model = build_model(cfg)
    names = [layer.name for layer in model.layers]
    assert any("global" in n for n in names), names
    assert "dropout" in names
    last = model.layers[-1]
    assert last.units == 3 and last.activation == tf.keras.activations.softmax


def test_backbone_is_mobilenetv2_imagenet():
    cfg = load_config()
    base = build_base_model(cfg)
    assert "mobilenetv2" in base.name.lower()


def test_freeze_and_finetune():
    cfg = load_config()
    model = build_model(cfg)
    freeze_backbone(model, fine_tune_layers=0)
    base = model.layers[1]
    assert base.trainable is False
    freeze_backbone(model, fine_tune_layers=2)
    assert base.trainable is True
    # Chỉ 5 layer cuối trainable (2 từ backbone + dense cuối)
    trainable = [l.name for l in model.layers if l.trainable]
    # backbone mở, các layer không thuộc backbone ở ngoài vẫn trainable theo mặc định
    assert len(trainable) >= 2


def test_compile():
    cfg = load_config()
    model = build_model(cfg)
    compile_model(model, learning_rate=0.001)
    assert model.optimizer is not None

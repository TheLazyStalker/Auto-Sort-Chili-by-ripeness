"""MobileNetV2 / MobileNetV3Small (ImageNet pretrained) + Transfer Learning.

Kiến trúc: Input 224x224x3 → backbone → GAP → Dropout → Dense(3, softmax).
Huấn luyện 2 giai đoạn: (1) freeze backbone, train head; (2) fine-tune N layer cuối.

Backbone chọn qua ``cfg["model"]["base"]``: ``mobilenetv2`` (mặc định, giữ nguyên
hành vi cũ) hoặc ``mobilenetv3small``.

⚠️ MobileNetV3Small preprocessing (Keras 3 / TF 2.17):
- ``preprocess_input()`` của V3 là **placeholder no-op** — KHÔNG được gọi.
- Model  ``include_preprocessing=True`` (mặc định) nhúng layer ``Rescaling``
  (scale=1/128, offset=-1) ngay sau Input → biến uint8/float [0,255] → [-1,1].
- Vì vậy: không gọi preprocess_input, không thêm Rescaling(1/255) ngoài model, và
  mọi consumer (dataset/predict/evaluate/export/Android) vẫn truyền RGB [0,255].
"""
from __future__ import annotations

import tensorflow as tf

BASE_ALIASES = {
    "mobilenetv2": ("mobilenetv2", "mobilenet_v2", "v2", "mobilenetv2small"),
    "mobilenetv3small": ("mobilenetv3small", "mobilenet_v3_small", "v3", "v3small", "mobilenetv3"),
}


def normalize_base_name(name) -> str:
    key = str(name or "").strip().lower()
    for canonical, aliases in BASE_ALIASES.items():
        if key in aliases:
            return canonical
    raise ValueError(
        f"Model base không hợp lệ: '{name}'. Chọn: {sorted(BASE_ALIASES)}")


def build_base_model(cfg: dict, trainable: bool = False):
    """Backbone (V2 hoặc V3Small) ImageNet, include_top=False.

    - V2: preprocess_input ngoài model (như cũ).
    - V3Small: include_preprocessing=True → preprocessing nhúng trong model.
    """
    image_size = int(cfg["data"]["image_size"])
    weights = cfg["model"].get("weights", "imagenet")
    base_name = normalize_base_name(cfg["model"].get("base", "mobilenetv2"))

    if base_name == "mobilenetv3small":
        base = tf.keras.applications.MobileNetV3Small(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights=weights,
            include_preprocessing=True,
        )
    else:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights=weights,
        )
    base.trainable = trainable
    base._base_name = base_name  # type: ignore[attr-defined]
    return base


def build_model(cfg: dict, trainable_backbone: bool = False):
    """Dựng model đầy đủ: backbone + GAP + Dropout + Dense(3, softmax)."""
    classes = cfg["project"]["classes"]
    num_classes = int(cfg["project"]["num_classes"])
    image_size = int(cfg["data"]["image_size"])
    dropout = float(cfg["model"].get("dropout", 0.3))
    base_name = normalize_base_name(cfg["model"].get("base", "mobilenetv2"))

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="input_image")
    if base_name == "mobilenetv3small":
        # V3: preprocessing nằm trong base (include_preprocessing=True). KHÔNG gọi
        # preprocess_input (no-op) — tránh nhầm lẫn khái niệm "bị áp 2 lần".
        x = inputs
    else:
        # V2: áp dụng preprocess_input ở đây ([0,255] → [-1,1]); dataset/predict/
        # evaluate/export/Android KHÔNG được Rescaling(1/255) trước (tránh double-scaling).
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    base = build_base_model(cfg, trainable=trainable_backbone)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="softmax_output")(x)
    model = tf.keras.Model(inputs, outputs, name=f"chili_ripeness_{base_name}")
    model.classes_ = list(classes)  # gắn danh sách lớp để predict dùng
    model.base_name = base_name
    return model


def _find_backbone(model: tf.keras.Model):
    """Tìm layer backbone bên trong model (V2 hoặc V3)."""
    for layer in model.layers:
        name = layer.name.lower()
        if "mobilenetv2" in name or "mobilenetv3" in name:
            return layer
    # fallback: layer thứ 2 (inputs, backbone, ...)
    if len(model.layers) > 1:
        return model.layers[1]
    return None


def freeze_backbone(model: tf.keras.Model, fine_tune_layers: int = 0) -> None:
    """Đóng băng backbone; nếu fine_tune_layers>0 thì mở N layer cuối để fine-tune.

    Lưu ý BatchNormalization (V3 có nhiều BN): trong fine-tune, các layer BN giữ
    trainable=True (chạy trong training mode) — đúng chuẩn fine-tune V3; không
    ép BN frozen vì sẽ phá hành vi BN khi fine-tune.
    """
    base = _find_backbone(model)
    if base is None:
        raise RuntimeError("Không tìm thấy backbone trong model.")
    if fine_tune_layers <= 0:
        base.trainable = False
    else:
        base.trainable = True
        n = len(base.layers)
        for i, layer in enumerate(base.layers):
            layer.trainable = i >= n - fine_tune_layers


def compile_model(model: tf.keras.Model, learning_rate: float = 0.001, loss: str = "categorical_crossentropy"):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def get_callbacks(cfg: dict, prefix: str = "") -> list:
    """EarlyStopping + ModelCheckpoint + ReduceLROnPlateau.

    ``prefix`` (VD "phase_a"/"phase_b") được thêm vào tên checkpoint để lưu riêng.
    """
    paths = cfg["paths"]
    t = cfg.get("training", {})
    ckpt = f"{paths['checkpoints']}/best_{prefix}chili.keras" if prefix else \
        f"{paths['checkpoints']}/best_chili.keras"
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=int(t.get("early_stopping_patience", 5)),
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt, monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=float(t.get("reduce_lr_factor", 0.5)),
            patience=int(t.get("reduce_lr_patience", 3)),
            min_lr=float(t.get("reduce_lr_min_lr", 1e-6)), verbose=1,
        ),
    ]

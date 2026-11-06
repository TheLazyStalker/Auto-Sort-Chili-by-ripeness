"""MobileNetV2 (ImageNet pretrained) + Transfer Learning cho phân loại độ chín ớt.

Kiến trúc: Input 224x224x3 → MobileNetV2 → GAP → Dropout → Dense(3, softmax).
Huấn luyện 2 giai đoạn: (1) freeze backbone, train head; (2) fine-tune N layer cuối.
"""
from __future__ import annotations

import tensorflow as tf


def build_base_model(cfg: dict, trainable: bool = False):
    """Backbone MobileNetV2 ImageNet, include_top=False."""
    image_size = int(cfg["data"]["image_size"])
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=cfg["model"].get("weights", "imagenet"),
    )
    base.trainable = trainable
    return base


def build_model(cfg: dict, trainable_backbone: bool = False):
    """Dựng model đầy đủ: backbone + GAP + Dropout + Dense(3, softmax)."""
    classes = cfg["project"]["classes"]
    num_classes = int(cfg["project"]["num_classes"])
    image_size = int(cfg["data"]["image_size"])
    dropout = float(cfg["model"].get("dropout", 0.3))

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="input_image")
    # Preprocessing MobileNetV2 chuẩn ImageNet: preprocess_input maps [0,255] -> [-1,1].
    # Model nhận float32 [0,255] và tự tiền xử lý bên trong; dataset/predict/evaluate/
    # export/Android KHÔNG được Rescaling(1/255) trước khi đưa vào model (tránh double-scaling).
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    base = build_base_model(cfg, trainable=trainable_backbone)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="softmax_output")(x)
    model = tf.keras.Model(inputs, outputs, name="chili_ripeness_mobilenetv2")
    model.classes_ = list(classes)  # gắn danh sách lớp để predict dùng
    return model


def freeze_backbone(model: tf.keras.Model, fine_tune_layers: int = 0) -> None:
    """Đóng băng backbone; nếu fine_tune_layers>0 thì mở N layer cuối để fine-tune."""
    base = next((layer for layer in model.layers if "mobilenetv2" in layer.name.lower()), None)
    # Build xong model mới có sub-layer; truy cập qua model.layers[1] nếu cần.
    if base is None and len(model.layers) > 1:
        base = model.layers[1]
    if base is None:
        raise RuntimeError("Không tìm thấy backbone MobileNetV2 trong model.")
    if fine_tune_layers <= 0:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers[-fine_tune_layers:]:
            layer.trainable = True


def compile_model(model: tf.keras.Model, learning_rate: float = 0.001, loss: str = "categorical_crossentropy"):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def get_callbacks(cfg: dict) -> list:
    """EarlyStopping + ModelCheckpoint + ReduceLROnPlateau."""
    paths = cfg["paths"]
    t = cfg.get("training", {})
    ckpt = f"{paths['checkpoints']}/best_chili.keras"
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

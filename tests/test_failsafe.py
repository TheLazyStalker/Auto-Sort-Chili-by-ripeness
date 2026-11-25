"""Tests cho fail-safe evaluate/predict, smoke weights path, và doc preprocessing."""
import os
import subprocess
import sys

import pytest

from src.config import PROJECT_ROOT, load_config


def test_evaluate_missing_model_exits_2():
    """evaluate.py khi thiếu model → exit code 2, thông báo rõ, không traceback."""
    cfg = load_config()
    proc = subprocess.run(
        [sys.executable, "-m", "src.evaluate", "--split", "test"],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "LỖI" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_predict_missing_model_exits_2():
    """predict.py khi thiếu model → exit code 2, thông báo rõ, không traceback."""
    proc = subprocess.run(
        [sys.executable, "-m", "src.predict", "--image", "dataset/raw/ripe/x.jpg"],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "LỖI" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_smoke_checks_correct_weights_filename():
    """smoke_test_model.py phải trỏ đúng file weights no_top (không phải tensorflow chính)."""
    import inspect
    from scripts import smoke_test_model as smoke
    src = inspect.getsource(smoke)
    # Phải dùng file no_top (đúng file ImageNet include_top=False trong keras cache)
    assert "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5" in src
    # Không được trỏ vào file ..._tf_kernels.h5 (không tồn tại trong cache)
    assert "mobilenet_v2_weights_tf_dim_ordering_tf_kernels.h5" not in src


def test_preprocessing_invariant_documented_and_configured():
    """Chuẩn tiền xử lý: RGB [0,255] → model tự Rescaling(1/128,-1); KHÔNG chia 255."""
    import json
    doc_path = os.path.join(PROJECT_ROOT, "deployment/task6/README.md")
    with open(doc_path, encoding="utf-8") as f:
        doc = f.read()
    assert "[0,255]" in doc
    assert "Rescaling" in doc
    assert "chia 255" in doc  # có nhắc việc KHÔNG được chia 255
    # pipeline_config: V3 giữ preprocessing trong model
    with open(os.path.join(PROJECT_ROOT, "pipeline_config.json"), encoding="utf-8") as f:
        pcfg = json.load(f)
    assert pcfg["model"]["include_preprocessing"] is True
    assert pcfg["model"]["base"] == "mobilenetv3small"
    # Không có Rescaling(1/255) ĐƯỢC ÁP DỤNG trong model (chỉ nhắc trong docstring)
    import inspect
    from src import model as model_mod
    src = inspect.getsource(model_mod)
    assert "layers.Rescaling(" not in src, "model.py không được tự thêm lớp Rescaling"
    assert "mobilenet_v2.preprocess_input" in src or "include_preprocessing" in src


def test_evaluate_load_model_checks_existence_first():
    """evaluate.py phải kiểm tra model tồn tại trước khi load_model."""
    import inspect
    from src import evaluate
    src = inspect.getsource(evaluate.evaluate_model)
    # Phải có kiểm tra os.path.exists + raise FileNotFoundError trước load_model
    assert "os.path.exists(model_path)" in src
    assert "FileNotFoundError" in src
    # load_model nằm SAU khối kiểm tra (theo thứ tự source)
    assert src.index("FileNotFoundError") < src.index("load_model")

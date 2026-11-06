import os

from src.config import get_class_mapping, get_classes, get_split_ratios, load_config, resolve_seed


def test_config_loads_and_has_three_classes():
    cfg = load_config()
    assert get_classes(cfg) == ["unripe", "medium_ripe", "ripe"]
    assert cfg["project"]["num_classes"] == 3


def test_class_mapping_is_fixed_order():
    cfg = load_config()
    mapping = get_class_mapping(cfg)
    assert mapping == {"unripe": 0, "medium_ripe": 1, "ripe": 2}


def test_split_ratios_sum_to_one():
    cfg = load_config()
    assert get_split_ratios(cfg) == (0.70, 0.15, 0.15)


def test_seed_and_paths_resolved():
    cfg = load_config()
    assert resolve_seed(cfg).seed == 42
    assert os.path.isabs(cfg["paths"]["dataset_raw"])


def test_image_size_and_dropout_from_yaml():
    cfg = load_config()
    assert cfg["data"]["image_size"] == 224
    assert cfg["data"]["image_channels"] == 3
    assert cfg["model"]["dropout"] == 0.3

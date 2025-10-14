import pytest


def test_basic_imports():
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    import sklearn  # noqa: F401

    # lightgbm требует libomp на macOS; если не установлено — пропустим тест
    pytest.importorskip("lightgbm")

    # внешние из GitHub — просто импорт
    import pymssa  # noqa: F401
    import skccm  # noqa: F401

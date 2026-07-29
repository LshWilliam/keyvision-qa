import pytest

pytest.importorskip("gradio")

from keyvision.demo import build_demo


def test_demo_builds_without_launching() -> None:
    demo = build_demo("configs/smoke.yaml")
    assert demo is not None

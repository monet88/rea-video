"""Pytest isolation so personal ~/.rea-video prefs and env_file never leak into unit tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_rea_video_home(tmp_path, monkeypatch):
    monkeypatch.setenv("REA_VIDEO_HOME", str(tmp_path / ".rea-video"))

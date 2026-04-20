from __future__ import annotations

import os

from deep_quality.config import PROJECT_ROOT


def build_project_env() -> dict[str, str]:
    env = dict(os.environ)
    src_path = str(PROJECT_ROOT / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current else f"{src_path}:{current}"
    return env

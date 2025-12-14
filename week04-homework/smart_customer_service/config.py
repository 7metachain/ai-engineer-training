from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppConfig:
    """Runtime config that can be hot-reloaded."""

    model_version: str
    plugin_modules: list[str]


DEFAULT_CONFIG = AppConfig(
    model_version="rules-v1",
    plugin_modules=["smart_customer_service.plugins.invoice_plugin"],
)


def _config_path() -> Path:
    # Keep config file inside the package so it's easy for homework submission.
    return Path(__file__).resolve().parent / "runtime_config.json"


def ensure_default_config_file() -> None:
    path = _config_path()
    if path.exists():
        return
    path.write_text(
        json.dumps(
            {
                "model_version": DEFAULT_CONFIG.model_version,
                "plugin_modules": DEFAULT_CONFIG.plugin_modules,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_config() -> AppConfig:
    ensure_default_config_file()
    raw: dict[str, Any] = json.loads(_config_path().read_text(encoding="utf-8"))
    model_version = str(raw.get("model_version", DEFAULT_CONFIG.model_version))
    plugin_modules = raw.get("plugin_modules", DEFAULT_CONFIG.plugin_modules)
    if not isinstance(plugin_modules, list) or not all(
        isinstance(x, str) for x in plugin_modules
    ):
        plugin_modules = DEFAULT_CONFIG.plugin_modules
    return AppConfig(model_version=model_version, plugin_modules=list(plugin_modules))



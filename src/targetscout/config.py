"""Central config: loads config.yaml + .env."""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"


@lru_cache
def settings() -> dict:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)
    cfg["env"] = {
        "ncbi_email": os.getenv("NCBI_EMAIL", ""),
        "ncbi_api_key": os.getenv("NCBI_API_KEY", ""),
        "database_url": os.getenv("DATABASE_URL", ""),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    }
    return cfg

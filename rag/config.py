import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
INDEX = ROOT / 'index'


def setting(key, default=None):
    value = os.getenv(key, default)
    if value is None or not value:
        raise RuntimeError(f'Set {key} in .env (see .env.example)')
    return value

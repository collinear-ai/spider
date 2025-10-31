import os, json
from fastapi import Header, HTTPException

def load_api_keys(API_KEYS_FILE: str, ALLOWED_API_KEYS: dict[str, str]) -> dict[str, str]:
    """Load API keys from disk. If file doesn't exist, write the current
    ALLOWED_API_KEYS to disk and return that list."""
    try:
        if os.path.exists(API_KEYS_FILE):
            with open(API_KEYS_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    # fallback: persist the bundled keys
    try:
        with open(API_KEYS_FILE, "w") as f:
            json.dump(ALLOWED_API_KEYS, f, indent=2)
    except Exception:
        pass
    return ALLOWED_API_KEYS.copy()

def save_api_keys(keys: dict[str, str], API_KEYS_FILE: str) -> None:
    try:
        with open(API_KEYS_FILE, "w") as f:
            json.dump(keys, f, indent=2)
    except Exception as e:
        # best-effort; don't raise to avoid breaking runtime
        print(f"Failed to save API keys: {e}")
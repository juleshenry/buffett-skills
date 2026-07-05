import hashlib
import os
import pickle
from pathlib import Path
from functools import wraps
from datetime import datetime

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_cache_key(func, *args, **kwargs):
    # Try to make a stable string representation
    key_str = f"{func.__name__}:{repr(args)}:{repr(sorted(kwargs.items()))}"
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()

def disk_cache(days=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = _get_cache_key(func, *args, **kwargs)
            cache_file = CACHE_DIR / f"{func.__name__}_{key}.pkl"
            
            # Check if cache exists and is valid
            if cache_file.exists():
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if (datetime.now() - mtime).days < days:
                    try:
                        with open(cache_file, "rb") as f:
                            return pickle.load(f)
                    except Exception:
                        pass # File corrupted or unpicklable, ignore and re-run
            
            # Run the function
            result = func(*args, **kwargs)
            
            # Cache the result
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            except Exception:
                if cache_file.exists():
                    cache_file.unlink() # Cleanup if dump failed
                
            return result
        return wrapper
    return decorator

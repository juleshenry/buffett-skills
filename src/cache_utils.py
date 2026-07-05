import hashlib
import inspect
import os
import pickle
from pathlib import Path
from functools import wraps
from datetime import datetime

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_source_fingerprint(func) -> str:
    """
    Hash of the function's own source code. Any edit to the function body
    changes this fingerprint, which changes the cache key below, which makes
    old cached results for that function unreachable (they just become
    orphaned files instead of being served up as stale/incorrect data).

    Without this, a `days=N` TTL has no idea the code changed and will
    happily keep serving results computed under old, possibly-buggy logic
    for up to N days after a fix lands.
    """
    try:
        return hashlib.md5(inspect.getsource(func).encode("utf-8")).hexdigest()[:12]
    except (OSError, TypeError):
        # Can't introspect source (e.g. built-in, C extension, REPL). Fall back
        # to a constant so caching still works, just without auto-invalidation.
        return "nosource"

def _get_cache_key(func, source_fingerprint, *args, **kwargs):
    # Try to make a stable string representation
    key_str = f"{func.__name__}:{source_fingerprint}:{repr(args)}:{repr(sorted(kwargs.items()))}"
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()

def disk_cache(days=1, cache_dir=None):
    target_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func):
        source_fingerprint = _get_source_fingerprint(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = _get_cache_key(func, source_fingerprint, *args, **kwargs)
            cache_file = target_dir / f"{func.__name__}_{key}.pkl"
            
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

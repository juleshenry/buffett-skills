import os


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_GENERATE_URL = f"{DEFAULT_OLLAMA_HOST}/api/generate"
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

DEFAULT_SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "BuffettSkillsBot buffettskills@example.com",
)

DEFAULT_BENCHMARK = os.environ.get("DEFAULT_BENCHMARK", "^GSPC")
DEFAULT_LOOKBACK_YEARS = _get_int_env("DEFAULT_LOOKBACK_YEARS", 10)
DEFAULT_INTRINSIC_VALUE_YEARS = _get_int_env("DEFAULT_INTRINSIC_VALUE_YEARS", 10)
DEFAULT_MR_MARKET_PERIOD = os.environ.get("DEFAULT_MR_MARKET_PERIOD", "1y")

DEFAULT_CIRCLE_OF_COMPETENCE_TEMPERATURE = _get_float_env(
    "DEFAULT_CIRCLE_OF_COMPETENCE_TEMPERATURE",
    0.2,
)
DEFAULT_INVERSION_TEMPERATURE = _get_float_env("DEFAULT_INVERSION_TEMPERATURE", 0.3)
RISK_FREE_RATE_FALLBACK = _get_float_env("RISK_FREE_RATE_FALLBACK", 0.042)

import os
import json
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests


def _load_repo_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_repo_env()


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


def _resolve_ollama_model(model_name: str) -> str:
    normalized = (model_name or "").strip().lower()
    aliases = {
        "ollama3": "llama3",
        "llama3": "llama3",
        "llama3.1": "llama3",
        "gemma": "gemma4:26b",
        "gemma2": "gemma4:26b",
        "gemma4": "gemma4:26b",
    }
    return aliases.get(normalized, model_name)


def get_ollama_panel_models() -> list[str]:
    raw_models = os.environ.get("OLLAMA_PANEL_MODELS", "")
    if raw_models.strip():
        return _dedupe_preserve_order(
            [_resolve_ollama_model(model.strip()) for model in raw_models.split(",") if model.strip()]
        )

    return _default_ollama_panel()


def get_ollama_panel_sleep_seconds() -> float:
    value = os.environ.get("OLLAMA_PANEL_SLEEP_SECONDS")
    if value is None:
        return 1.0
    try:
        return max(0.0, float(value))
    except ValueError:
        return 1.0


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _default_ollama_panel(model_override: str | None = None) -> list[str]:
    panel = ["qwen2.5:7b", "llama3"]
    resolved_override = _resolve_ollama_model(model_override) if model_override else None
    if resolved_override and resolved_override not in panel:
        panel.append(resolved_override)
    return _dedupe_preserve_order(panel)


def get_ollama_models_for_call(model_override: str | None = None) -> list[str]:
    raw_models = os.environ.get("OLLAMA_PANEL_MODELS", "")
    if raw_models.strip():
        return get_ollama_panel_models()
    return _default_ollama_panel(model_override=model_override)


def _strip_json_fences(response_text: str) -> str:
    cleaned_text = response_text.strip()
    cleaned_text = cleaned_text.removeprefix("```json").removeprefix("```").strip()
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3].strip()
    return cleaned_text


def _parse_json_response(response_text: str) -> dict[str, Any]:
    return json.loads(_strip_json_fences(response_text or "{}"))


def _consensus_text(texts: list[str]) -> str:
    non_empty = [text for text in texts if text]
    if not non_empty:
        return ""
    if len(non_empty) == 1:
        return non_empty[0]

    counts = Counter(non_empty)
    most_common_text, count = counts.most_common(1)[0]
    if count > 1:
        return most_common_text

    best_text = non_empty[0]
    best_score = float("-inf")
    for candidate in non_empty:
        score = sum(SequenceMatcher(None, candidate, other).ratio() for other in non_empty if other is not candidate)
        if score > best_score:
            best_score = score
            best_text = candidate
    return best_text


def _aggregate_json_values(values: list[Any]) -> Any:
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None

    if all(isinstance(value, dict) for value in present_values):
        keys = set().union(*(value.keys() for value in present_values))
        aggregated = {}
        for key in keys:
            aggregated_value = _aggregate_json_values([value.get(key) for value in present_values])
            if aggregated_value is not None:
                aggregated[key] = aggregated_value
        return aggregated

    if all(isinstance(value, bool) for value in present_values):
        true_votes = sum(1 for value in present_values if value)
        return true_votes >= ((len(present_values) // 2) + 1)

    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in present_values):
        average = sum(float(value) for value in present_values) / len(present_values)
        if all(isinstance(value, int) and not isinstance(value, bool) for value in present_values):
            return int(round(average))
        return average

    if all(isinstance(value, str) for value in present_values):
        return _consensus_text(present_values)

    if all(isinstance(value, list) for value in present_values):
        serialized = [json.dumps(value, sort_keys=True) for value in present_values]
        return json.loads(_consensus_text(serialized))

    return present_values[0]


def aggregate_panel_json(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregated = _aggregate_json_values(results)
    if not isinstance(aggregated, dict):
        return {}

    aggregated["panel_judgments"] = {
        result.get("_panel_model", f"judge_{index + 1}"): {
            key: value for key, value in result.items() if key != "_panel_model"
        }
        for index, result in enumerate(results)
        if result
    }
    aggregated["panel_models"] = [
        result.get("_panel_model") for result in results if result and result.get("_panel_model")
    ]
    return aggregated


def aggregate_panel_judgment(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid_results = [result for result in results if result]
    if not valid_results:
        return {}

    bool_values = [bool(result.get("inside_circle", False)) for result in valid_results]
    confidences = [int(result.get("confidence", 0) or 0) for result in valid_results]
    inside_votes = sum(1 for value in bool_values if value)
    inside_circle = inside_votes >= ((len(bool_values) // 2) + 1)
    confidence = round(sum(confidences) / len(confidences)) if confidences else 0

    supporting_explanations = [
        result.get("explanation", "") for result in valid_results if bool(result.get("inside_circle", False)) == inside_circle and result.get("explanation")
    ]
    explanation = supporting_explanations[0] if supporting_explanations else valid_results[0].get("explanation", "")

    return {
        "inside_circle": inside_circle,
        "confidence": max(0, min(int(confidence), 100)),
        "explanation": explanation,
        "panel_models": [result.get("_panel_model") for result in valid_results if result.get("_panel_model")],
        "panel_judgments": {
            result.get("_panel_model", f"judge_{index + 1}"): {
                key: value for key, value in result.items() if key != "_panel_model"
            }
            for index, result in enumerate(valid_results)
        },
        "panel_vote_split": {
            "inside_circle": inside_votes,
            "outside_circle": len(bool_values) - inside_votes,
        },
    }


def _call_single_ollama_model(
    prompt: str,
    model: str,
    json_format: bool,
    options: dict[str, Any] | None = None,
    timeout: int = 60,
    host: str | None = None,
) -> dict[str, Any]:
    import logging
    logger = logging.getLogger(__name__)
    
    base_host = (host or DEFAULT_OLLAMA_HOST).rstrip("/")
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if json_format:
        payload["format"] = "json"
    if options:
        payload["options"] = options

    logger.info(f"  -> [Ollama API] Sending request to {base_host}/api/generate with model '{model}'")
    try:
        response = requests.post(f"{base_host}/api/generate", json=payload, timeout=timeout)
        response.raise_for_status()
        response_json = response.json()
        logger.info(f"  -> [Ollama API] Received successful response from '{model}'")
        return {
            "model": model,
            "response": response_json.get("response", ""),
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"  -> [Ollama API] Call failed for model '{model}': {e}")
        raise


def call_ollama_panel_json(
    prompt: str,
    model: str | None = None,
    options: dict[str, Any] | None = None,
    timeout: int = 60,
    host: str | None = None,
    aggregator=None,
) -> dict[str, Any]:
    models = get_ollama_models_for_call(model_override=model)
    sleep_seconds = get_ollama_panel_sleep_seconds()
    parsed_results: list[dict[str, Any]] = []

    for index, panel_model in enumerate(models):
        try:
            raw_result = _call_single_ollama_model(
                prompt=prompt,
                model=panel_model,
                json_format=True,
                options=options,
                timeout=timeout,
                host=host,
            )
            parsed = _parse_json_response(raw_result["response"])
            if isinstance(parsed, dict):
                parsed["_panel_model"] = panel_model
                parsed_results.append(parsed)
        except Exception:
            pass

        if index < len(models) - 1 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if aggregator is not None:
        return aggregator(parsed_results)
    return aggregate_panel_json(parsed_results)


def call_ollama_panel_text(
    prompt: str,
    model: str | None = None,
    options: dict[str, Any] | None = None,
    timeout: int = 60,
    host: str | None = None,
) -> dict[str, Any]:
    models = get_ollama_models_for_call(model_override=model)
    sleep_seconds = get_ollama_panel_sleep_seconds()
    responses: list[str] = []

    for index, panel_model in enumerate(models):
        try:
            raw_result = _call_single_ollama_model(
                prompt=prompt,
                model=panel_model,
                json_format=False,
                options=options,
                timeout=timeout,
                host=host,
            )
            responses.append(raw_result["response"])
        except Exception:
            pass

        if index < len(models) - 1 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return {"response": _consensus_text(responses)}


DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_GENERATE_URL = f"{DEFAULT_OLLAMA_HOST}/api/generate"
DEFAULT_OLLAMA_MODEL = _resolve_ollama_model(os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"))

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

EARNINGSCALLS_API_BASE_URL = os.environ.get("EARNINGSCALLS_API_BASE_URL", "https://earningscalls.dev/api/v1").rstrip("/")
EARNINGSCALLS_API_KEY = os.environ.get("EARNINGSCALLS_API_KEY") or os.environ.get("EARNINGS_CALL_API_KEY", "")

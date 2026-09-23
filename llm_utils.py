import time
import httpx
from google.genai import errors

DEFAULT_MODELS = ("gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash-lite")

# Errors worth retrying: Gemini's own "overloaded" response, plus raw
# network-level hiccups (dropped connections, timeouts) that are usually
# transient and unrelated to anything wrong in our code or account.
RETRYABLE_ERRORS = (
    errors.ServerError,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ConnectTimeout,
)


def call_with_retry(call_fn, models=DEFAULT_MODELS, max_attempts: int = 3, base_delay: float = 3.0, max_delay: float = 20.0):
    """
    Call call_fn(model_name) and retry with exponential backoff on transient
    errors: Gemini reporting itself overloaded (503), or raw network-level
    connection drops/timeouts. If a model keeps failing after max_attempts,
    automatically fall back to the next model in `models` before giving up.
    """
    last_exception = None
    for model_index, model_name in enumerate(models):
        is_last_model = model_index == len(models) - 1
        for attempt in range(1, max_attempts + 1):
            try:
                return call_fn(model_name)
            except RETRYABLE_ERRORS as e:
                last_exception = e
                if attempt == max_attempts:
                    if not is_last_model:
                        print(f"    ({model_name} still unavailable after {max_attempts} attempts — falling back to {models[model_index + 1]})")
                    break
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                print(f"    ({model_name} temporarily unavailable, retrying in {delay:.0f}s — attempt {attempt}/{max_attempts})")
                time.sleep(delay)
    raise last_exception
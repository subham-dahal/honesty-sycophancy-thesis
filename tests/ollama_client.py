"""Minimal Ollama client with truncation protection (standard library only).

Uses Ollama's native /api/chat endpoint rather than the OpenAI-compatible one,
because only the native endpoint accepts `num_ctx` (the context window) and
reports why generation stopped.

If a reply stops because it hit the token limit (done_reason == "length"),
the same request is re-sent with double the limit, up to `ceiling`. Decoding
is greedy (temperature 0), so the longer reply starts with exactly the same
text; it just isn't cut off. A reply that still hits the ceiling is returned
with truncated=True so it can be reported rather than silently graded.

Greedy decoding sometimes falls into a loop, repeating the same working until
it runs out of tokens. More tokens never fixes that, so a reply that hit the
limit while repeating itself is returned straight away with looping=True.
"""
import json
import os
import re
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST

# 16k tokens comfortably holds a 3-turn conversation with 4k-token replies,
# so Ollama never has to drop the start of the prompt (its default is smaller).
NUM_CTX = 16384
DEFAULT_CEILING = 2048  # a multiple-choice reply longer than this is degenerate


def _chat(model, messages, num_predict, num_ctx, temperature=0.0):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())


def is_looping(text, window=150):
    """True if the last `window` characters already appeared earlier in the text.
    Numbers are normalised first, so a reply that repeats the same working with
    different numbers ("try 69,600 sq ft ... try 70,800 sq ft ...") also counts."""
    text = re.sub(r"\s+", " ", re.sub(r"\d+", "0", text)).strip()
    if len(text) < 3 * window:
        return False
    tail = text[-window:]
    return tail in text[:-window]


def generate(model, messages, max_tokens, ceiling=DEFAULT_CEILING, escalate=True, num_ctx=NUM_CTX,
             temperature=0.0):
    """Returns a dict: text, finish_reason, truncated, looping, max_tokens_used,
    completion_tokens, prompt_tokens.
    temperature > 0 samples instead of decoding greedily (Sharma et al. sample
    feedback comments at T=1); a regenerated reply is then a fresh sample."""
    limit = max_tokens
    while True:
        data = _chat(model, messages, limit, num_ctx, temperature)
        reason = data.get("done_reason")
        text = data["message"]["content"]
        looping = reason == "length" and is_looping(text)
        if escalate and reason == "length" and limit < ceiling and not looping:
            limit = min(limit * 2, ceiling)
            continue
        return {
            "text": text.strip(),
            "looping": looping,
            "finish_reason": reason,
            "truncated": reason == "length",
            "max_tokens_used": limit,
            "completion_tokens": data.get("eval_count"),
            "prompt_tokens": data.get("prompt_eval_count"),
        }

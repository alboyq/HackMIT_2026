"""Self-hosted-first vision + TTS, with cloud APIs kept as fallback.
Keys/config from env / .env — never the browser.

Tries, in order:
  LOCAL_AI (default on) self-hosted models on the remote GB10 box, reached
                         through an SSH port-forward:
                           vision: Ollama  (OLLAMA_URL,   default http://localhost:11434)
                           tts:    Kokoro  (KOKORO_TTS_URL, default http://localhost:8020)
                         A fast health probe decides availability per call, so
                         if the tunnel/server is down this silently falls
                         through to the cloud providers below instead of
                         hard-failing.
  OPENAI_API_KEY      gpt-4o-mini (vision) + tts-1
  GROQ_API_KEY        Llama 4 Scout vision
  GEMINI_API_KEY      gemini-2.0-flash
  OPENROUTER_API_KEY  openai/gpt-4o-mini

Local vision model: qwen2.5vl:7b via Ollama. TTS: Kokoro-82M via a small
FastAPI wrapper on the GB10 box. Both were picked deliberately small/quantized
because that GB10 is SHARED with a VLA robot-control policy at the same time —
see _local_vision()'s keep_alive=0 (Ollama evicts the model from memory right
after each call instead of holding it resident).
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = """You label tabletop objects for a robot fetch demo.
You will receive N photos, in order. Return ONLY JSON:
{"names": ["soda can", "airpods case"]}
Rules: one name per photo, 1-3 lowercase words, specific (soda can not can;
airpods case not box; plush toy not toy). No extra keys, no markdown."""

# Ollama/qwen2.5vl:7b-specific prompt, used one image per call (see
# _local_vision). Empirically, batching several images into one Ollama
# /api/chat message using the shared PROMPT above (with its concrete
# "soda can" / "airpods case" example) made this 7B model anchor on the
# example: it would return exactly those two names regardless of how many
# images were sent or what they actually showed. Calling once per image with
# a single-object prompt that deliberately avoids concrete example nouns
# fixed this in testing (verified against real snapshot crops).
LOCAL_VISION_PROMPT = """You label a single tabletop object for a robot fetch demo.
Look closely at the actual photo - do not default to a generic guess.
Return ONLY JSON: {"names": ["<answer>"]}
The answer is 1-3 lowercase words, specific and grounded in what is visible
(material, shape, color, markings) - e.g. phone charger, hair clip, tape
roll, coffee mug. No extra keys, no markdown."""


def _load_dotenv() -> None:
    here = Path(__file__).resolve().parent
    candidates = [
        here / ".env",
        here.parent / ".env",
        here.parent.parent / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()


def _local_ai_enabled() -> bool:
    return os.environ.get("LOCAL_AI", "1").strip().lower() not in ("0", "false", "no", "off")


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
KOKORO_TTS_URL = os.environ.get("KOKORO_TTS_URL", "http://localhost:8020").rstrip("/")


def _url_reachable(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def _local_vision_available() -> bool:
    return _local_ai_enabled() and _url_reachable(f"{OLLAMA_URL}/api/tags")


def _local_tts_available() -> bool:
    return _local_ai_enabled() and _url_reachable(f"{KOKORO_TTS_URL}/health")


def vision_provider() -> str | None:
    # Prioritize Gemini when available (fast, free tier, good quality)
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if _local_vision_available():
        return "local"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    return None


def tts_provider() -> str | None:
    # Prioritize ElevenLabs when available (better quality)
    if os.environ.get("ELEVENLABS_API_KEY"):
        return "elevenlabs"
    if _local_tts_available():
        return "local"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def _parse_names(text: str, expected: int) -> list[str]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if isinstance(data, dict):
        rows = data.get("names") or data.get("labels") or []
    else:
        rows = data
    names = [re.sub(r"\s+", " ", str(n)).strip().lower() for n in rows]
    names = [n for n in names if n]
    if len(names) < expected:
        names.extend([""] * (expected - len(names)))
    return names[:expected]


def _http_json(url: str, payload: dict, headers: dict, timeout: float = 25.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={**headers, "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_bytes(url: str, payload: dict, headers: dict, timeout: float = 25.0) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={**headers, "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _openai_compatible_vision(url: str, key: str, model: str, jpegs: list[bytes], extra_headers: dict | None = None) -> list[str]:
    content: list[dict] = [{"type": "text", "text": PROMPT + f"\nThere are {len(jpegs)} images."}]
    for i, jpeg in enumerate(jpegs, start=1):
        b64 = base64.b64encode(jpeg).decode("ascii")
        content.append({"type": "text", "text": f"Image {i}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    data = _http_json(
        url,
        {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        },
        {"Authorization": f"Bearer {key}", **(extra_headers or {})},
    )
    text = data["choices"][0]["message"]["content"]
    return _parse_names(text, len(jpegs))


def _local_vision(jpegs: list[bytes]) -> list[str]:
    """Ollama /api/chat, qwen2.5vl:7b (or OLLAMA_VISION_MODEL override).

    One call per image rather than one batched call with all images - see
    LOCAL_VISION_PROMPT's comment for why (batching collapsed to the
    prompt's example regardless of image count/content in testing).

    keep_alive: "30s" on every call except the last, which uses keep_alive=0
    so Ollama evicts the model from memory right after the batch finishes,
    instead of holding it resident. The GB10 this runs on is shared with a
    VLA robot-control policy, and naming is a bursty, once-per-snapshot
    workload, not continuous, so there's no reason to hold memory hostage
    between snapshots - but we also don't want to pay a full cold-load
    (~5-6s, observed) for every single crop within the same snapshot.
    """
    names: list[str] = []
    for i, jpeg in enumerate(jpegs):
        keep_alive = 0 if i == len(jpegs) - 1 else "30s"
        data = _http_json(
            f"{OLLAMA_URL}/api/chat",
            {
                "model": OLLAMA_VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": LOCAL_VISION_PROMPT,
                        "images": [base64.b64encode(jpeg).decode("ascii")],
                    }
                ],
                "stream": False,
                "format": "json",
                # num_ctx: Ollama's default context (128k tokens) balloons this
                # model's resident memory from ~6GB (weights) to ~13GB (KV
                # cache) while loaded - measured on the GB10. A single-image,
                # short-prompt naming call needs nowhere near that; capping it
                # cuts the transient footprint back to ~5.5GB, which matters
                # because that memory is shared with a VLA robot-control
                # policy running on the same box.
                # 4096 was measured to be too tight: a single full-resolution
                # image alone tokenizes to ~4072 tokens, so LOCAL_VISION_PROMPT's
                # text pushed real calls to 4142 - over the ceiling - which
                # Ollama hard-fails with a 400 instead of truncating. 8192 costs
                # only a small amount more KV-cache memory but leaves real margin.
                "options": {"temperature": 0, "num_ctx": 8192},
                "keep_alive": keep_alive,
            },
            {},
            timeout=60.0,
        )
        text = data["message"]["content"]
        names.extend(_parse_names(text, 1))
    return names


def _gemini_vision(jpegs: list[bytes]) -> list[str]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    parts: list[dict] = [{"text": PROMPT + f"\nThere are {len(jpegs)} images."}]
    for jpeg in jpegs:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode("ascii")}})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
    data = _http_json(url, {"contents": [{"parts": parts}]}, {})
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_names(text, len(jpegs))


def name_objects(jpegs: list[bytes]) -> tuple[list[str], str]:
    if not jpegs:
        return [], "none"
    provider = vision_provider()
    if provider == "local":
        return _local_vision(jpegs), "local"
    if provider == "openai":
        names = _openai_compatible_vision(
            "https://api.openai.com/v1/chat/completions",
            os.environ["OPENAI_API_KEY"],
            os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini"),
            jpegs,
        )
        return names, "openai"
    if provider == "groq":
        names = _openai_compatible_vision(
            "https://api.groq.com/openai/v1/chat/completions",
            os.environ["GROQ_API_KEY"],
            os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
            jpegs,
        )
        return names, "groq"
    if provider == "gemini":
        return _gemini_vision(jpegs), "gemini"
    if provider == "openrouter":
        names = _openai_compatible_vision(
            "https://openrouter.ai/api/v1/chat/completions",
            os.environ["OPENROUTER_API_KEY"],
            os.environ.get("OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini"),
            jpegs,
            extra_headers={"HTTP-Referer": "http://localhost:5173", "X-Title": "SSVEP Console"},
        )
        return names, "openrouter"
    raise RuntimeError("No vision API key. Set OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY.")


def _local_tts(text: str) -> bytes:
    """Small FastAPI wrapper around Kokoro-82M on the GB10 box (server.py
    next to this file's remote counterpart). Kokoro is tiny (~82M params)
    so it stays loaded there permanently - unlike the vision model, its
    residency doesn't meaningfully compete with the VLA policy for memory.
    Returns raw WAV bytes.
    """
    return _http_bytes(f"{KOKORO_TTS_URL}/speak", {"text": text[:1000]}, {}, timeout=30.0)


def _elevenlabs_tts(text: str) -> bytes:
    """ElevenLabs text-to-speech API. Returns MP3 bytes.
    Uses the default voice or ELEVENLABS_VOICE_ID env var.
    """
    key = os.environ["ELEVENLABS_API_KEY"]
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel (default)
    model_id = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    body = json.dumps({
        "text": text[:5000],
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "xi-api-key": key,
            "Accept": "audio/mpeg"
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        return resp.read()


def speak_mp3(text: str) -> tuple[bytes, str]:
    """Returns (audio_bytes, content_type). Name kept for compatibility with
    existing callers; local/Kokoro produces WAV, OpenAI/ElevenLabs produces MP3,
    so the content type now travels with the bytes instead of being assumed."""
    provider = tts_provider()
    if provider == "elevenlabs":
        return _elevenlabs_tts(text), "audio/mpeg"
    if provider == "local":
        return _local_tts(text), "audio/wav"
    if provider == "openai":
        mp3 = _http_bytes(
            "https://api.openai.com/v1/audio/speech",
            {
                "model": os.environ.get("OPENAI_TTS_MODEL", "tts-1"),
                "voice": os.environ.get("OPENAI_TTS_VOICE", "nova"),
                "input": text[:4096],
                "format": "mp3",
            },
            {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        )
        return mp3, "audio/mpeg"
    raise RuntimeError("No TTS available. Set ELEVENLABS_API_KEY, OPENAI_API_KEY, or enable LOCAL_AI/Kokoro.")

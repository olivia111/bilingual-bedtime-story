#!/usr/bin/env python3
"""Check the configured Lithos AI endpoint: reachability, params, and vision.

    ./.venv/bin/python check_lithos.py                 # text only
    ./.venv/bin/python check_lithos.py page1.jpg ...   # also test image input

Exits non-zero if a step fails, so it is usable in a pre-deploy check.
"""
import sys
import time

sys.path.insert(0, ".")

from openai import OpenAI  # noqa: E402

from app import config  # noqa: E402
from app.images import prepare_images  # noqa: E402
from app.lithos_client import _data_url  # noqa: E402


def client() -> OpenAI:
    if not config.LITHOS_API_KEY:
        sys.exit("LITHOSAI_API_KEY is not set (check .env).")
    return OpenAI(api_key=config.LITHOS_API_KEY, base_url=config.LITHOS_BASE_URL,
                  timeout=120, max_retries=0)


def call(c: OpenAI, label: str, **kwargs) -> bool:
    t = time.time()
    try:
        r = c.chat.completions.create(model=config.LITHOS_MODEL, **kwargs)
    except Exception as exc:
        print(f"  FAIL  {label}: {type(exc).__name__}: {str(exc)[:300]}")
        return False
    reply = (r.choices[0].message.content or "").strip().replace("\n", " ")
    usage = getattr(r, "usage", None)
    tokens = f", {usage.total_tokens} tokens" if usage else ""
    print(f"  ok    {label}  ({time.time() - t:.1f}s{tokens}): {reply[:120]}")
    return True


def main() -> None:
    print(f"endpoint : {config.LITHOS_BASE_URL}")
    print(f"model    : {config.LITHOS_MODEL}")
    print(f"effort   : {config.LITHOS_REASONING_EFFORT or '(omitted)'}\n")
    c = client()
    hello = [{"role": "user", "content": "Reply with exactly: pong"}]
    ok = True

    # 1. Plain call — does the endpoint work at all?
    ok &= call(c, "plain chat", messages=hello)

    # 2. reasoning_effort — supported by this model, or does it 400?
    if config.LITHOS_REASONING_EFFORT:
        if not call(c, f"reasoning_effort={config.LITHOS_REASONING_EFFORT}",
                    messages=hello, reasoning_effort=config.LITHOS_REASONING_EFFORT):
            print("        -> set LITHOSAI_REASONING_EFFORT= (empty) in .env to omit it")
            ok = False

    # 3. response_format json_schema — is the schema enforced server-side?
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    if not call(c, "response_format=json_schema",
                messages=[{"role": "user", "content": 'Reply {"ok": true}'}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "probe", "schema": schema}}):
        print("        -> the app falls back to json_object, so this is not fatal")

    # 4. Vision — the one that decides whether this app can use this model.
    paths = sys.argv[1:]
    if not paths:
        print("\n  (no images given — pass page photos to test vision)")
    else:
        raw = []
        for p in paths:
            with open(p, "rb") as f:
                ext = p.lower().rsplit(".", 1)[-1]
                mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                        "heic": "image/heic", "heif": "image/heif",
                        "webp": "image/webp"}.get(ext, "image/jpeg")
                raw.append((f.read(), mime))
        pages = prepare_images(raw)
        print(f"\n  {len(pages)} page(s) after resize: "
              + ", ".join(f"{len(d)//1024} KB {m}" for d, m in pages))
        content = [{"type": "text",
                    "text": "What Chinese text do you see on these pages? "
                            "Transcribe it exactly. If you cannot see an image, say NO IMAGE."}]
        for d, m in pages:
            content.append({"type": "image_url", "image_url": {"url": _data_url(d, m)}})
        vision_ok = call(c, "vision (image_url)", messages=[{"role": "user", "content": content}])
        if not vision_ok:
            print("        -> this model may be text-only; page reading will not work")
        ok &= vision_ok

    print("\nAll good." if ok else "\nSomething failed above.")
    sys.exit(0 if ok else 1)


main()

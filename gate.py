"""What a public deployment needs in front of the generator: input limits, per-visitor rate limits,
the communal daily budget, the visitor's own key. No generation logic lives here.

Nothing a visitor sends is written to disk. The only file is the day's communal spend."""
from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone

MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "100"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "40"))                      # user messages per conversation
RATE_PER_MINUTE = int(os.environ.get("RATE_PER_MINUTE", "6"))           # per visitor, on the communal key
RATE_PER_DAY = int(os.environ.get("RATE_PER_DAY", "80"))
OWN_KEY_RATE_PER_MINUTE = int(os.environ.get("OWN_KEY_RATE_PER_MINUTE", "20"))
COMMUNAL_DAILY_USD = float(os.environ.get("COMMUNAL_DAILY_USD", "5"))
USD_PER_MTOK = float(os.environ.get("USD_PER_MTOK", "0.042"))           # TypeSafe's listed input price
CHARS_PER_TOKEN = 4.0
SESSION_TTL = int(os.environ.get("SESSION_TTL", "7200"))
BUSY_TTL = 180                                                          # longer than any reply                # seconds a conversation may sit idle
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "2000"))
DATA_DIR = os.environ.get("DATA_DIR", "data")
TRUST_PROXY = os.environ.get("TRUST_PROXY", "0") == "1"                 # behind cloudflared: CF-Connecting-IP
OR_KEY_RE = re.compile(r"^sk-or-[A-Za-z0-9_-]{20,200}$")
TS_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{20,200}$")


class Refused(Exception):
    """A request the gate does not let through. `code` is what the page shows a message for."""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def visitor(headers, client_host: str | None) -> str:
    """Who is asking. Behind the tunnel the peer is cloudflared, so Cloudflare's header is the visitor."""
    if TRUST_PROXY:
        ip = headers.get("cf-connecting-ip") or (headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if ip:
            return ip
    return client_host or "unknown"


def clean_key(raw) -> str | None:
    """The visitor's own API key (OpenRouter or TypeSafe), or None. Never logged, never stored."""
    key = (raw or "").strip() if isinstance(raw, str) else ""
    if not key:
        return None
    if not (OR_KEY_RE.match(key) or TS_KEY_RE.match(key)):
        raise Refused("key", "that does not look like a valid api key", 400)
    return key


MAX_BODY_BYTES = 4096
# The X bot (xbot/) proves itself with a shared token and may send a whole thread as one message.
XBOT_TOKEN = os.environ.get("XBOT_TOKEN", "")
XBOT_MAX_MESSAGE_CHARS = int(os.environ.get("XBOT_MAX_MESSAGE_CHARS", "1500"))
XBOT_MAX_BODY_BYTES = 16384


def is_bot(headers) -> bool:
    return bool(XBOT_TOKEN) and headers.get("authorization") == f"Bearer {XBOT_TOKEN}"


def clean_message(raw, bot: bool = False) -> str:
    """The message as sent; the bot's keeps its lines, one per post of the thread."""
    if not bot:
        return clean_text(raw, MAX_MESSAGE_CHARS + 1)
    lines = [clean_text(line, XBOT_MAX_MESSAGE_CHARS + 1) for line in str(raw).split("\n")]
    return "\n".join(line for line in lines if line)[:XBOT_MAX_MESSAGE_CHARS + 1]


async def read_json(request, limit: int = MAX_BODY_BYTES) -> dict:
    """The request body as a JSON object, read with a hard cap: nothing larger than a chat message
    plus a key is ever needed, and an unbounded read is a memory attack."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise Refused("too_large", "that request is too large", 413)
    data = b""
    async for chunk in request.stream():
        data += chunk
        if len(data) > limit:
            raise Refused("too_large", "that request is too large", 413)
    try:
        body = json.loads(data or b"{}")
    except ValueError:
        raise Refused("bad_json", "that request is not valid json", 400)
    if not isinstance(body, dict):
        raise Refused("bad_json", "that request is not a json object", 400)
    return body


def clean_text(raw, limit: int) -> str:
    """Printable text only, whitespace collapsed, cut to `limit`: control and zero-width characters
    have no business in a chat message or an id."""
    text = "".join(ch if ch.isprintable() else " " for ch in str(raw))
    return " ".join(text.split())[:limit]


def clean_time(raw) -> str | None:
    """The browser's date and time goes into Jev's state, so only what a date looks like passes."""
    text = clean_text(raw or "", 60)
    return text if text and re.fullmatch(r"[A-Za-z0-9 ,:]+", text) else None


def check_message(message: str, turns_so_far: int, bot: bool = False) -> None:
    limit = XBOT_MAX_MESSAGE_CHARS if bot else MAX_MESSAGE_CHARS
    if not message:
        raise Refused("empty", "say something", 400)
    if len(message) > limit:
        raise Refused("too_long", f"keep it under {limit} characters", 400)
    if turns_so_far >= MAX_TURNS:
        raise Refused("too_many_turns", "this conversation is long enough; start a new chat", 400)


class RateLimiter:
    """Sliding minute window and a per-day count, per visitor, in memory."""

    def __init__(self):
        self.minute: dict[str, deque[float]] = {}
        self.day: dict[str, tuple[str, int]] = {}
        self.busy: dict[str, float] = {}  # visitor -> when their reply started

    def admit(self, who: str, own_key: bool, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if now - self.busy.get(who, -1e9) < BUSY_TTL:  # a mark older than any reply can be is stale
            raise Refused("busy", "one at a time: wait for the reply", 429)
        window = self.minute.setdefault(who, deque())
        while window and now - window[0] > 60:
            window.popleft()
        limit = OWN_KEY_RATE_PER_MINUTE if own_key else RATE_PER_MINUTE
        if len(window) >= limit:
            raise Refused("rate", "too fast: wait a minute", 429)
        today = _today()
        date, count = self.day.get(who, (today, 0))
        if date != today:
            count = 0
        if not own_key and count >= RATE_PER_DAY:
            raise Refused("rate_day", "that is all for today on the shared key; add your own key or come back tomorrow", 429)
        window.append(now)
        self.day[who] = (today, count + (0 if own_key else 1))
        if len(self.minute) > 20000:  # a flood of visitors: forget the idle ones
            for k in [k for k, w in self.minute.items() if not w or now - w[-1] > 3600]:
                self.minute.pop(k, None)
                self.day.pop(k, None)


class Budget:
    """The communal key's spend for the current UTC day, estimated from what is sent, kept across
    restarts in DATA_DIR/budget.json. The provider's own daily limit on the key is the hard stop;
    this is the friendly one."""

    def __init__(self, name: str = "", path: str | None = None):
        filename = f"budget_{name}.json" if name else "budget.json"
        self.path = path or os.path.join(DATA_DIR, filename)
        self.date, self.usd, self._saved = _today(), 0.0, 0.0
        try:
            with open(self.path) as fh:
                d = json.load(fh)
            if d.get("date") == self.date:
                self.usd = self._saved = float(d.get("usd", 0.0))
        except (OSError, ValueError):
            pass

    def _roll(self) -> None:
        if self.date != _today():
            self.date, self.usd, self._saved = _today(), 0.0, 0.0

    def add_chars(self, n: int) -> None:
        self._roll()
        self.usd += n / CHARS_PER_TOKEN * USD_PER_MTOK / 1e6
        if self.usd - self._saved >= 0.01:
            self.save()

    def exhausted(self) -> bool:
        self._roll()
        return self.usd >= COMMUNAL_DAILY_USD

    def left(self) -> float:
        self._roll()
        return max(COMMUNAL_DAILY_USD - self.usd, 0.0)

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"date": self.date, "usd": round(self.usd, 4)}, fh)
            os.replace(tmp, self.path)
            self._saved = self.usd
        except OSError:
            pass


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def evict_idle(sessions: dict, selves: dict, touched: dict[str, float], now: float | None = None) -> None:
    """Conversations live in memory only; idle ones are forgotten, and the oldest go first past the cap."""
    now = time.time() if now is None else now
    for sid in [s for s, t in touched.items() if now - t > SESSION_TTL]:
        sessions.pop(sid, None), selves.pop(sid, None), touched.pop(sid, None)
    if len(touched) > MAX_SESSIONS:
        for sid, _ in sorted(touched.items(), key=lambda kv: kv[1])[: len(touched) - MAX_SESSIONS]:
            sessions.pop(sid, None), selves.pop(sid, None), touched.pop(sid, None)

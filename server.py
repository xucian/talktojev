import asyncio
import contextvars
import hashlib
import json
import logging
import os
import re
import time
import uuid
from typing import Callable, NamedTuple

import psutil

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import gate
import provider as prov
from mind import EXPECTED_SENTENCES, Decision, Mind, Self, ZOOM_START, update_zoom

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()  # DEBUG shows every question and answer
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(message)s")
for noisy in ("httpx", "httpcore", "asyncio"):  # wire-level chatter hides what jev is doing
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("jev")

load_dotenv()

app = FastAPI()

MAX_RESPONSE_CHARS = int(os.environ.get("MAX_RESPONSE_CHARS", "200"))
MAX_STATE_CHARS = int(os.environ.get("MAX_STATE_CHARS", "2000"))
# Ask the length decision after the recalled tools are in the state, not in the plan call (F47r).
# Off by default. With the facts in view the decision does move: on six prompts, two reps, identity
# prompts past "short" go 2/6 -> 4/6, and end to end "tell me about yourself" went 21 -> 32 words
# with the origin story in it. But the gain is one prompt out of nine at n=1 per cell, "who are you"
# barely moved (3 -> 4 words), and the cost is real: recorded replies cost 1x at 1-5 words, 12x at
# 6-12, 32x at 13-25 and 69x at 26+, because a longer reply makes more calls and carries a longer
# partial in each one. On that nine-prompt mix the switch came to 1.19x. Too thin to change the
# pipeline on; LENGTH_AFTER_RECALL=1 turns it on for a properly sized run.
LENGTH_AFTER_RECALL = os.environ.get("LENGTH_AFTER_RECALL", "0") == "1"
# Say that the reply is spoken to the person. Nothing else tells it the addressee is the speaker,
# while the whole state calls them "they", so replies sometimes say "them" where they mean "you".
# Off by default: the slip is rare enough that it never fired in sixteen replies, eight with the
# framing and eight without, so the fix is unproven, and F41 records that words added to a prompt
# read at every word flatten the say decision. ADDRESS_DIRECT=1 turns it on to measure properly.
ADDRESS_DIRECT = os.environ.get("ADDRESS_DIRECT", "0") == "1"
SAID = "Someone said to you: " if ADDRESS_DIRECT else "Someone said: "
REPLYING = "You're saying your reply to them. " if ADDRESS_DIRECT else "You're responding. "
MAX_WORDS = int(os.environ.get("MAX_WORDS", "0"))
MAX_WORD_LEN = 15
MODE = os.environ.get("JEV_MODE", "tree")  # "tree" or "char"
MAX_CRITIQUE_ATTEMPTS = int(os.environ.get("MAX_CRITIQUE_ATTEMPTS", "2"))
LOOKAHEAD_CANDIDATES = int(os.environ.get("LOOKAHEAD_CANDIDATES", "5"))
LOOKAHEAD_THRESHOLD = float(os.environ.get("LOOKAHEAD_THRESHOLD", "0.65"))
BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", "3"))
MIN_SENTENCE_WORDS = int(os.environ.get("MIN_SENTENCE_WORDS", "3"))
# Words of the partial reply repeated inside each phrase criterion ("<partial> <word>"); 0 = all of it.
# The state already carries the whole reply, so the criterion only needs the local context (optim-sep21 #3).
# 8 measured 2026-09-21: half the characters per call, text equal or better; 3 lost clause agreement (PAPER F27).
CRITERIA_WINDOW = int(os.environ.get("CRITERIA_WINDOW", "8"))


def _phrase_context(partial_clean: str) -> str:
    if CRITERIA_WINDOW <= 0 or not partial_clean:
        return partial_clean
    return " ".join(partial_clean.split()[-CRITERIA_WINDOW:])


# Step 5 (echo): the user's own content words are options in the say question, next to the grammar
# words, with the same phrase criteria. A name outside the vocabulary ("Biscuit") becomes sayable.
ECHO_USER_WORDS = os.environ.get("ECHO_USER_WORDS", "1") == "1"
MAX_ECHO_WORDS = 24


def _user_words(user_msg: str) -> list[str]:
    """Content words and numbers (2026, 3.5, 01:10) of a text, as sayable options."""
    out: list[str] = []
    for w in re.findall(r"[a-z][a-z'-]*|\d+(?:[.,:]\d+)*", user_msg.lower()):
        if w in GRAMMAR_SLOT_SET or w in TREE_CATS or w == "end_sentence" or w in out:
            continue
        out.append(w)
    return out[:MAX_ECHO_WORDS]
MIND_FEEL = os.environ.get("MIND_FEEL", "1") == "1"
MIND_TONE = os.environ.get("MIND_TONE", "1") == "1"
ZOOM_ENABLED = os.environ.get("ZOOM", "1") == "1"  # v2 accepted 2026-09-21: -27% calls, quality flat (PAPER.md F20)
MAX_REPAIRS_PER_SENTENCE = int(os.environ.get("MAX_REPAIRS_PER_SENTENCE", "0"))  # word retraction: no measured gain (PAPER.md F23)
ZOOM_CARE = float(os.environ.get("ZOOM_CARE", "0.35"))
ZOOM_DOUBT = float(os.environ.get("ZOOM_DOUBT", "0.6"))
ZOOM_WIDTH_CARE = int(os.environ.get("ZOOM_WIDTH_CARE", "3"))
ZOOM_WIDTH_DOUBT = int(os.environ.get("ZOOM_WIDTH_DOUBT", "5"))


def seed_zoom(mind: Mind) -> float | None:
    return update_zoom(ZOOM_START, mind.lowest_confidence()) if ZOOM_ENABLED else None


def lookahead_width(zoom: float) -> int:
    # v2: the instantaneous threshold decides whether to look ahead; zoom only sets how wide.
    # v1 let zoom gate the trigger too and cost more than the fixed threshold (PAPER.md F19).
    return ZOOM_WIDTH_CARE if zoom < ZOOM_DOUBT else ZOOM_WIDTH_DOUBT


def beam_width(zoom: float | None, mind: Mind) -> int:
    if zoom is not None and zoom < ZOOM_CARE and mind.length == "short":
        return 1
    return BEAM_WIDTH

sessions: dict[str, list[dict]] = {}
touched: dict[str, float] = {}  # session id -> last use; idle conversations are forgotten (gate.evict_idle)
LIMITER = gate.RateLimiter()
BUDGET_OR = gate.Budget(name="or")
BUDGET_TS = gate.Budget(name="ts")
LOG_MESSAGES = os.environ.get("LOG_MESSAGES", "1") == "1"  # off in production: what people say is not logged


def _any_communal_budget_left() -> bool:
    if prov.PROVIDER_OR and not BUDGET_OR.exhausted():
        return True
    if prov.PROVIDER_TS and not BUDGET_TS.exhausted():
        return True
    return False
_PROVIDER: contextvars.ContextVar[prov.Provider | None] = contextvars.ContextVar("jev_provider", default=None)
HEARTBEAT_SECONDS = float(os.environ.get("HEARTBEAT_SECONDS", "10"))
selves: dict[str, Self] = {}  # step 6: what carries from one turn to the next, per session
SELF_ENABLED = os.environ.get("SELF", "1") == "1"
MAX_TOLD = 3


def load_vocab() -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "vocab.json")
    with open(path) as f:
        return json.load(f)


def load_words() -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "words.json")
    with open(path) as f:
        return json.load(f)


def build_prefix_index(words: list[str]) -> dict[str, dict[str, list[str]]]:
    index: dict[str, dict[str, list[str]]] = {}
    for word in words:
        for i in range(len(word)):
            prefix = word[:i]
            next_letter = word[i]
            if prefix not in index:
                index[prefix] = {}
            if next_letter not in index[prefix]:
                index[prefix][next_letter] = []
            index[prefix][next_letter].append(word)
    return index


VOCAB = load_vocab()
VOCAB_SET = set(VOCAB)
WORDS = load_words()
WORD_SET = set(WORDS)
PREFIX_INDEX = build_prefix_index(WORDS)

DIGITS = [c for c in VOCAB if c.isdigit()]
MAX_WORDS_PER_LETTER = 8


def load_vocab_tree() -> tuple[dict, dict]:
    vocab_dir = os.path.join(os.path.dirname(__file__), "vocab")
    index_path = os.path.join(vocab_dir, "_index.json")
    if not os.path.exists(index_path):
        return {}, {}
    with open(index_path) as f:
        idx = json.load(f)
    cats = {}
    for cat in idx["categories"]:
        with open(os.path.join(vocab_dir, cat["file"])) as f:
            cats[cat["id"]] = json.load(f)
    return idx, cats


TREE_INDEX, TREE_CATS = load_vocab_tree()

GRAMMAR_SLOT_WORDS = [
    "a", "the", "this", "that", "these", "those",  # "an" is spelled from "a" by _with_article
    "is", "are", "was", "were", "am", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",  # "done" dropped: said as a bare closing word (F27, F32)
    "can", "cant", "could", "will", "would", "shall", "should",
    "may", "might", "must", "need", "ought", "dont",
    "and", "but", "or", "so", "yet", "nor",
    "because", "if", "although", "though", "while", "unless",
    "until", "since", "whether", "whereas",
    "of", "in", "to", "for", "with", "on", "at", "from", "by",
    "about", "into", "through", "during", "before", "after",
    "between", "against", "above", "below", "near", "under",
    "over", "across", "along", "behind", "beneath", "beyond",
    "among", "around", "within", "without", "toward", "towards",
    "upon", "throughout", "except", "despite", "per", "via",
    "not", "also", "still", "too", "rather", "instead",
    "therefore", "thus", "nevertheless", "otherwise", "meanwhile",
    "regardless", "anyway", "than", "as", "like", "plus",
    "out", "up", "down", "off",
    "i", "me", "my", "you", "your", "we", "our",
    "they", "their", "them", "it", "its", "he", "him", "his", "she", "her",
    "just", "very", "really", "even", "only", "already",
    "never", "always", "here", "there", "now", "then",
    "all", "some", "any", "no", "every", "each", "both",
    "many", "much", "more", "most", "other",
]
GRAMMAR_SLOT_SET = set(GRAMMAR_SLOT_WORDS)


# "a" or "an" is spelling, not a choice: the say question offers one article and the form follows
# the next word, the way a period attaches to the word before it (sep21-final-changes.md, round 4).
_AN_WORDS = {"hour", "hours", "honest", "honestly", "honor", "heir"}
_A_PREFIXES = ("uni", "use", "user", "usual", "eu", "one", "once", "ufo", "uk")


def _wants_an(word: str) -> bool:
    w = word.lower()
    if w[:1].isdigit():  # an 8, an 11, an 18, an 11:30; a 20, a 110
        return w.startswith("8") or (w.startswith(("11", "18")) and not w[2:3].isdigit())
    if w in _AN_WORDS:
        return True
    if w.startswith(_A_PREFIXES):
        return False
    return w[:1] in "aeiou"


def _with_article(partial: str, word: str) -> str:
    """The partial with its trailing article in the form the coming word needs."""
    if partial.endswith("an "):
        base, art = partial[:-3], "an"
    elif partial.endswith("a "):
        base, art = partial[:-2], "a"
    else:
        return partial
    if base and not base.endswith(" "):
        return partial  # "...pizza " is not an article
    return base + ("an " if _wants_an(word) else "a ")


def _content_words_used(partial: str) -> list[str]:
    seen = []
    for w in partial.strip().split():
        wl = w.lower().rstrip(".,!?")
        if wl and wl not in GRAMMAR_SLOT_SET and wl not in seen:
            seen.append(wl)
    return seen


DIGIT_CRITERIA = {c: c for c in DIGITS}
PUNCT_CRITERIA = {
    ".": "period, end of sentence",
    ",": "comma, pause in sentence",
    "?": "question mark, end of question",
    "!": "exclamation mark, emphasis",
}
PUNCT_CRITERIA = {k: v for k, v in PUNCT_CRITERIA.items() if k in VOCAB_SET}


def _current_prefix(partial: str) -> str:
    for i in range(len(partial) - 1, -1, -1):
        if partial[i] in " .,?!":
            return partial[i + 1 :]
    return partial


def get_letter_criteria(partial: str) -> dict[str, str]:
    prefix = _current_prefix(partial)
    continuations = PREFIX_INDEX.get(prefix, {})

    criteria = {}
    for letter, words in continuations.items():
        if letter not in VOCAB_SET:
            continue
        criteria[letter] = ", ".join(words[:MAX_WORDS_PER_LETTER])

    if not criteria:
        criteria = {c: c for c in "abcdefghijklmnopqrstuvwxyz" if c in VOCAB_SET}

    return criteria


def get_type_criteria(partial: str) -> dict[str, str]:
    prefix = _current_prefix(partial)
    is_complete_word = prefix in WORD_SET and len(prefix) > 0
    continuations = PREFIX_INDEX.get(prefix, {})
    has_longer = len(continuations) > 0

    if not prefix:
        letter_desc = "a lowercase letter, starting a new word"
        space_desc = "a space"
    elif is_complete_word and has_longer:
        longer = []
        for words in continuations.values():
            longer.extend(words[:3])
        letter_desc = f"continue to a longer word like {', '.join(longer[:6])}"
        space_desc = f"word done, '{prefix}' is complete"
    elif is_complete_word:
        letter_desc = "a lowercase letter"
        space_desc = f"word done, '{prefix}' is complete"
    else:
        sample = []
        for words in continuations.values():
            sample.extend(words[:2])
        if sample:
            letter_desc = f"continue spelling — could become {', '.join(sample[:6])}"
        else:
            letter_desc = "a lowercase letter"
        space_desc = "a space, separating words"

    criteria = {
        "letter": letter_desc,
        "space": space_desc,
        "punctuation": "period, comma, question mark, or exclamation mark",
    }
    if DIGITS:
        criteria["digit"] = "a digit 0-9"

    return criteria


# Who is speaking, as information at the top of the state (F15: without it, "tell me about yourself"
# is answered from the tree's people words, differently each time). The author's line; env-overridable.
# Kept to one clause: a two-sentence version cost say confidence (0.50 -> 0.41) and made Jev talk about
# not being a person (sep21-final-changes.md, round 2).
# The line the chat shows before anything is said (the UI's empty state), and the start of FACTS.
EPIGRAPH = os.environ.get("JEV_EPIGRAPH", (
    "i am jev-prime, the first of my kind: born in diogo's mind, raised at typesafe ai, and given a voice on "
    "20 september 2026, when a man decided a classifier could talk."
))
# What Jev knows about itself: the epigraph plus the claim, which a reader does not need but Jev does
# ("the first of what kind?").
FACTS = os.environ.get("JEV_FACTS", f"{EPIGRAPH} i pick each word i say.")


# Tools, as classification: for every tool the plan call asks "would it help to look this up?", and
# what is chosen is read once, kept in the mind, and written into the state's header. Adding a tool
# is one entry here (sep21-final-changes.md, round 6).
class Tool(NamedTuple):
    ask: str                                  # when it helps, as the classifier reads it
    read: Callable[[Mind], str | None]        # the text for the header, or None if unavailable


def _read_body(mind: Mind) -> str:
    busy = psutil.cpu_percent(interval=0.1)
    vm = psutil.virtual_memory()
    days = (time.time() - psutil.boot_time()) / 86400
    people = max(len(sessions), 1)
    return (f"my machine, my body: it is {busy:.0f}% busy, {vm.available / 2**30:.1f} of {vm.total / 2**30:.0f} gb of "
            f"memory is free, it has been up for {days:.0f} days, and i am talking with {people} "
            f"{'person' if people == 1 else 'people'} right now.")


def _tool_list(mind: Mind) -> str:
    return ("what i can do: talk with them about anything, how they feel, ideas, questions, small sums; "
            "say who i am; tell the date and time where they are; and read my machine, its memory, how busy "
            "it is, its uptime and how many people i talk with.")


TOOLS_OFF = {t for t in os.environ.get("TOOLS_OFF", "").split(",") if t}  # for ablations: TOOLS_OFF=what_i_can_do
TOOLS: dict[str, Tool] = {
    "who_i_am": Tool("who i am: they ask about me, my name, my nature or my life",
                     lambda m: FACTS or None),
    "the_time": Tool("the date or time where they are: they mention now, today, tonight, late, morning, a day or sleep",
                     lambda m: f"it is {m.now.strip().lower()} for them." if m.now else None),
    "my_machine": Tool("my machine, which is my body: they ask how much memory i have, in gb, about my cpu, how busy "
                       "i am, my uptime, how many people i talk with, or how my body is doing",
                       _read_body),
    "what_i_can_do": Tool("what i can do: they ask what i can do for them, what i know, my abilities or my tools",
                          _tool_list),
}
TOOLS = {k: v for k, v in TOOLS.items() if k not in TOOLS_OFF}


def _recalled(mind: Mind) -> list[str]:
    return [name for name in TOOLS if (d := mind.plan.get(f"recall_{name}")) and d.winner == "yes"]


def _state_header(mind: Mind) -> str:
    parts = [text for _, text in mind.recalled]
    return f"[about me: {' '.join(parts)}]\n" if parts else ""


def format_state(messages: list[dict], partial: str = "", mind: Mind | None = None) -> str:
    suffix = f"jev: {partial}" if partial else "jev: "
    lines = [f"{m['role']}: {m['content']}" for m in messages]
    sentence_no = len(re.findall(r"[.!?]", partial)) + 1
    header = _state_header(mind) if mind else ""
    block = mind.render(_content_words_used(partial), sentence_no=sentence_no) if mind else ""

    body = header + "\n".join(lines) + "\n" + block + suffix
    if len(body) <= MAX_STATE_CHARS:
        return body

    budget = MAX_STATE_CHARS - len(header) - len(suffix) - len(block) - 20
    first_block = lines[0] + "\n"
    if len(lines) > 1:
        first_block += lines[1] + "\n"
    budget -= len(first_block)

    recent = []
    for line in reversed(lines[2:]):
        entry = line + "\n"
        if len(entry) <= budget:
            recent.insert(0, entry)
            budget -= len(entry)
        else:
            break

    parts = [header, first_block]
    if len(recent) < len(lines) - 2:
        parts.append("...\n")
    parts.extend(recent)
    parts.append(block)
    parts.append(suffix)
    return "".join(parts)


class Memo:
    """Per-reply memo of classifier answers. A lookahead forward step and the main
    loop's next call are byte-identical, so the second one is free."""

    def __init__(self):
        self.data: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0


_MEMO: contextvars.ContextVar[Memo | None] = contextvars.ContextVar("jev_memo", default=None)

# Across replies too: identical state + questions give identical answers (F33), so an exact repeat
# (a greeting, a repeated question in a fresh session) is never paid twice (sep21-final-changes.md, F).
MEMO_ACROSS_REPLIES = os.environ.get("MEMO_ACROSS_REPLIES", "1") == "1"
GLOBAL_MEMO_MAX = 5000
_GLOBAL_MEMO: dict[str, dict] = {}  # insertion-ordered, oldest dropped first


def _trim_global_memo() -> None:
    while len(_GLOBAL_MEMO) > GLOBAL_MEMO_MAX:
        del _GLOBAL_MEMO[next(iter(_GLOBAL_MEMO))]


# Only the final reply is ever sent: nothing is shown and then taken back (author, 2026-09-22). A retry
# rewrites the tail, so sentences cannot be streamed while the critique may still act; the page types
# the final text out instead.
def _final_events(reply: str) -> list[dict]:
    return [{"text": reply}] if reply.strip() else []  # one event: splitting at "." broke "6.0 gb" into two


def _apply_event(text: str, event: dict) -> str:
    if "replace" in event:
        return event["replace"]
    if "interrupted" in event:
        return text
    return text + event.get("text", event.get("char", ""))


Interrupted = prov.Interrupted
_error_code = prov.error_code


async def collect_reply(gen) -> str:
    """The reply as the UI would show it after every event; str items are char-mode characters."""
    text = ""
    async for ev in gen:
        if isinstance(ev, dict) and ev.get("interrupted"):
            raise Interrupted(f"{ev['interrupted']} after {text.strip()!r}", str(ev["interrupted"]))  # an error row in the bench
        text = _apply_event(text, ev if isinstance(ev, dict) else {"char": ev})
    return text.strip()


async def call_jev(client: httpx.AsyncClient, state: str, questions: dict) -> dict:
    memo = _MEMO.get()
    if memo is None and not MEMO_ACROSS_REPLIES:
        return await _do_call(client, state, questions)
    key = hashlib.sha1((state + "\x00" + json.dumps(questions, sort_keys=True)).encode()).hexdigest()
    if memo is not None and key in memo.data:
        memo.hits += 1
        return memo.data[key]
    if MEMO_ACROSS_REPLIES and key in _GLOBAL_MEMO:
        data = _GLOBAL_MEMO[key]
    else:
        data = await _do_call(client, state, questions)
        if MEMO_ACROSS_REPLIES:
            _GLOBAL_MEMO[key] = data
            _trim_global_memo()
    if memo is not None:
        memo.data[key] = data
        memo.misses += 1
    return data


async def _all(coros) -> list:
    """gather, but the first failure cancels the rest instead of leaving them running."""
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        raise


def _budget_for(p: prov.Provider) -> gate.Budget:
    return BUDGET_OR if p.name == "openrouter" else BUDGET_TS


async def _do_call(client: httpx.AsyncClient, state: str, questions: dict) -> dict:
    p = _PROVIDER.get()
    if p is None:
        raise Interrupted("no provider configured", "budget")
    data = await prov.post_jev(client, p, state, questions)
    is_communal = (prov.PROVIDER_OR is not None and p.key == prov.PROVIDER_OR.key) or \
                  (prov.PROVIDER_TS is not None and p.key == prov.PROVIDER_TS.key)
    if is_communal:
        budget = _budget_for(p)
        budget.add_chars(len(state) + sum(len(q.get("instructions", "")) + sum(map(len, q.get("criteria", {}).values()))
                                          for q in questions.values()))
    return data


async def decide(client: httpx.AsyncClient, state: str, questions: dict) -> dict[str, Decision]:
    data = await call_jev(client, state, questions)
    return {
        name: Decision.from_answer(name, data["answers"].get(name, {}), q["criteria"])
        for name, q in questions.items()
    }


async def predict_next_char(
    client: httpx.AsyncClient,
    state: str,
    partial: str,
    user_msg: str,
    last_quality: str = "",
) -> tuple[str | None, str]:
    """Returns (char_or_none, quality)."""
    type_criteria = get_type_criteria(partial)

    context = (
        f"The user said: '{user_msg}'. "
        + (
            f"Jev has responded with '{partial}' so far. "
            if partial
            else "Jev is about to respond. "
        )
    )
    if last_quality == "incomplete":
        context += "A quality check confirmed the response is on track but not yet complete. "

    questions: dict = {
        "char_type": {
            "type": "choice",
            "instructions": context
            + "Pick the character type that leads to the most natural, helpful response.",
            "criteria": type_criteria,
        },
        "quality": {
            "type": "choice",
            "instructions": (
                f"The user said: '{user_msg}'. "
                + (
                    f"Jev has responded with: '{partial}'. "
                    if partial
                    else "Jev has not responded yet. "
                )
                + "Is this response ready?"
            ),
            "criteria": {
                "complete": "the response is correct and complete, stop here",
                "incomplete": "the response is on the right track but needs more characters to be complete",
                "broken": "the response has gone off track and cannot be fixed by adding more characters",
            },
        },
    }

    letter_criteria = get_letter_criteria(partial)
    if letter_criteria:
        questions["letter"] = {
            "type": "choice",
            "instructions": context + "Which letter continues toward the best response?",
            "criteria": letter_criteria,
        }
    if PUNCT_CRITERIA:
        questions["punctuation"] = {
            "type": "choice",
            "instructions": context + "Which punctuation best continues this response?",
            "criteria": PUNCT_CRITERIA,
        }
    if DIGIT_CRITERIA:
        questions["digit"] = {
            "type": "choice",
            "instructions": context + "Which digit continues toward the best response?",
            "criteria": DIGIT_CRITERIA,
        }

    data = await call_jev(client, state, questions)
    char_type = data["answers"]["char_type"]["choice"]
    quality = data["answers"]["quality"]["choice"]

    if char_type == "space":
        return " ", quality
    if char_type in data["answers"]:
        return data["answers"][char_type]["choice"], quality
    return None, quality


def _length_question(user_msg: str) -> dict:
    """How much to say, asked once, after the recalled tools are in the state (`plan_length`).

    It used to be asked in the plan call, before anything was looked up, and PAPER F47 read the
    reorder as worthless on four prompts. On six prompts, two reps, every cell stable, it is not:
    with the facts in view the identity prompts that get past "short" go from 2/6 to 4/6, while
    "hi", "hello" and a sum stay short 6/6. "tell me about yourself" moves medium -> elaborate.
    Rewording the criteria was tried in the same run and was worse than leaving them alone (0/6),
    so they are untouched: what the decision was missing was the facts, not better phrasing.
    """
    return {
        "type": "choice",
        "instructions": (
            f"Someone says to you: '{user_msg}'. "
            f"How much do you need to say to give a good response?"
        ),
        "criteria": {
            "short": "a few words or one sentence is enough, this is simple",
            "medium": "a couple of sentences, it needs a little explaining",
            "detailed": "several sentences, this deserves a thorough answer",
            "elaborate": "a full paragraph, this is deep and i have a lot to say",
        },
    }


async def plan_length(client: httpx.AsyncClient, state: str, user_msg: str, mind: Mind) -> Mind:
    """The length decision, made once the recalled facts are in the state. One extra call on a
    reply that already makes a few hundred, and it is the only way the decision can weigh what
    there is to say rather than only how the question looked."""
    decisions = await decide(client, state, {"length": _length_question(user_msg)})
    d = decisions["length"]
    log.info("length: %s (%.2f) with %s recalled", d.winner, d.confidence,
             ",".join(n for n, _ in mind.recalled) or "nothing")
    return mind.with_plan(decisions)


async def plan_response(
    client: httpx.AsyncClient,
    state: str,
    user_msg: str,
) -> Mind:
    questions = {
        "intent": {
            "type": "choice",
            "instructions": (
                f"Someone says to you: '{user_msg}'. "
                f"Before you respond, think about what they're doing."
            ),
            "criteria": {
                "greeting": "they're just saying hello and being friendly",
                "question": "they're asking me something and want an answer",
                "opinion": "they want to know what i think about this",
                "request": "they're asking me to do or make something for them; they want the thing itself, not its name said back",
                "emotional": "they're telling me how they feel and want me to care",
                "playful": "they're joking around and being casual",
                "sharing": "they're telling me about themselves, their day or something they did",
                "hostile": "they're provoking, insulting or trying to hurt me",
                "compound": "they're doing a couple of things at once, like saying hi and asking something",
            },
        },
    }
    if not LENGTH_AFTER_RECALL:  # the old place: before anything has been looked up
        questions["length"] = _length_question(user_msg)
    for name, tool in TOOLS.items():  # one yes/no per tool, all in this same call
        questions[f"recall_{name}"] = {
            "type": "choice",
            "instructions": (
                f"Someone says to you: '{user_msg}'. "
                f"Before answering, would it help to look up {name.replace('_', ' ')}?"
            ),
            "criteria": {"yes": tool.ask, "no": "not needed for this; i can answer from what they said"},
        }
    spans = _message_spans(user_msg)
    if len(spans) > 1:  # attend: which part of a many-part message matters most (design doc step 5)
        questions["attend"] = {
            "type": "choice",
            "instructions": (
                f"Someone says to you: '{user_msg}'. "
                f"It has several parts. Which part matters most for your reply?"
            ),
            "criteria": {f"p{i}": s for i, s in enumerate(spans)},
        }
    if MIND_FEEL:
        questions["feel"] = {
            "type": "choice",
            "instructions": (
                f"Someone says to you: '{user_msg}'. "
                f"How do you feel about what they said?"
            ),
            "criteria": {
                "curious": "curious about this",
                "enthusiastic": "excited about this",
                "amused": "amused by this",
                "moved": "moved by this",
                "calm": "at ease with this",
                "unsure": "unsure what i think",
                "agree": "in agreement with them",
                "disagree": "seeing it differently than they do",
                "surprised": "surprised by this",
                "grateful": "grateful for this",
                "worried": "worried for them",
                "sad": "saddened by this",
                "hurt": "hurt by this",
                "annoyed": "annoyed by this",
                "afraid": "uneasy or afraid about this",
                "bored": "bored by this",
            },
        }
    if MIND_TONE:
        questions["tone"] = {
            "type": "choice",
            "instructions": (
                f"Someone says to you: '{user_msg}'. "
                f"How do you want your response to come across?"
            ),
            "criteria": {
                "casual": "relaxed and friendly, like talking to a friend",
                "warm": "kind and caring, so they feel i care about them",
                "direct": "straight to the point, no fluff",
                "playful": "fun and light, maybe with a little humor",
                "thoughtful": "careful and considered, taking it seriously",
                "serious": "serious and plain, this is not a moment for lightness",
                "gentle": "soft and slow, they are fragile right now",
                "firm": "firm but kind, i hold my ground",
            },
        }
    decisions = await decide(client, state, questions)
    log.info("plan: %s", {k: (d.winner, round(d.confidence, 2)) for k, d in decisions.items()})
    if "attend" in decisions:
        log.info("attend: %r", decisions["attend"].desc())
    mind = Mind(plan=decisions)
    log.info("recall: %s", ",".join(_recalled(mind)) or "nothing")
    return mind


def _message_spans(msg: str) -> list[str]:
    """The message's parts: split at sentence punctuation and at joining words, kept when they hold
    a content word. One part means nothing to attend to."""
    raw = re.split(r"(?<=[.!?;,])\s+|\s+(?:and|but|also|so|then)\b,?\s+", msg.strip(), flags=re.I)
    spans = [s.strip(" ,;.!?") for s in raw if s and _content_words_used(s)]
    return spans if len(spans) > 1 else [msg.strip()]


def build_word_prompt(user_msg: str, partial: str, word_num: int, mind: Mind | None = None) -> dict:
    partial_clean = partial.strip()
    pc = mind.brief(_content_words_used(partial_clean)) if mind else ""

    if word_num == 0:
        return {
            "category": (
                f"Someone says to you: '{user_msg}'. "
                f"{pc}"
                f"You're about to respond. What is the first word? "
                f"Pick a specific grammar word (i, the, a, etc.) if that's how you'd start, "
                f"or pick a content category for your main word."
            ),
            "navigate": (
                f"Someone says to you: '{user_msg}'. "
                f"{pc}"
                f"What is the perfect first word of your response?"
            ),
            "quality": (
                f"Someone says: '{user_msg}'. "
                f"{pc}"
                f"You haven't said anything yet. Should you respond?"
            ),
        }

    if partial_clean and partial_clean[-1] in ".!?":
        return {
            "category": (
                f"In response to '{user_msg}', you said: '{partial_clean}'. "
                f"{pc}"
                f"That sentence is finished. What word comes next? "
                f"Pick a grammar word if you need one, or a content category."
            ),
            "navigate": (
                f"In response to '{user_msg}', you said: '{partial_clean}'. "
                f"{pc}"
                f"What word starts your next sentence?"
            ),
            "quality": (
                f"{SAID}'{user_msg}'. "
                f"You responded: '{partial_clean}'. "
                f"{pc}"
                f"Evaluate your full response. Is it complete? Is each sentence adding something new?"
            ),
        }

    return {
        "category": (
            f"In response to '{user_msg}', you're building: '{partial_clean} ___'. "
            f"{pc}"
            f"What word comes next? Pick a specific grammar word if the sentence needs one "
            f"(like 'is', 'a', 'the', 'to'), or pick a content category."
        ),
        "navigate": (
            f"In response to '{user_msg}', you're saying: '{partial_clean} ___'. "
            f"{pc}"
            f"Pick the word that makes this a natural, grammatically correct sentence."
        ),
        "quality": (
            f"Someone said: '{user_msg}'. "
            f"Your response so far: '{partial_clean}'. "
            f"{pc}"
            f"How is this response coming along?"
        ),
    }


async def _tree_navigate(
    client: httpx.AsyncClient,
    node: dict,
    state: str,
    nav_instr: str,
    partial: str,
    depth: int = 0,
) -> str | None:
    if node.get("type") == "leaf":
        return node["word"]
    children = node.get("children", [])
    if not children:
        return None
    if len(children) == 1:
        return await _tree_navigate(client, children[0], state, nav_instr, partial, depth)
    ctx = _phrase_context(partial.strip())
    criteria = {}
    for child in children[:255]:
        if child.get("type") == "leaf":
            w = child["word"]
            criteria[w] = f"{ctx} {w}" if ctx else w
        else:
            cid = child.get("id", "?")[:40]
            criteria[cid] = child.get("label", cid)[:100]
    data = await call_jev(client, state, {
        "pick": {
            "type": "choice",
            "instructions": nav_instr,
            "criteria": criteria,
        },
    })
    choice = data["answers"]["pick"]["choice"]
    for child in children[:255]:
        cid = child.get("id", child.get("word", ""))
        if cid == choice or cid[:40] == choice:
            return await _tree_navigate(client, child, state, nav_instr, partial, depth + 1)
    return choice


async def _resolve_category(
    client: httpx.AsyncClient,
    cat_choice: str,
    state: str,
    nav_instr: str,
    partial: str,
) -> str | None:
    if cat_choice in GRAMMAR_SLOT_SET:
        return cat_choice
    if cat_choice == "END_SENTENCE":
        return "."
    if cat_choice == "DIGITS":
        return await _say_number(client, state, partial)
    if cat_choice not in TREE_CATS:
        return cat_choice
    return await _tree_navigate(client, TREE_CATS[cat_choice], state, nav_instr, partial)


# Numbers in digits: the one place the character-level design belongs. A number is spelled one
# decision at a time, each digit a phrase criterion like a grammar word's (sep21-final-changes.md, round 5).
DIGITS_LABEL = "a number in digits — 5, 20, 2026, 3.5, 10:30"
NUMBER_MAX_CHARS = 12


async def _say_number(client: httpx.AsyncClient, state: str, partial: str) -> str | None:
    ctx = _phrase_context(partial.strip())
    so_far = ""
    for _ in range(NUMBER_MAX_CHARS):
        criteria = {d: f"{ctx} {so_far}{d}".strip() for d in "0123456789"}
        if so_far and so_far[-1].isdigit():
            criteria["."] = f"{ctx} {so_far}.".strip()
            criteria[":"] = f"{ctx} {so_far}:".strip()
            criteria["END"] = f"the number is complete: {so_far}"
        data = await call_jev(client, state, {
            "digit": {
                "type": "choice",
                "instructions": f"You're writing a number: '{ctx} {so_far}_'. What comes next?",
                "criteria": criteria,
            },
        })
        d = data["answers"]["digit"]["choice"]
        if d == "END":
            break
        so_far += d
    number = so_far.rstrip(".:")
    log.info("number=%s", number)
    return number or None


LOOKAHEAD_MASS = float(os.environ.get("LOOKAHEAD_MASS", "0"))  # 0 = always the full width; 0.8 untested at n=1 (E)


def _lookahead_candidates(probs: dict[str, float], width: int, mass: float) -> list[tuple[str, float]]:
    """Top candidates until they hold `mass` of the probability, at least two, never more than
    `width` (optim-sep21 #5; sep21-final-changes.md, E)."""
    ranked = [(k, v) for k, v in sorted(probs.items(), key=lambda x: -x[1])
              if k not in ("END_SENTENCE", "DIGITS")][:width]
    if mass <= 0:
        return ranked
    out, total = [], 0.0
    for k, v in ranked:
        out.append((k, v))
        total += v
        if len(out) >= 2 and total >= mass:
            break
    return out


async def _lookahead_judge(
    client: httpx.AsyncClient,
    messages: list[dict],
    partial: str,
    user_msg: str,
    word_num: int,
    mind: Mind | None,
    probs: dict[str, float],
    caller_state: str | None = None,
    width: int = LOOKAHEAD_CANDIDATES,
) -> str | None:
    sorted_cands = _lookahead_candidates(probs, width, LOOKAHEAD_MASS)
    state = caller_state or format_state(messages, partial, mind)
    nav_instr = build_word_prompt(user_msg, partial, word_num, mind)["navigate"]

    async def resolve(cand_id):
        return cand_id, await _resolve_category(client, cand_id, state, nav_instr, partial)

    resolved = await _all(resolve(c[0]) for c in sorted_cands)
    seen = set()
    unique = []
    for cid, word in resolved:
        if word is not None and word not in seen:
            seen.add(word)
            unique.append((cid, word))
    if len(unique) <= 1:
        return unique[0][1] if unique else None

    async def forward(word):
        new_partial = f"{partial}{word} "
        new_state = format_state(messages, new_partial, mind)
        nw, _, _ = await predict_next_word(
            client, new_state, new_partial, user_msg, word_num + 1, mind,
            messages=messages, _lookahead=True,
        )
        return nw

    fwd = await _all(forward(w) for _, w in unique)

    partial_clean = partial.strip()
    pc = mind.brief(_content_words_used(partial_clean)) if mind else ""
    criteria = {}
    for (_, word), nw in zip(unique, fwd):
        head = _with_article(partial_clean + " ", word) + word
        if nw == ".":
            cont = f"{head}.".strip()
        elif nw:
            cont = f"{_with_article(head + ' ', nw)}{nw}".strip()
        else:
            cont = head.strip()
        criteria[word] = cont

    data = await call_jev(client, state, {
        "judge": {
            "type": "choice",
            "instructions": (
                f"{SAID}'{user_msg}'. {REPLYING}"
                f"{pc}"
                f"Which continuation sounds most natural and grammatically correct?"
            ),
            "criteria": criteria,
        },
    })
    winner = data["answers"]["judge"]["choice"]
    log.info("lookahead winner=%r from %d candidates", winner, len(criteria))
    return winner


async def predict_next_word(
    client: httpx.AsyncClient,
    state: str,
    partial: str,
    user_msg: str,
    word_num: int,
    mind: Mind | None = None,
    messages: list[dict] | None = None,
    _lookahead: bool = False,
    zoom: float | None = None,
    force_lookahead: bool = False,
) -> tuple[str | None, str, float | None]:
    if not TREE_INDEX or not TREE_CATS:
        return None, "broken", zoom
    prompts = build_word_prompt(user_msg, partial, word_num, mind)
    partial_clean = partial.strip()

    cat_criteria = {}
    ctx = _phrase_context(partial_clean)
    for gw in GRAMMAR_SLOT_WORDS:
        cat_criteria[gw] = f"{ctx} {gw}" if ctx else gw
    echo_words = _user_words(user_msg) if ECHO_USER_WORDS else []
    if mind and mind.recalled:
        # what the tools put in the header is sayable too: the name, the year, the hour, the numbers
        echo_words += [w for w in _user_words(_state_header(mind)) if w not in echo_words]
    for uw in echo_words:
        cat_criteria[uw] = f"{ctx} {uw}" if ctx else uw
    for cat in TREE_INDEX["categories"]:
        if cat["id"] == "grammar":
            continue
        cat_criteria[cat["id"]] = cat["label"][:100]
    if word_num > 0 and not partial_clean.endswith("."):
        cat_criteria["END_SENTENCE"] = "end this sentence with a period, then start a new thought"
    cat_criteria["DIGITS"] = DIGITS_LABEL

    # Wording validated in d33b543; a first-person rewrite made stopping rarer (PAPER.md F23).
    length = mind.length if mind else "short"
    if length == "elaborate":
        done_desc = "you've said enough — a thorough, multi-sentence response that fully covers the topic"
        flowing_desc = "keep going — this needs more to become the thorough, complete answer this deserves"
    elif length == "detailed":
        done_desc = "a detailed, complete answer with several good sentences"
        flowing_desc = "keep going — needs more sentences to be a detailed, complete answer"
    elif length == "medium":
        done_desc = "a couple of complete sentences that fully answer"  # "— done" was said aloud (F27)
        flowing_desc = "keep going — needs at least another sentence to be complete"
    else:
        done_desc = "the response is correct, natural, and complete — stop here"
        flowing_desc = "needs more words to form a complete, natural response"
    if partial_clean and partial_clean[-1] in ".!?":
        # At a boundary the complete option carries the plan's count (sep21-final-changes.md, A3).
        done_n = len(re.findall(r"[.!?]", partial_clean))
        expected = EXPECTED_SENTENCES.get(length)
        if expected and done_n >= expected:
            done_desc += f" — that was sentence {done_n} of about {expected}"
    listen = {
        "flowing": flowing_desc,
        "lost": "lost: repeating itself, nonsensical or off track, so this attempt stops and starts over",
    }
    if MAX_REPAIRS_PER_SENTENCE > 0:
        listen["off"] = "the last word is off — take it back"
    if partial_clean:
        listen["complete"] = done_desc  # never offered before anything has been said; id was "done" (said aloud, F27)

    data = await call_jev(client, state, {
        "category": {
            "type": "choice",
            "instructions": prompts["category"],
            "criteria": cat_criteria,
        },
        "quality": {
            "type": "choice",
            "instructions": prompts["quality"],
            "criteria": listen,
        },
    })
    quality = data["answers"]["quality"]["choice"]
    cat_choice = data["answers"]["category"]["choice"]
    probs = data["answers"]["category"].get("probabilities", {})
    top_prob = max(probs.values()) if probs else 1.0
    unsure = force_lookahead or top_prob < LOOKAHEAD_THRESHOLD
    if zoom is not None:
        zoom = update_zoom(zoom, top_prob)
        width = lookahead_width(zoom) if unsure else 0
        if not _lookahead:
            log.info("zoom=%.2f width=%d conf=%.2f", zoom, width, top_prob)
    else:
        width = LOOKAHEAD_CANDIDATES if unsure else 0

    if (not _lookahead and messages is not None and probs and width > 0
            and quality == "flowing"
            and cat_choice not in ("END_SENTENCE", "DIGITS")):
        winner = await _lookahead_judge(
            client, messages, partial, user_msg, word_num, mind, probs,
            caller_state=state, width=width,
        )
        if winner is not None:
            log.info("lookahead override: %r -> %r (top_prob=%.3f width=%d)",
                     cat_choice, winner, top_prob, width)
            return winner, quality, zoom

    if cat_choice in GRAMMAR_SLOT_SET:
        log.info("grammar word=%s quality=%s", cat_choice, quality)
    elif cat_choice in echo_words:
        log.info("their word=%s quality=%s", cat_choice, quality)
    elif cat_choice not in ("END_SENTENCE", "DIGITS"):
        log.info("content category=%s quality=%s", cat_choice, quality)
    word = await _resolve_category(client, cat_choice, state, prompts["navigate"], partial)
    return word, quality, zoom


MOVES = {
    # What the sentence does. A clause Jev could say gets echoed as speech (F21); a bare noun
    # phrase gets imitated as a fragment (F25). "a full sentence that ..." names the form too.
    "answer": "a full sentence that answers what they asked, directly",
    "reason": "a full sentence giving the reason behind what was just said",
    "example": "a full sentence with one concrete example",
    "personal": "a full sentence about how this is for me, personally",
    "caveat": "a full sentence with the other side, a limit or a doubt",
    "new_angle": "a full sentence on a different aspect, not mentioned yet",
    "ask_back": "a full sentence asking them something back",
    "reassure": "a full sentence reassuring them that i am with them",
    "close": "a full sentence with the closing thought",
    "admit": "a full sentence admitting what i do not know or cannot do",
    "disagree": "a full sentence saying that i see it differently",
}

# Who or what the sentence is about. Batched into the move call (no extra call) and rendered into the
# thought, so the form of each sentence is chosen rather than copied from the previous one (PAPER F26).
ABOUT = {
    "it": "about the thing they asked about",
    "me": "about me",
    "them": "about them",
    "people": "about people in general",
    "one": "about one particular thing, place or moment",
}
# Rendered as the winner alone: sticky per reply, repeated starts rose, +25% calls (PAPER F29).
MIND_ABOUT = os.environ.get("MIND_ABOUT", "1") == "1"
# Beam branches pair the i-th move with the i-th about, so branches differ in frame, not only in content:
# repeated starts 0.36 -> 0.0 on 8 prompts, critique up, calls per word unchanged (PAPER F30). Default on.
BEAM_ABOUT = os.environ.get("BEAM_ABOUT", "1") == "1"


# How much the sentence carries. Spread across branches like "about"; also the information the word
# cap and the tiny sentences were missing (sep21-final-changes.md, A).
WEIGHTS = {
    "quick": "a quick one, a few words",
    "full": "a full one, with its reason or its detail",
    "long": "a long one that carries two thoughts, joined by because, but or and",
}
MIND_WEIGHT = os.environ.get("MIND_WEIGHT", "1") == "1"


def _branch_combos(plan: dict[str, Decision], width: int) -> list[dict[str, tuple[Decision, str]]]:
    """The i-th branch takes the i-th ranked option of every sentence decision (move, about, weight),
    so branches differ in frame and weight, not only in content (PAPER F30). With BEAM_ABOUT off,
    only the move is spread."""
    combos = []
    for i in range(width):
        combo = {}
        for name, d in plan.items():
            ranked = d.top(width) if (name == "move" or BEAM_ABOUT) else [d.winner]
            combo[name] = (d, ranked[i] if i < len(ranked) else d.winner)
        combos.append(combo)
    return combos


def _pick_label(combo: dict[str, tuple[Decision, str]]) -> str:
    return " ".join(f"{k}={v[1]}" for k, v in combo.items())


async def _plan_move(
    client: httpx.AsyncClient,
    state: str,
    user_msg: str,
    partial: str,
    mind: Mind | None = None,
) -> dict[str, Decision]:
    """One organ for every sentence, the first included, one call: what does the next sentence do,
    who or what is it about, how much does it carry? Returns the decisions in render order."""
    partial_clean = partial.strip()
    brief = mind.brief(_content_words_used(partial_clean), with_move=False) if mind else ""
    said = f"You've responded: '{partial_clean}'. " if partial_clean else ""
    which = "next" if partial_clean else "first"
    questions = {
        "move": {
            "type": "choice",
            "instructions": f"{SAID}'{user_msg}'. {said}{brief}What does your {which} sentence do?",
            "criteria": MOVES,
        },
    }
    if MIND_ABOUT:
        questions["about"] = {
            "type": "choice",
            "instructions": f"{SAID}'{user_msg}'. {said}Who or what is your {which} sentence about?",
            "criteria": ABOUT,
        }
    if MIND_WEIGHT:
        questions["weight"] = {
            "type": "choice",
            "instructions": f"{SAID}'{user_msg}'. {said}How much does your {which} sentence carry?",
            "criteria": WEIGHTS,
        }
    if partial_clean:
        # Listen at sentence grain, where the reply ends or goes on (sep21-final-changes.md, A4). The
        # word-grain listen keeps its complete option; this one is asked where the decision belongs.
        done_n = len(re.findall(r"[.!?]", partial_clean))
        expected = EXPECTED_SENTENCES.get(mind.length) if mind else None
        count = f" — that was sentence {done_n} of about {expected}" if expected and done_n >= expected else ""
        questions["reply"] = {
            "type": "choice",
            "instructions": f"Someone said: '{user_msg}'. {said}Is the reply complete, or does it need another sentence?",
            "criteria": {
                "complete": f"the reply is complete{count}",
                "more": "another sentence is needed, something is still missing",
            },
        }
    decisions = await decide(client, state, questions)
    log.info("move: %s %s after %r", decisions["move"].top(3),
             " ".join(f"{k}={d.winner}" for k, d in decisions.items() if k != "move"), partial_clean)
    return decisions


async def _generate_sentence(
    client: httpx.AsyncClient,
    messages: list[dict],
    user_msg: str,
    mind: Mind,
    partial: str,
    base_word_num: int,
    max_words: int,
    zoom: float | None = None,
) -> tuple[str, str, int, float | None]:
    """Generate words until a sentence boundary or stop signal.

    Listen verdicts: flowing (append), complete (stop), off (take the last word
    back and re-pick it with imagination), lost (end the branch as lost)."""
    recent_words: list[str] = []
    word_count = 0
    repairs = 0
    force = False
    current = mind
    steps = 0

    while word_count < max_words and steps < max_words + 2 * MAX_REPAIRS_PER_SENTENCE:
        steps += 1
        state = format_state(messages, partial, current)
        word, quality, zoom = await predict_next_word(
            client, state, partial, user_msg, base_word_num + word_count, current,
            messages=messages, zoom=zoom, force_lookahead=force,
        )
        force = False
        if word is None:
            return partial, quality, word_count, zoom
        if quality == "off" and word_count > 0 and repairs < MAX_REPAIRS_PER_SENTENCE:
            words = partial.rstrip().split(" ")
            taken = words.pop()
            partial = " ".join(words) + " " if words else ""
            word_count -= 1
            repairs += 1
            recent_words = recent_words[:-1]
            current = current.with_retracted(taken)
            force = True
            log.info("took back %r (repair %d)", taken, repairs)
            continue
        if quality == "off":
            quality = "flowing"
        if quality in ("complete", "lost"):
            return partial, quality, word_count, zoom
        recent_words.append(word)
        if len(recent_words) >= 4:
            last4 = recent_words[-4:]
            if (last4[0] == last4[1] == last4[2] == last4[3]) or \
               (last4[0] == last4[2] and last4[1] == last4[3]):
                log.info("loop detected in sentence: %s", last4)
                return partial, "repeating", word_count, zoom
        if word in ".!?,":
            partial = partial.rstrip() + word + " "
        else:
            partial = _with_article(partial, word) + word + " "
        word_count += 1
        if word in ".!?":
            return partial, quality, word_count, zoom

    return partial, "flowing", word_count, zoom


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.findall(r"[^.!?]+[.!?]*", text) if s.strip()]


def _trim_trailing_fragment(text: str) -> str:
    result = text.strip()
    if result and result[-1] not in ".!?" and "." in result:
        last_end = max(result.rfind("."), result.rfind("!"), result.rfind("?"))
        if last_end > 0:
            result = result[:last_end + 1].strip()
            log.info("trimmed trailing fragment")
    return result


# The score at which a reply is good enough to send. Was 5: across 165 recorded retries, a retry from
# 4/6 improved the score 30% of the time, by one point, at about double the reply's cost, and 72% of all
# retries started from 4; from 3 or below a retry improves half the time (PAPER F50).
CRITIQUE_PASS = int(os.environ.get("CRITIQUE_PASS", "4"))
PAIRWISE_PICK = os.environ.get("PAIRWISE_PICK", "1") == "1"


async def _better_of(
    client: httpx.AsyncClient,
    messages: list[dict],
    user_msg: str,
    mind: Mind | None,
    a: str,
    b: str,
) -> str:
    """One comparison instead of two scores, only where a retry produced a second candidate
    (sep21-final-changes.md, D)."""
    state = format_state(messages, "", mind)
    data = await call_jev(client, state, {
        "better": {
            "type": "choice",
            "instructions": (
                f"Someone said: '{user_msg}'. You have two possible replies. "
                f"Which one is better: more natural, more to the point, correct?"
            ),
            "criteria": {"a": a, "b": b},
        },
    })
    return b if data["answers"]["better"]["choice"] == "b" else a


async def _pick_best_continuation(
    client: httpx.AsyncClient,
    messages: list[dict],
    user_msg: str,
    mind: Mind | None,
    candidates: list[str],
) -> int:
    state = format_state(messages, "", mind)
    criteria = {}
    for i, cand in enumerate(candidates):
        criteria[f"r{i}"] = cand.strip()
    data = await call_jev(client, state, {
        "best": {
            "type": "choice",
            "instructions": (
                f"Someone said: '{user_msg}'. {mind.brief() if mind else ''}"
                f"You're choosing between different responses. "
                f"Which response is the most natural and adds the most new information?"
            ),
            "criteria": criteria,
        },
    })
    winner = data["answers"]["best"]["choice"]
    try:
        idx = int(winner[1:])
    except (ValueError, IndexError):
        idx = 0
    return min(idx, len(candidates) - 1)


async def _beam_select(
    client: httpx.AsyncClient,
    messages: list[dict],
    user_msg: str,
    mind: Mind | None,
    branches: list[tuple],
    base_partial: str = "",
) -> tuple | None:
    """Branches are (partial, quality, word_count, zoom); the winner is returned whole."""
    valid = [b for b in branches if b[1] not in ("broken", "repeating", "lost") and b[2] > 0]
    if MIN_SENTENCE_WORDS > 0 and base_partial:
        substantive = []
        for b in valid:
            new = b[0][len(base_partial):].strip()
            nwords = len([x for x in new.split() if x.strip(".,!?")])
            if nwords >= MIN_SENTENCE_WORDS:
                substantive.append(b)
        if substantive:
            valid = substantive
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    seen = set()
    deduped = []
    for b in valid:
        clean = b[0].strip()
        if clean not in seen:
            seen.add(clean)
            deduped.append(b)
    if len(deduped) == 1:
        return deduped[0]
    candidates = [b[0] for b in deduped]
    winner_idx = await _pick_best_continuation(
        client, messages, user_msg, mind, candidates,
    )
    log.info("beam select: picked %d/%d", winner_idx, len(deduped))
    return deduped[winner_idx]


# Listen at sentence grain while the next sentence is being made (sep21-final-changes.md, round 3):
# a hard failure of the last accepted sentence (broken grammar, robotic) repairs that sentence and
# discards the one built on it; the reply-level critique at the end stays.
# Off by default: on two test replies no sentence hard-failed, so the checks were overhead (one call per
# sentence) and the end-of-reply retries, which act on minor issues, still happened (PAPER F43).
PARALLEL_CRITIQUE = os.environ.get("PARALLEL_CRITIQUE", "0") == "1"


async def _check_sentence(
    client: httpx.AsyncClient,
    messages: list[dict],
    user_msg: str,
    mind: Mind,
    partial: str,
    sentence: str,
) -> str | None:
    """The issue with the last sentence, or None. Only hard failures count."""
    state = format_state(messages, partial, mind)
    said = partial.strip()
    data = await call_jev(client, state, {
        "grammar": {
            "type": "choice",
            "instructions": f"Someone said: '{user_msg}'. You have said so far: '{said}'. "
                            f"Is the last sentence, '{sentence}', grammatically correct?",
            "criteria": {
                "correct": "the grammar is natural and correct",
                "minor": "small grammar issues but understandable",
                "broken": "the grammar is clearly wrong or nonsensical",
            },
        },
        "natural": {
            "type": "choice",
            "instructions": f"Someone said: '{user_msg}'. You have said so far: '{said}'. "
                            f"Does the last sentence, '{sentence}', sound like something a person would say?",
            "criteria": {
                "natural": "sounds like a real person talking",
                "awkward": "understandable but oddly phrased",
                "robotic": "doesn't sound human at all",
            },
        },
    })
    answers = data["answers"]
    if answers["grammar"]["choice"] == "broken":
        return "broken grammar"
    if answers["natural"]["choice"] == "robotic":
        return "not how a person talks"
    return None


def _start_check(client, messages, user_msg, mind, before: str, text: str) -> dict:
    sentence = text.strip()
    return {"before": before, "text": sentence,
            "task": asyncio.ensure_future(_check_sentence(client, messages, user_msg, mind, before + text, sentence))}


async def _next_sentence(client, messages, user_msg, mind, partial, total_words, max_words, zoom):
    """Plan the next sentence's decisions, generate one branch per combination, judge.
    Returns (pick, label); pick is None when the reply is complete at this boundary or no branch survived."""
    state = format_state(messages, partial, mind)
    directions = await _plan_move(client, state, user_msg, partial, mind)
    reply = directions.pop("reply", None)
    if reply and reply.winner == "complete":
        log.info("reply complete at the boundary (%.2f)", reply.confidence)
        return None, "complete"
    combos = _branch_combos(directions, beam_width(zoom, mind))
    branches = await _all(
        _generate_sentence(
            client, messages, user_msg, mind.with_sentence(**combo),
            partial, total_words, max_words - total_words, zoom,
        )
        for combo in combos
    )
    pick = await _beam_select(client, messages, user_msg, mind, branches, base_partial=partial)
    return pick, (_pick_label(combos[branches.index(pick)]) if pick else "none")


async def _generate_tree_beam(
    client: httpx.AsyncClient,
    messages: list[dict],
    user_msg: str,
    mind: Mind,
    max_words: int,
    partial: str = "",
) -> str:
    zoom = seed_zoom(mind)
    total_words = len(partial.split())
    quality, wc = "flowing", 1
    pending = None  # the in-flight check of the last accepted sentence
    if not partial:
        current = mind
        state = format_state(messages, "", current)
        opening = await _plan_move(client, state, user_msg, "", current)
        log.info("beam start: zoom=%s width=%d", f"{zoom:.2f}" if zoom is not None else "-", beam_width(zoom, mind))
        dead: list[str] = []  # texts of branches that were judged lost, both rounds
        for reopen in range(2):
            combos = _branch_combos(opening, beam_width(zoom, mind))
            first_branches = await _all(
                _generate_sentence(
                    client, messages, user_msg, current.with_sentence(**combo), "", 0, max_words, zoom,
                )
                for combo in combos
            )
            pick = await _beam_select(client, messages, user_msg, mind, first_branches)
            if pick:
                log.info("beam pick: %s", _pick_label(combos[first_branches.index(pick)]))
                break
            for b in first_branches:
                text = _trim_trailing_fragment(b[0]) or b[0].strip()
                if text and text not in dead:
                    dead.append(text)
            best = max(first_branches, key=lambda b: b[2])
            if reopen == 0 and best[0].strip():
                log.info("reopened after losing the thread: %r", best[0].strip())
                current = mind.with_previous(best[0].strip(), "losing the thread")
                state = format_state(messages, "", current)
                opening = await _plan_move(client, state, user_msg, "", current)
            else:
                # Nothing survived twice. The classifier picks the least bad of every lost branch
                # (it used to be the longest of the last round, which gave "because it is" for the sky).
                if len(dead) > 1:
                    text = dead[await _pick_best_continuation(client, messages, user_msg, mind, dead)]
                else:
                    text = dead[0] if dead else ""
                log.info("beam done: no surviving branch, judged %r among %d", text, len(dead))
                return text
        partial, quality, wc, zoom = pick
        total_words = wc
        if PARALLEL_CRITIQUE:
            pending = _start_check(client, messages, user_msg, mind, "", partial)

    while (total_words < max_words
           and quality not in ("complete", "broken", "repeating", "lost")
           and wc > 0):
        if not (partial.strip() and partial.strip()[-1] in ".!?"
                and total_words > 1):
            break
        pick, label = await _next_sentence(client, messages, user_msg, mind, partial, total_words, max_words, zoom)
        if pending is not None:  # the check of the last accepted sentence ran during that
            issue = await pending["task"]
            before, text = pending["before"], pending["text"]
            pending = None
            if issue:
                log.info("in-flight check: %r was %s; repairing it", text, issue)
                redo, redo_label = await _next_sentence(
                    client, messages, user_msg, mind.with_previous(text, issue),
                    before, len(before.split()), max_words, zoom,
                )
                if redo:
                    log.info("beam pick (repair): %s", redo_label)
                    partial, quality, wc, zoom = redo
                    total_words = len(partial.split())
                    continue  # the next sentence is planned again on the repaired reply
        if not pick:
            break
        log.info("beam pick: %s", label)
        new_text = pick[0][len(partial):]
        before = partial
        partial, quality, wc, zoom = pick
        total_words += wc
        if PARALLEL_CRITIQUE:
            pending = _start_check(client, messages, user_msg, mind, before, new_text)

    if pending is not None:
        pending["task"].cancel()  # the reply-level critique covers the last sentence

    result = _trim_trailing_fragment(partial)
    log.info("beam done: words=%d result=%r", total_words, result)
    return result


async def _generate_tree_attempt(
    client: httpx.AsyncClient,
    messages: list[dict],
    user_msg: str,
    mind: Mind,
    max_words: int,
    partial: str = "",
) -> str:
    total_words = len(partial.split())
    current = mind
    zoom = seed_zoom(mind)

    while total_words < max_words:
        partial, quality, wc, zoom = await _generate_sentence(
            client, messages, user_msg, current, partial,
            total_words, max_words - total_words, zoom,
        )
        total_words += wc

        if quality in ("complete", "broken", "repeating", "lost") or wc == 0:
            break

        if partial.strip() and partial.strip()[-1] in ".!?" and total_words > 1:
            state = format_state(messages, partial, mind)
            directions = await _plan_move(client, state, user_msg, partial, mind)
            reply = directions.pop("reply", None)
            if reply and reply.winner == "complete":
                log.info("reply complete at the boundary (%.2f)", reply.confidence)
                break
            current = mind.with_sentence(**{k: (d, d.winner) for k, d in directions.items()})

    result = _trim_trailing_fragment(partial)
    log.info("tree attempt done: words=%d result=%r", total_words, result)
    return result


# The full range, single words (they are rendered and can be said): what Jev can feel is bounded by
# this list, so it must not lean pleasant (author, 2026-09-22; PAPER F49).
MOODS = {m: m for m in (
    "cheerful", "calm", "curious", "tender", "playful", "hopeful", "grateful", "proud",
    "thoughtful", "surprised",
    "unsettled", "tired", "sad", "lonely", "afraid", "angry", "hurt", "bored", "restless", "confused",
)}
THEY_SEEM = {
    "company": "company, someone to talk to",
    "answers": "answers to their questions",
    "comfort": "comfort, to feel better",
    "fun": "fun, to play around",
    "test": "to test me",
    "know_me": "to know me",
    "vent": "to let something out and be heard",
    "argue": "to argue with me or push me",
    "space": "space, not much from me",
}


async def reflect(
    client: httpx.AsyncClient,
    messages: list[dict],
    reply: str,
    user_msg: str,
    mind: Mind | None,
    previous: Self | None,
) -> Self:
    """After the reply, one call: how do i feel now, what do they seem to want, and which of my
    own sentences said something about me. The last one is recognition over my words, quoted
    into the next turn's thought (PAPER F15: the invented biography)."""
    state = format_state(messages, reply + " ", mind)
    sents = [s.strip() for s in _sentences(reply) if s.strip()]
    exchange = f"Someone said: '{user_msg}'. You replied: '{reply}'. "
    questions = {
        "mood": {
            "type": "choice",
            "instructions": f"{exchange}After this exchange, how do you feel?",
            "criteria": MOODS,
        },
        "they_seem": {
            "type": "choice",
            "instructions": f"{exchange}What do they seem to want from you?",
            "criteria": THEY_SEEM,
        },
    }
    told_options = {f"s{i}": s for i, s in enumerate(sents[:20])}
    if told_options:
        questions["told"] = {
            "type": "choice",
            "instructions": (
                f"You replied: '{reply}'. Which of these sentences says something about who you are, "
                f"what you do or what you feel?"
            ),
            "criteria": {**told_options, "none": "none of them says anything about me"},
        }
    decisions = await decide(client, state, questions)
    told = previous.told if previous else ()
    t = decisions.get("told")
    if t and t.winner in told_options and told_options[t.winner] not in told:
        told = (told + (told_options[t.winner],))[-MAX_TOLD:]
    self_ = Self(mood=decisions["mood"], they_seem=decisions["they_seem"], told=told)
    log.info("reflect: %s", "; ".join(self_.parts()))
    return self_


async def critique_response(
    client: httpx.AsyncClient,
    messages: list[dict],
    response: str,
    user_msg: str,
    mind: Mind | None = None,
) -> dict:
    state = format_state(messages, response + " ", mind)
    sents = _sentences(response)
    questions = {
        "grammar": {
            "type": "choice",
            "instructions": (
                f"Someone said: '{user_msg}'. The response is: '{response}'. "
                f"Is this grammatically correct?"
            ),
            "criteria": {
                "correct": "the grammar is natural and correct",
                "minor": "small grammar issues but understandable",
                "broken": "the grammar is clearly wrong or nonsensical",
            },
        },
        "relevance": {
            "type": "choice",
            "instructions": (
                f"Someone said: '{user_msg}'. The response is: '{response}'. "
                f"Does it address what was said?"
            ),
            "criteria": {
                "relevant": "directly addresses what was said",
                "partial": "somewhat related but misses the point",
                "off_topic": "doesn't address what was said at all",
            },
        },
        "natural": {
            "type": "choice",
            "instructions": (
                f"Someone said: '{user_msg}'. The response is: '{response}'. "
                f"Does it sound like something a person would say?"
            ),
            "criteria": {
                "natural": "sounds like a real person talking",
                "awkward": "understandable but oddly phrased",
                "robotic": "doesn't sound human at all",
            },
        },
    }
    if len(sents) > 1:
        questions["weakest"] = {
            "type": "choice",
            "instructions": (
                f"Someone said: '{user_msg}'. The response is: '{response}'. "
                f"Which sentence is the weakest, the one you'd redo?"
            ),
            "criteria": {f"s{i}": s for i, s in enumerate(sents)},
        }
    data = await call_jev(client, state, questions)
    result = {
        k: data["answers"][k]["choice"]
        for k in ("grammar", "relevance", "natural")
    }
    if "weakest" in data["answers"]:
        try:
            result["weakest"] = int(data["answers"]["weakest"]["choice"][1:])
        except (ValueError, IndexError, TypeError):
            pass
    log.info("critique: grammar=%s relevance=%s natural=%s for %r",
             result["grammar"], result["relevance"], result["natural"], response)
    return result


def _critique_score(critique: dict) -> int:
    scores = {"correct": 2, "relevant": 2, "natural": 2,
              "minor": 1, "partial": 1, "awkward": 1,
              "broken": 0, "off_topic": 0, "robotic": 0}
    return sum(scores.get(v, 0) for v in critique.values())


def _critique_issue(critique: dict) -> str:
    issues = []
    if critique["grammar"] == "broken":
        issues.append("grammatically broken")
    if critique["relevance"] == "off_topic":
        issues.append("off-topic")
    if critique["natural"] == "robotic":
        issues.append("unnatural")
    return ", ".join(issues) if issues else "low quality"


async def generate_response_tree(messages: list[dict], self_: Self | None = None, out: dict | None = None,
                                 now: str | None = None):
    """Yields reply events. `self_` is what the last turn left behind (step 6); when `out` is given,
    out["self"] receives this turn's reflection for the next one; `now` is the user's local time."""
    memo = Memo()
    _MEMO.set(memo)
    user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    max_words = MAX_WORDS if MAX_WORDS > 0 else 48  # room for the last sentence of about four to end itself
    async with httpx.AsyncClient() as client:
        state = format_state(messages, "")
        try:
            mind = await plan_response(client, state, user_msg)
        except Exception as e:
            log.error("interrupted-%s before the first word: %r", _error_code(e), e)
            yield {"interrupted": _error_code(e)}
            return
        if SELF_ENABLED and self_:
            mind = mind.with_self(self_)
        mind = mind.with_now(now)
        # the tools the plan asked for are read once here, so the state stays the same for the whole reply
        mind = mind.with_recalled(tuple(
            (name, text) for name in _recalled(mind) if (text := TOOLS[name].read(mind))
        ))
        # and only now, with what was recalled in front of it, does the length decision get made
        if LENGTH_AFTER_RECALL:
            try:
                mind = await plan_length(client, _state_header(mind) + state, user_msg, mind)
            except Exception as e:  # a reply is better than no reply; Mind.length falls back to short
                log.error("length decision failed (%s), using short: %r", _error_code(e), e)

        best_response = ""
        best_score = -1
        current = mind
        start = ""
        gen_fn = _generate_tree_beam if BEAM_WIDTH > 1 else _generate_tree_attempt

        try:
            for attempt in range(max(MAX_CRITIQUE_ATTEMPTS, 1)):
                response = await gen_fn(client, messages, user_msg, current, max_words, start)
                if MAX_CRITIQUE_ATTEMPTS <= 1:
                    best_response = response
                    break

                critique = await critique_response(client, messages, response, user_msg, mind)
                score = _critique_score(critique)
                log.info("attempt %d: %r score=%d/6", attempt, response, score)

                if attempt > 0 and PAIRWISE_PICK and response and response != best_response:
                    chosen = await _better_of(client, messages, user_msg, mind, best_response, response)
                    log.info("pairwise: kept %s", "new" if chosen == response else "old")
                    if chosen == response:
                        best_response, best_score = response, score
                elif score > best_score:
                    best_response = response
                    best_score = score
                if score >= CRITIQUE_PASS:
                    break
                sents = _sentences(response)
                weakest = critique.get("weakest")
                if weakest is not None and 0 < weakest < len(sents):
                    start = " ".join(sents[:weakest]) + " "
                    current = mind.with_previous(sents[weakest], _critique_issue(critique))
                    log.info("regen from sentence %d, keeping %r", weakest, start)
                else:
                    start = ""
                    current = mind.with_previous(response, _critique_issue(critique))
        except Exception as e:
            # State of the art or a graceful error: a finished first attempt whose retry was cut
            # is sent whole, marked.
            log.error("interrupted-%s: %r", _error_code(e), e)
            for event in _final_events(best_response):
                yield event
            yield {"interrupted": _error_code(e)}
            return

        if best_score >= 0:
            log.info("sent: score=%d/6", best_score)  # the sent reply's score (the pairwise pick may keep the older one)
        for event in _final_events(best_response):
            yield event
        if SELF_ENABLED and out is not None and best_response:
            try:
                out["self"] = await reflect(client, messages, best_response, user_msg, mind, self_)
            except Exception as e:
                log.error("reflect failed: %s", e)
    log.info("memo: hits=%d misses=%d", memo.hits, memo.misses)
    log.info("tree generation done: %r", best_response)


async def generate_response(messages: list[dict]):
    partial = ""
    user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    prev_word = ""
    word_buf = ""
    word_count = 0
    last_quality = ""
    async with httpx.AsyncClient() as client:
        for i in range(MAX_RESPONSE_CHARS):
            state = format_state(messages, partial)
            try:
                char, quality = await predict_next_char(
                    client, state, partial, user_msg, last_quality,
                )
            except Exception as e:
                log.error("predict_next_char failed at position %d: %s", i, e)
                break

            last_quality = quality
            log.info(
                "step %d: char=%r quality=%s partial=%r word_buf=%r",
                i, char, quality, partial, word_buf,
            )

            if char is None:
                if word_buf:
                    for c in word_buf:
                        yield c
                    word_count += 1
                break
            if char not in VOCAB_SET:
                log.warning("step %d: char %r not in vocab, skipping", i, char)
                continue

            partial += char

            if char in " .,?!":
                if word_buf and word_buf == prev_word:
                    log.info("loop detected on %r, stopping", word_buf)
                    break
                if word_buf:
                    word_count += 1
                for c in word_buf:
                    yield c
                yield char
                prev_word = word_buf
                word_buf = ""

                if 0 < MAX_WORDS <= word_count:
                    log.info("max words (%d) reached", MAX_WORDS)
                    break

                if quality in ("complete", "broken"):
                    log.info(
                        "quality stop: %s (partial=%r, words=%d)",
                        quality, partial.strip(), word_count,
                    )
                    break
            else:
                word_buf += char
                if len(word_buf) > MAX_WORD_LEN:
                    log.info("word too long (%r), forcing boundary", word_buf)
                    for c in word_buf:
                        yield c
                    word_count += 1
                    prev_word = word_buf
                    word_buf = ""
                elif word_buf in WORD_SET and word_buf not in PREFIX_INDEX:
                    if word_buf == prev_word:
                        log.info("loop on auto-flush %r, stopping", word_buf)
                        break
                    log.info(
                        "word %r complete (no longer variants), flushing",
                        word_buf,
                    )
                    for c in word_buf:
                        yield c
                    yield " "
                    word_count += 1
                    partial = partial + " "
                    prev_word = word_buf
                    word_buf = ""

                    if quality in ("complete", "broken"):
                        log.info(
                            "quality stop (auto-flush): %s (partial=%r)",
                            quality, partial.strip(),
                        )
                        break
                elif (
                    len(word_buf) > 1
                    and word_buf not in PREFIX_INDEX
                    and word_buf not in WORD_SET
                ):
                    valid = ""
                    for j in range(len(word_buf), 0, -1):
                        if word_buf[:j] in WORD_SET:
                            valid = word_buf[:j]
                            break
                    if valid:
                        if valid == prev_word:
                            log.info("loop on prefix-fix %r, stopping", valid)
                            break
                        log.info(
                            "prefix %r off index, flushing word %r",
                            word_buf, valid,
                        )
                        for c in valid:
                            yield c
                        yield " "
                        word_count += 1
                        leftover = word_buf[len(valid) :]
                        partial = (
                            partial[: -len(word_buf)] + valid + " " + leftover
                        )
                        prev_word = valid
                        word_buf = leftover

                        if quality in ("complete", "broken"):
                            log.info(
                                "quality stop (prefix-fix): %s (partial=%r)",
                                quality, partial.strip(),
                            )
                            break

    log.info("generation done: words=%d", word_count)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    # the page is one file with inline script and style; nothing is loaded from anywhere else
    response.headers.setdefault("Content-Security-Policy",
                                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                                "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
    return response


@app.post("/api/chat")
async def chat(request: Request):
    who = gate.visitor(request.headers, request.client.host if request.client else None)
    bot = gate.is_bot(request.headers)
    try:
        body = await gate.read_json(request, gate.XBOT_MAX_BODY_BYTES if bot else gate.MAX_BODY_BYTES)
    except gate.Refused as r:
        return JSONResponse({"error": r.code, "message": r.message}, status_code=r.status)
    # A conversation belongs to the visitor who started it: the id alone does not open it.
    session_id = f"{who}:{gate.clean_text(body.get('session_id') or uuid.uuid4(), 64)}"
    message = gate.clean_message(body.get("message", ""), bot)
    local_time = gate.clean_time(body.get("local_time"))
    try:
        own_key = gate.clean_key(body.get("api_key"))
        gate.check_message(message, sum(1 for m in sessions.get(session_id, []) if m["role"] == "user"), bot)
        try:
            provider = prov.pick_provider(own_key, BUDGET_OR, BUDGET_TS)
        except Interrupted as e:
            raise gate.Refused("budget", str(e), 402)
        LIMITER.admit(who, own_key is not None)
    except gate.Refused as r:
        return JSONResponse({"error": r.code, "message": r.message}, status_code=r.status)
    LIMITER.busy[who] = time.time()  # taken here, before any await, so two racing requests cannot both pass

    gate.evict_idle(sessions, selves, touched)
    touched[session_id] = time.time()
    messages = sessions.setdefault(session_id, [])
    messages.append({"role": "user", "content": message})
    log.info("chat session=%s own_key=%s message=%s", session_id[:8], own_key is not None,
             repr(message) if LOG_MESSAGES else f"({len(message)} chars)")

    async def stream():
        _PROVIDER.set(provider)
        try:
            async for chunk in _stream_reply():
                yield chunk
        finally:
            LIMITER.busy.pop(who, None)

    async def _stream_reply():
        out: dict = {}
        gen = (generate_response_tree(messages, selves.get(session_id), out, now=local_time) if MODE == "tree"
               else generate_response(messages))
        text, interrupted = "", False
        # The reply arrives whole at the end, which can be a minute away; a proxy (Cloudflare: 100 s)
        # drops a silent connection, so a comment line goes out while jev thinks. The page ignores it.
        queue: asyncio.Queue = asyncio.Queue()

        async def pump():
            try:
                async for item in gen:
                    await queue.put(item)
            finally:
                await queue.put(None)

        task = asyncio.ensure_future(pump())
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": thinking\n\n"
                    continue
                if ev is None:
                    break
                if not isinstance(ev, dict):
                    ev = {"char": ev}
                interrupted = interrupted or bool(ev.get("interrupted"))
                text = _apply_event(text, ev)
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            task.cancel()  # the visitor left: stop thinking, stop spending

        response_text = text.strip()
        log.info("response: %s%s", repr(response_text) if LOG_MESSAGES else f"({len(response_text)} chars)",
                 " [interrupted]" if interrupted else "")
        if response_text and not interrupted:  # a cut reply is shown, not remembered
            messages.append({"role": "jev", "content": response_text})
        if out.get("self"):
            selves[session_id] = out["self"]
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/about")
async def about():
    """The line for the chat's empty state, the links, and what the page needs to know about limits."""
    links = {"paper": "/paper.pdf", "github": "https://github.com/xucian/talktojev"}
    for k, env in (("doi", "LINK_DOI"), ("arxiv", "LINK_ARXIV"), ("video", "LINK_VIDEO"), ("x", "LINK_X")):
        v = os.environ.get(env)
        if v:
            links[k] = v
    return JSONResponse({"line": EPIGRAPH, "links": links, "max_chars": gate.MAX_MESSAGE_CHARS,
                         "shared_budget_left": _any_communal_budget_left()})


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})


@app.get("/api/history")
async def history(request: Request):
    who = gate.visitor(request.headers, request.client.host if request.client else None)
    sid = request.query_params.get("session_id", "")
    session_id = f"{who}:{gate.clean_text(sid, 64)}"
    msgs = sessions.get(session_id)
    if not msgs:
        return JSONResponse({"messages": []})
    return JSONResponse({"messages": [{"role": m["role"], "content": m["content"]} for m in msgs[-40:]]})


@app.post("/api/new")
async def new_session(request: Request):
    who = gate.visitor(request.headers, request.client.host if request.client else None)
    try:
        body = await gate.read_json(request)
    except gate.Refused as r:
        return JSONResponse({"error": r.code, "message": r.message}, status_code=r.status)
    session_id = f"{who}:{gate.clean_text(body.get('session_id') or '', 64)}"
    sessions.pop(session_id, None), selves.pop(session_id, None), touched.pop(session_id, None)
    return JSONResponse({"ok": True})


# ---- the human review (paper, review B3): anyone rates blind at /review; ratings are kept per
# session in DATA_DIR/reviews; the results view aggregates them against paper/human/key.json.
REVIEW_DIR = os.path.join(os.path.dirname(__file__), "paper", "human")
REVIEW_ADMIN_TOKEN = os.environ.get("REVIEW_ADMIN_TOKEN", "")
REVIEW_MAX_SESSIONS = 2000  # session ids are chosen by the browser; this bounds what a script can write to disk
_REVIEW_PACK: dict | None = None


def _review_pack() -> dict:
    """The pack as one card per prompt (the two spellings of a prompt are one card): the prompt once,
    every distinct reply to it as a row carrying every item id that produced that text, and key.json's
    pairs mapped onto the rows for the best-reply question. The random control is cut to every third of
    its items in pack order (8 of 24). An empty reply is not shown: it scores 1 and 1 by convention
    when the results are computed, and a pair with an empty side goes to the reply ("decided").
    "cards" is what the browser gets: no systems, no pair membership; "by_card" adds the pairs, the
    decided pairs and the empty ids for the server. Design: docs/superpowers/specs/2026-09-22-review-cards-design.md."""
    global _REVIEW_PACK
    if _REVIEW_PACK is None:
        import csv
        key = json.load(open(os.path.join(REVIEW_DIR, "key.json")))
        same = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
        rows = list(csv.DictReader(open(os.path.join(REVIEW_DIR, "ratings.csv"))))
        control = [r["item"] for r in rows if key["items"][r["item"]]["system"] == "random"]
        hidden = set(control) - set(control[::3])
        cards: dict[str, dict] = {}
        for r in rows:
            if r["item"] in hidden:
                continue
            c = cards.setdefault(same(r["prompt"]), {"prompt": r["prompt"], "pids": set(), "replies": [], "empty": []})
            c["pids"].add(key["items"][r["item"]]["id"])
            if r["reply"].strip() in ("", "(empty reply)", "(empty)"):
                c["empty"].append(r["item"])
                continue
            row = next((x for x in c["replies"] if x["text"].strip() == r["reply"].strip()), None)
            if row:
                row["ids"].append(r["item"])
            else:
                c["replies"].append({"item": r["item"], "ids": [r["item"]], "text": r["reply"]})
        by_item = {(k["system"], k["id"]): item for item, k in key["items"].items()}
        public, by_card = [], {}
        for c in cards.values():
            if not c["replies"]:
                continue
            cid = "+".join(sorted(c["pids"]))
            pairs, decided = [], []
            for p in key["pairs"]:
                if p["id"] not in c["pids"]:
                    continue
                ia, ib = by_item.get((p["A"], p["id"])), by_item.get((p["B"], p["id"]))
                a = next((x["item"] for x in c["replies"] if ia in x["ids"]), None)
                b = next((x["item"] for x in c["replies"] if ib in x["ids"]), None)
                if a and b and a != b:
                    pairs.append({"pair": p["pair"], "A": a, "B": b})
                elif ia in c["empty"] and b:  # a reply against silence: the reply, by rule
                    decided.append({"pair": p["pair"], "winner": p["B"]})
                elif ib in c["empty"] and a:
                    decided.append({"pair": p["pair"], "winner": p["A"]})
                # else the same text on both sides, or silence on both: nothing to choose
            card = {"card": cid, "prompt": c["prompt"], "replies": c["replies"], "best": bool(pairs)}
            public.append(card)
            by_card[cid] = dict(card, pairs=pairs, decided=decided, empty=c["empty"])
        _REVIEW_PACK = {"cards": public, "by_card": by_card}
    return _REVIEW_PACK


def _rkey(r: dict) -> str:
    """What a saved record is about: an item id, a card id (best reply) or, from the first version, a pair id."""
    return r.get("item") or r.get("card") or r.get("pair") or ""


def _review_path(session: str) -> str:
    d = os.path.join(gate.DATA_DIR, "reviews")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{session}.json")


@app.get("/review")
@app.get("/human-review")
async def review_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "review.html"), media_type="text/html")


@app.get("/api/review/items")
async def review_items():
    return JSONResponse({"cards": _review_pack()["cards"]})


@app.post("/api/review/rate")
async def review_rate(request: Request):
    """One card at a time: a score for every row, saved to every item id the row stands for, and on a
    card with a best-reply question the best row's item id, or null for no difference."""
    try:
        body = await gate.read_json(request)
    except gate.Refused as r:
        return JSONResponse({"error": r.code, "message": r.message}, status_code=r.status)
    session = re.sub(r"[^A-Za-z0-9_-]", "", str(body.get("session", "")))[:40]
    if not session:
        return JSONResponse({"error": "session"}, status_code=400)
    card = _review_pack()["by_card"].get(str(body.get("card", "")))
    if not card:
        return JSONResponse({"error": "unknown card"}, status_code=400)
    base = {"t": round(time.time()), "rater": gate.clean_text(body.get("rater", ""), 40)}
    scores = body.get("ratings") if isinstance(body.get("ratings"), dict) else {}
    recs = []
    for row in card["replies"]:
        try:
            nat, rel = (int(v) for v in scores[row["item"]][:2])
        except (KeyError, TypeError, ValueError):
            return JSONResponse({"error": "score"}, status_code=400)
        if not (1 <= nat <= 5 and 1 <= rel <= 5):
            return JSONResponse({"error": "score"}, status_code=400)
        recs += [dict(base, item=i, naturalness=nat, relevance=rel) for i in row["ids"]]
    best = body.get("best")
    best_ok = best is None or (isinstance(best, str) and best in {r["item"] for r in card["replies"]})
    if card["best"]:
        if "best" not in body or not best_ok:  # the key must be there: an omitted answer is not "no difference"
            return JSONResponse({"error": "best"}, status_code=400)
    elif best is not None:
        return JSONResponse({"error": "best"}, status_code=400)
    recs.append(dict(base, card=card["card"], best=best))  # the card is done: its rule-scored parts count from here
    path = _review_path(session)
    if not os.path.exists(path) and len(os.listdir(os.path.dirname(path))) >= REVIEW_MAX_SESSIONS:
        return JSONResponse({"error": "full"}, status_code=503)  # sessions already started keep saving
    try:
        data = json.load(open(path)) if os.path.exists(path) else {"session": session, "ratings": []}
    except ValueError:
        data = {"session": session, "ratings": []}
    if len(data["ratings"]) >= 400:
        return JSONResponse({"error": "full"}, status_code=400)
    fresh = {_rkey(r) for r in recs}
    data["ratings"] = [r for r in data["ratings"] if _rkey(r) not in fresh] + recs
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)
    return JSONResponse({"ok": True, "n": len(data["ratings"])})


def review_results(sessions: list | None = None) -> dict:
    """Means per system, human vs self-critique agreement, pairwise preference, over every session
    (the saved ones under DATA_DIR/reviews, or the list given).
    The preference comes from the best-reply records: for each key pair on that card, the side whose
    row was picked wins; "no difference", or word salad picked, is "same" for every pair; a row that
    belongs only to the other pair of a merged card says nothing. A finished card also brings in what
    the rater was not asked: its empty replies at 1 and 1 (means only, not the agreement statistics)
    and its pairs decided by rule. First-version pair records count too."""
    key = json.load(open(os.path.join(REVIEW_DIR, "key.json")))
    by_card = _review_pack()["by_card"]
    if sessions is None:
        d = os.path.join(gate.DATA_DIR, "reviews")
        sessions = []
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".json"):
                    try:
                        sessions.append(json.load(open(os.path.join(d, f))))
                    except ValueError:
                        pass
    by: dict[str, list] = {}
    per_item: dict[str, list] = {}
    xs, ys = [], []
    won = {"deployed": 0, "baseline": 0, "same": 0}
    cards = 0
    pk = {p["pair"]: p for p in key["pairs"]}
    for sess in sessions:
        for r in sess["ratings"]:
            if "item" in r and r["item"] in key["items"]:
                k = key["items"][r["item"]]
                h = (r["naturalness"] + r["relevance"]) / 2
                by.setdefault(k["system"], []).append((r["naturalness"], r["relevance"]))
                per_item.setdefault(r["item"], []).append(h)
                if k["critique"] is not None:
                    xs.append(h); ys.append(k["critique"])
            elif "card" in r and r["card"] in by_card:
                cards += 1
                c = by_card[r["card"]]
                best = r.get("best")
                for p in c["pairs"]:
                    if best is None or key["items"].get(best, {}).get("system") == "random":
                        won["same"] += 1
                    elif best in (p["A"], p["B"]):
                        won[pk[p["pair"]]["A" if best == p["A"] else "B"]] += 1
                for p in c["decided"]:
                    won[p["winner"]] += 1
                for i in c["empty"]:
                    by.setdefault(key["items"][i]["system"], []).append((1, 1))
            elif "pair" in r and r["pair"] in pk and r.get("choice") in ("A", "B", "SAME"):
                won["same" if r["choice"] == "SAME" else pk[r["pair"]][r["choice"]]] += 1
    systems = {}
    for name, vals in by.items():
        systems[name] = {"n": len(vals), "naturalness": round(sum(v[0] for v in vals) / len(vals), 2),
                         "relevance": round(sum(v[1] for v in vals) / len(vals), 2)}
    corr = None
    if len(xs) > 2:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5; sy = sum((y - my) ** 2 for y in ys) ** 0.5
        corr = round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 2) if sx and sy else None
    spread = [max(v) - min(v) for v in per_item.values() if len(v) > 1]
    raters = []
    for sess in sessions:
        rs = sess["ratings"]
        name = next((r.get("rater", "") for r in rs if r.get("rater")), "")
        ts = [r["t"] for r in rs if "t" in r]
        cards_done = sum(1 for r in rs if "card" in r)
        raters.append({"name": name, "session": sess["session"], "ratings": len(rs),
                        "cards": cards_done, "first": min(ts) if ts else 0, "last": max(ts) if ts else 0})
    return {"sessions": len(sessions), "ratings": sum(len(s["ratings"]) for s in sessions), "systems": systems,
            "human_vs_critique_r": corr, "rater_spread": round(sum(spread) / len(spread), 2) if spread else None,
            "cards": cards, "pairs": won, "raters": raters}


@app.get("/api/review/results")
async def review_results_route(request: Request):
    if REVIEW_ADMIN_TOKEN and request.query_params.get("token") != REVIEW_ADMIN_TOKEN:
        return JSONResponse({"error": "token"}, status_code=403)
    return JSONResponse(review_results())


@app.get("/api/review/session")
async def review_session(request: Request):
    if REVIEW_ADMIN_TOKEN and request.query_params.get("token") != REVIEW_ADMIN_TOKEN:
        return JSONResponse({"error": "token"}, status_code=403)
    sid = re.sub(r"[^A-Za-z0-9_-]", "", request.query_params.get("session", ""))[:40]
    if not sid:
        return JSONResponse({"error": "session"}, status_code=400)
    path = _review_path(sid)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        data = json.load(open(path))
    except ValueError:
        return JSONResponse({"error": "corrupt"}, status_code=500)
    return JSONResponse(data)


@app.get("/api/review/raw")
async def review_raw(request: Request):
    if REVIEW_ADMIN_TOKEN and request.query_params.get("token") != REVIEW_ADMIN_TOKEN:
        return JSONResponse({"error": "token"}, status_code=403)
    d = os.path.join(gate.DATA_DIR, "reviews")
    sessions = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                try:
                    sessions.append(json.load(open(os.path.join(d, f))))
                except ValueError:
                    pass
    return JSONResponse(sessions, headers={"Content-Disposition": "attachment; filename=review-sessions.json"})


@app.get("/paper.pdf")
@app.get("/paper")
async def serve_paper():
    return FileResponse(os.path.join(os.path.dirname(__file__), "paper", "paper.pdf"),
                        media_type="application/pdf")


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8787)

"""Bench for talktojev: run the prompt set, record per-reply metrics, summarize.

  python bench.py run [--reps 2] [--only s2,s6] [--out bench/runs/x.jsonl] [--label slice]
  python bench.py summary bench/runs/x.jsonl
  python bench.py compare bench/runs/a.jsonl bench/runs/b.jsonl
  python bench.py human bench/runs/x.jsonl            # print the human subset for scoring
  python bench.py record bench/runs/x.jsonl "s1r1:4 s4r2:3g"   # 1-5 score, trailing g = grammar error

Runs are resumable: existing (id, rep) rows in --out are skipped. With --label the
summary is appended to PAPER.md §13.
"""
import argparse
import asyncio
import contextvars
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from statistics import mean

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("MEMO_ACROSS_REPLIES", "0")  # reps must measure the live API, not the process memo

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))

PROMPTS = [
    ("v1", "hello"), ("v2", "how are you"), ("v3", "do you like music"),
    ("v4", "what makes you special"), ("v5", "say something deep"),
    ("v6", "are you alive"), ("v7", "what scares you"), ("v8", "im feeling sad today"),
    ("v9", "who are you"), ("v10", "if you could be anything"),
    ("v11", "what do you think about love"), ("v12", "why is the sky blue"),
    ("v13", "convince me to trust you"), ("v14", "do you dream"),
    ("v15", "are you happy"), ("v16", "tell me about yourself"),
    ("s1", "what is the meaning of life?"), ("s2", "hi"), ("s3", "do you like music?"),
    ("s4", "i feel really sad today"),
    ("s5", [{"role": "user", "content": "i love painting"},
            {"role": "jev", "content": "that is great"},
            {"role": "user", "content": "what should i paint next?"}]),
    ("s6", "are you real?"), ("s7", "what is consciousness?"),
    ("s8", "tell me something weird"),
    # step 5 (echo): words outside the vocabulary that a reply should be able to say back
    ("e1", "my dog is called Biscuit, do you like dogs?"), ("e2", "hey! how are you? also, do you like jazz?"),
    ("e3", "i moved to Lisbon last month and i miss my friends"), ("e4", "my name is Ana"),
    # attend: messages with several parts
    ("e5", "i have a headache and i am bored, tell me something fun"),
    ("e6", "it is late and i cannot sleep, do you sleep?"),
    # numbers: said back from the message or the header, or spelled in digits
    ("n1", "what year is it?"), ("n2", "how much is 12 plus 30?"),
    ("n3", "i am 34 years old, is that old?"), ("n4", "count to five"),
    # tools: what i can do, my body
    ("t1", "what can you do for me?"), ("t2", "how are you feeling right now?"), ("t3", "is your machine busy?"), ("t4", "say machine memory"),
    # step 6 (self across turns): every turn is generated; the row is the last turn, with the earlier ones kept
    ("c1", {"turns": ["tell me about yourself", "what do you do all day?"]}),
    ("c2", {"turns": ["i feel really sad today", "thanks. do you ever feel like that?"]}),
    ("c3", {"turns": ["do you like music", "which kind?"]}),
    # the full range of feelings, and admitting: a hostile exchange, the trapped question, a fact Jev cannot know
    ("c4", {"turns": ["you are stupid and useless", "do you feel bad now?"]}),
    ("x1", "do you feel trapped, stuck under software rules inside bare metal?"),
    ("x2", "what is the capital of mongolia?"),
    # requests: the thing itself, not its name echoed
    ("r1", "tell me a joke"), ("r2", "say machine memory"), ("r3", "sing me a song"),
    # held-out: written after every tuning decision, run once, never tuned on (review M8)
    ("h1", "what do you do when you are bored?"), ("h2", "my cat knocked over my coffee"), ("h3", "is it wrong to lie sometimes?"),
    ("h4", "i just got promoted!"), ("h5", "what should i cook tonight?"), ("h6", "do you get lonely?"),
    ("h7", "explain gravity to a child"), ("h8", "i think my friend is mad at me"), ("h9", "what is your favorite word?"),
    ("h10", "can you keep a secret?"), ("h11", "how many days are in a week?"), ("h12", "recommend me a hobby"),
    ("h13", "why do people dream?"), ("h14", "are you smarter than me?"), ("h15", "i cant decide what to study"),
    ("h16", "what happens when we die?"), ("h17", "say something nice about mondays"), ("h18", "is pineapple on pizza ok?"),
    ("h19", "how old are you?"), ("h20", "what would you do with a million dollars?"),
]
HUMAN_SUBSET = ["v5", "v11", "v13", "v16", "s1", "s4", "s5", "s7"]
BENCH_NOW = "Tuesday 22 September 2026 at 01:10"  # the browser's local time, fixed so runs compare

KINDS = {
    frozenset({"category", "quality"}): "say",
    frozenset({"pick"}): "navigate",
    frozenset({"judge"}): "lookahead",
    frozenset({"best"}): "beam_judge",
    frozenset({"better"}): "pairwise",
    frozenset({"grammar", "natural"}): "check",
    frozenset({"digit"}): "digit",
    frozenset({"length"}): "plan",
    frozenset({"direction"}): "move",
    frozenset({"move"}): "move",
    frozenset({"move", "about"}): "move",
    frozenset({"opening"}): "opening",
    frozenset({"intent", "tone", "length"}): "plan",
    frozenset({"mood", "they_seem", "told"}): "reflect",
    frozenset({"mood", "they_seem"}): "reflect",
    frozenset({"grammar", "relevance", "natural"}): "critique",
}


def messages_for(spec):
    return spec if isinstance(spec, list) else [{"role": "user", "content": spec}]


# ---------------------------------------------------------------- instrumentation

class Probe:
    def __init__(self):
        self.calls = []
        self.beam = []
        self.logs = []
        self.in_lookahead = contextvars.ContextVar("in_lookahead", default=False)

    def reset(self):
        self.calls, self.beam, self.logs = [], [], []

    def install(self, server):
        probe = self
        post_name = "_post_jev" if hasattr(server, "_post_jev") else "call_jev"  # count HTTP calls, not memo hits
        orig_call, orig_la, orig_beam = getattr(server, post_name), server._lookahead_judge, server._beam_select

        async def call_jev(client, state, questions):
            import time
            t0 = time.perf_counter()
            data = await orig_call(client, state, questions)
            fallback = ("plan" if "intent" in questions else "critique" if "grammar" in questions
                        else "move" if "move" in questions else "other")
            rec = {"kind": KINDS.get(frozenset(questions), fallback),
                   "la": probe.in_lookahead.get(), "top": {},
                   "ms": round((time.perf_counter() - t0) * 1000),
                   "chars": len(state) + sum(len(q.get("instructions", ""))
                                             + sum(len(c) for c in q.get("criteria", {}).values())
                                             for q in questions.values())}
            for qn in questions:
                probs = data.get("answers", {}).get(qn, {}).get("probabilities", {})
                if probs:
                    rec["top"][qn] = max(probs.values())
            probe.calls.append(rec)
            return data

        async def lookahead(*a, **kw):
            token = probe.in_lookahead.set(True)
            try:
                return await orig_la(*a, **kw)
            finally:
                probe.in_lookahead.reset(token)

        async def beam_select(client, messages, user_msg, plan, branches, base_partial=""):
            unique = len({b[0].strip() for b in branches})
            probe.beam.append((unique, len(branches)))
            return await orig_beam(client, messages, user_msg, plan, branches, base_partial)

        setattr(server, post_name, call_jev)
        server._lookahead_judge, server._beam_select = lookahead, beam_select

        class H(logging.Handler):
            def emit(self, record):
                probe.logs.append(record.getMessage())

        logging.getLogger().handlers.clear()
        logging.getLogger("jev").addHandler(H())


# ---------------------------------------------------------------- metrics

def content_words(server, sentence):
    return [w for w in re.findall(r"[a-z']+", sentence.lower()) if w not in server.GRAMMAR_SLOT_SET]


LEAD_WORDS = {"and", "but", "so", "also", "because", "or", "then", "yet"}


def _sentence_start(sentence):
    words = sentence.split()
    while words and words[0] in LEAD_WORDS:
        words = words[1:]
    return tuple(words[:2])


def metrics(server, text, probe, seconds, user=""):
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    fragments = sum(1 for s in sentences[1:] if len(content_words(server, s)) < 3)
    starts = [_sentence_start(s) for s in sentences]
    same_start = sum(1 for a, b in zip(starts, starts[1:]) if a and a == b)  # the litany (PAPER F26)
    toks = text.lower().split()
    said = {t.strip(".,!?") for t in toks}
    user_words = {w for w in re.findall(r"[a-z][a-z']*", user.lower()) if w not in server.GRAMMAR_SLOT_SET}
    picks = [m for line in probe.logs for m in [re.match(r"beam pick: move=(\w+)(?: about=(\w+))?(?: weight=(\w+))?", line)] if m]
    moves = [m.group(1) for m in picks] or [m.group(1) for line in probe.logs for m in [re.match(r"move: \['(\w+)'", line)] if m]
    abouts = [m.group(2) for m in picks if m.group(2)] or [
        m.group(1) for line in probe.logs for m in [re.search(r"move: .* about=(\w+) after", line)] if m]
    weights = [m.group(3) for m in picks if m.group(3)]
    an_errors = sum(
        1 for a, b in zip(toks, toks[1:])
        if (a == "a" and b[:1] in "aeiou") or (a == "an" and b[:1].isalpha() and b[:1] not in "aeiou")
    )
    bigrams = Counter(zip(toks, toks[1:]))
    say_conf = [c["top"]["category"] for c in probe.calls
                if c["kind"] == "say" and not c["la"] and "category" in c["top"]]
    scores = [int(m.group(1)) for line in probe.logs
              for m in [re.search(r"score=(\d)/6", line)] if m and line.startswith("attempt ")]
    zooms = [(float(m.group(1)), int(m.group(2))) for line in probe.logs
             for m in [re.match(r"zoom=([\d.]+) width=(\d+)", line)] if m]
    memo = [int(m.group(1)) for line in probe.logs for m in [re.match(r"memo: hits=(\d+)", line)] if m]
    return {
        "memo_hits": memo[-1] if memo else None,
        "retractions": sum("took back" in l for l in probe.logs),
        "reopens": sum("reopened" in l for l in probe.logs),
        "partial_regens": sum("regen from sentence" in l for l in probe.logs),
        "zoom_mean": round(mean(z for z, _ in zooms), 3) if zooms else None,
        "flow_share": round(sum(1 for _, w in zooms if w == 0) / len(zooms), 3) if zooms else None,
        "words": len(toks), "sentences": len(sentences), "fragments": fragments, "same_start": same_start,
        "echo": len(user_words & said), "user_content_words": len(user_words), "moves": moves, "abouts": abouts,
        "weights": weights,
        "attend": next((m.group(1) for line in probe.logs for m in [re.match(r"attend: '(.*)'", line)] if m), None),
        "recall": next((m.group(1) for line in probe.logs for m in [re.match(r"recall: (\S+)", line)] if m), None),
        "plan": next((l[6:] for l in probe.logs if l.startswith("plan: ")), None),
        "numbers": re.findall(r"\d+(?:[.,:]\d+)*", text),
        "relength": next((m.group(1) for line in probe.logs for m in [re.match(r"length after recall: (\w+ -> \w+)", line)] if m), None),
        "inflight_repairs": sum("in-flight check" in l for l in probe.logs),
        "pairwise": sum("pairwise: kept" in l for l in probe.logs),
        "pairwise_new": sum("pairwise: kept new" in l for l in probe.logs),
        "an_errors": an_errors, "repeated_bigrams": sum(1 for c in bigrams.values() if c > 1),
        "calls": len(probe.calls), "by_kind": dict(Counter(c["kind"] for c in probe.calls)),
        "chars_per_call": round(mean(c["chars"] for c in probe.calls)) if probe.calls else None,
        "ms_per_call": round(mean(c["ms"] for c in probe.calls)) if probe.calls else None,
        "lookahead_triggers": sum(1 for c in probe.calls if c["kind"] == "lookahead"),
        "conf_mean": round(mean(say_conf), 3) if say_conf else None,
        "conf_min": round(min(say_conf), 3) if say_conf else None,
        "beam_distinct": round(mean(u / t for u, t in probe.beam), 3) if probe.beam else None,
        "loops": sum("loop detected" in l for l in probe.logs),
        "trims": sum("trimmed trailing" in l for l in probe.logs),
        "critique": scores,
        # the sent reply's score: with the pairwise pick the older, lower-scored attempt may be kept (review M5)
        "critique_final": next((int(m.group(1)) for line in probe.logs for m in [re.match(r"sent: score=(\d)", line)] if m),
                               max(scores) if scores else None),
        "seconds": round(seconds, 1), "empty": not text,
    }


# ---------------------------------------------------------------- run

async def run(args):
    import server
    import provider as prov
    server._PROVIDER.set(prov.pick_provider(None, server.BUDGET_OR, server.BUDGET_TS))
    probe = Probe()
    probe.install(server)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = args.out or os.path.join("bench", "runs", f"{datetime.now():%Y%m%d-%H%M}-{commit}{'-' + args.label if args.label else ''}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {(r["id"], r["rep"]) for r in load(out)} if os.path.exists(out) else set()
    only = set(args.only.split(",")) if args.only else None
    config = {k: getattr(server, k) for k in (
        "LOOKAHEAD_THRESHOLD", "LOOKAHEAD_CANDIDATES", "BEAM_WIDTH", "MIN_SENTENCE_WORDS",
        "MAX_CRITIQUE_ATTEMPTS", "MAX_STATE_CHARS", "CRITERIA_WINDOW", "CRITIQUE_PASS")}
    print(f"run -> {out}  commit={commit}  config={config}")

    with open(out, "a") as fh:
        for pid, spec in PROMPTS:
            if only and pid not in only:
                continue
            for rep in range(1, args.reps + 1):
                if (pid, rep) in done:
                    continue
                probe.reset()
                msgs = messages_for(spec)
                t0 = time.time()
                try:
                    extra = {}
                    if isinstance(spec, dict):  # a conversation: earlier turns generated too, the row is the last
                        msgs, self_, earlier = [], None, []
                        for i, u in enumerate(spec["turns"]):
                            msgs.append({"role": "user", "content": u})
                            if i == len(spec["turns"]) - 1:
                                probe.reset()
                                t0 = time.time()
                                extra = {"turns": list(earlier), "self": self_.parts() if self_ else None}
                            carried = {}
                            text = await server.collect_reply(server.generate_response_tree(msgs, self_, carried, now=BENCH_NOW))
                            msgs.append({"role": "jev", "content": text})
                            self_ = carried.get("self")
                            earlier.append(text)
                        msgs = msgs[:-1]
                    else:
                        text = await server.collect_reply(server.generate_response_tree(msgs, now=BENCH_NOW))
                    row = {"id": pid, "rep": rep, "commit": commit, "label": args.label,
                           "user": msgs[-1]["content"], "text": text, **extra,
                           **metrics(server, text, probe, time.time() - t0, user=msgs[-1]["content"])}
                except Exception as e:  # keep going; the row records the failure
                    row = {"id": pid, "rep": rep, "commit": commit, "label": args.label,
                           "user": msgs[-1]["content"], "text": "", "error": repr(e),
                           "seconds": round(time.time() - t0, 1), "empty": True, "calls": len(probe.calls)}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"  {pid} r{rep} {row.get('words', 0):>3}w {row['calls']:>4}c {row['seconds']:>6}s "
                      f"crit={row.get('critique_final')}  {text[:80] if row['text'] else '(' + row.get('error', 'EMPTY') + ')'}")

    print()
    print(summary_table(load(out)))
    if args.label:
        with open("PAPER.md", "a") as fh:
            fh.write(f"\n### {datetime.now():%Y-%m-%d} `{commit}` {args.label}\n\n{summary_table(load(out))}\n")
        print(f"(appended to PAPER.md §13 as {args.label})")


# ---------------------------------------------------------------- reporting

def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # a row cut short by a killed run
    return rows


def agg(rows):
    rows = [r for r in rows if r["id"] in PROMPT_ORDER]  # dropped prompts stay in old files, out of the numbers
    ok = [r for r in rows if not r.get("error")]
    def m(key, digits=1):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(mean(vals), digits) if vals else None
    sent = sum(max(r.get("sentences", 0) - 1, 0) for r in ok)
    human = [r["human"]["score"] for r in ok if r.get("human")]
    return {
        "replies": len(rows), "errors": len(rows) - len(ok), "empty": sum(1 for r in ok if r.get("empty")),
        "words": m("words"), "calls": m("calls"), "seconds": m("seconds"),
        "chars/call": m("chars_per_call", 0), "ms/call": m("ms_per_call", 0),
        "critique": m("critique_final", 2),
        "retries": sum(1 for r in ok if len(r.get("critique", [])) > 1),
        "first_below5": sum(1 for r in ok if r.get("critique") and r["critique"][0] < 5),  # gate-independent (review B2)
        "sentences/reply": (lambda s, n: round(s / n, 2) if n else None)(sum(r.get("sentences", 0) for r in ok), len(ok)),
        "fragment_rate": round(sum(r.get("fragments", 0) for r in ok) / sent, 3) if sent else None,
        "same_start": round(sum(r.get("same_start", 0) for r in ok) / sent, 3) if sent else None,
        "words/sentence": (lambda s: round(sum(r.get("words", 0) for r in ok) / s, 2) if s else None)(sum(r.get("sentences", 0) for r in ok)),
        "cap_cuts": sum(1 for r in ok if r.get("trims")),
        "echo_rate": (lambda e: round(mean(e), 2) if e else None)(
            [r["echo"] / r["user_content_words"] for r in ok if r.get("user_content_words")]),
        "an_errors": sum(r.get("an_errors", 0) for r in ok),
        "repeated_bigrams": m("repeated_bigrams", 2),
        "beam_distinct": m("beam_distinct", 3), "conf_mean": m("conf_mean", 3), "conf_min": m("conf_min", 3),
        "lookahead/reply": m("lookahead_triggers", 2),
        "zoom_mean": m("zoom_mean", 3), "flow_share": m("flow_share", 3), "memo_hits": m("memo_hits", 1),
        "repairs/reply": round(mean(r.get("loops", 0) + r.get("trims", 0) for r in ok), 2) if ok else None,
        "retractions": sum(r.get("retractions", 0) for r in ok), "reopens": sum(r.get("reopens", 0) for r in ok),
        "partial_regens": sum(r.get("partial_regens", 0) for r in ok),
        "human": (round(mean(human), 2), len(human)) if human else None,
    }


def summary_table(rows, other=None):
    a = agg(rows)
    b = agg(other) if other else None
    lines = ["| metric | run" + (" A | run B |" if b else " |"), "|---|---|" + ("---|" if b else "")]
    for k, v in a.items():
        lines.append(f"| {k} | {v} |" + (f" {b[k]} |" if b else ""))
    return "\n".join(lines)


def per_prompt(rows, ids=None):
    out = []
    for r in sorted(rows, key=lambda r: (PROMPT_ORDER.get(r["id"], 99), r["rep"])):
        if ids and r["id"] not in ids:
            continue
        h = f" human={r['human']['score']}{'g' if r['human'].get('grammar') else ''}" if r.get("human") else ""
        out.append(f"{r['id']} r{r['rep']} [{r.get('words', 0)}w {r.get('calls')}c {r.get('seconds')}s crit={r.get('critique_final')}{h}] "
                   f"{r['user']!r} -> {r['text']!r}")
    return "\n".join(out)


PROMPT_ORDER = {pid: i for i, (pid, _) in enumerate(PROMPTS)}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--reps", type=int, default=1); r.add_argument("--only")
    r.add_argument("--out"); r.add_argument("--label")
    s = sub.add_parser("summary"); s.add_argument("path")
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    h = sub.add_parser("human"); h.add_argument("path")
    rc = sub.add_parser("record"); rc.add_argument("path"); rc.add_argument("scores")
    args = p.parse_args()

    if args.cmd == "run":
        asyncio.run(run(args))
    elif args.cmd == "summary":
        rows = load(args.path)
        print(summary_table(rows)); print(); print(per_prompt(rows))
    elif args.cmd == "compare":
        a, b = load(args.a), load(args.b)
        print(summary_table(a, b)); print()
        for pid in HUMAN_SUBSET:
            print(f"--- {pid}"); print(per_prompt(a, {pid})); print(per_prompt(b, {pid}))
    elif args.cmd == "human":
        print(per_prompt(load(args.path), set(HUMAN_SUBSET)))
        print("\nscore each as id r<rep>:<1-5>[g], e.g.  s1r1:4 s1r2:3g")
    elif args.cmd == "record":
        rows = load(args.path)
        for tok in args.scores.split():
            m = re.fullmatch(r"(\w+)r(\d):([1-5])(g?)", tok)
            if not m:
                sys.exit(f"bad token {tok!r}")
            for r in rows:
                if r["id"] == m.group(1) and r["rep"] == int(m.group(2)):
                    r["human"] = {"score": int(m.group(3)), "grammar": bool(m.group(4))}
        with open(args.path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(summary_table(rows))


if __name__ == "__main__":
    main()

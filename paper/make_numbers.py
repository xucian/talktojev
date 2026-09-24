"""Every number in the paper, computed from bench/runs into LaTeX macros (paper/numbers.tex).
No number in paper.tex is typed by hand. Run from the repo root: python paper/make_numbers.py"""
import glob, os, sys
sys.path.insert(0, os.getcwd())
from bench import load, agg

R = "bench/runs/"
ORIG = "v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 s1 s2 s3 s4 s5 s6 s7 s8".split()
M = {}


def rows(name, ids=None):
    return [r for r in load(R + name + ".jsonl") if not r.get("error") and (ids is None or r["id"] in ids)]


def put(key, value, digits=None):
    assert key.isalpha(), key
    if isinstance(value, float):
        value = f"{value:.{digits if digits is not None else 1}f}"
    M[key] = str(value)


def cpw(rs):  # calls per generated word
    return sum(r["calls"] for r in rs) / max(sum(r["words"] for r in rs), 1)


def pct(a, b):
    return round(100 * a / b)


# ---- stages on the original 24 prompts
for tag, name in (("Base", "baseline-2b4fd54"), ("Zoom", "step3-zoom-v2"), ("Move", "step4-move"), ("Final", "final-6a97002"), ("Dep", "deployed-1528882")):
    rs = rows(name, ORIG); a = agg(rs)
    put(tag + "N", len(rs)); put(tag + "Calls", float(a["calls"]), 0); put(tag + "Words", float(a["words"]), 1)
    put(tag + "CPW", cpw(rs), 1); put(tag + "Crit", float(a["critique"]), 2); put(tag + "Secs", float(a["seconds"]), 0)
    put(tag + "Empty", sum(1 for r in load(R + name + ".jsonl") if r["id"] in ORIG and r.get("empty")))
    put(tag + "Distinct", float(a["beam_distinct"]), 2)
put("CPWDrop", pct(float(M["BaseCPW"]) - float(M["FinalCPW"]), float(M["BaseCPW"])))
put("DepCPWDrop", pct(float(M["BaseCPW"]) - float(M["DepCPW"]), float(M["BaseCPW"])))
dep_all = rows("deployed-1528882"); put("DepAllN", len(dep_all)); put("DepAllCrit", float(agg(dep_all)["critique"]), 2)
put("DepRetries", agg(rows("deployed-1528882", ORIG))["retries"]); put("FinalRetries", agg(rows("final-6a97002", ORIG))["retries"])
put("DepAn", agg(rows("deployed-1528882", ORIG))["an_errors"]); put("FinalAn", agg(rows("final-6a97002", ORIG))["an_errors"])
put("CallsDrop", pct(float(M["BaseCalls"]) - float(M["FinalCalls"]), float(M["BaseCalls"])))
allfinal = rows("final-6a97002")
put("FinalAllN", len(load(R + "final-6a97002.jsonl"))); put("FinalPrompts", len({r["id"] for r in allfinal}))
put("MsPerCall", float(agg(allfinal)["ms/call"]), 0)

# ---- criteria window (F27)
for tag, name in (("WinFull", "optim3-window-full"), ("WinEight", "optim3-window-8"), ("WinThree", "optim3-window-3")):
    a = agg(rows(name)); put(tag + "Chars", f"{int(a['chars/call']):,}".replace(",", "{,}")); put(tag + "Ms", float(a["ms/call"]), 0)
    put(tag + "Calls", float(a["calls"]), 0); put(tag + "Conf", float(a["conf_mean"]), 3); put(tag + "Crit", float(a["critique"]), 2)

# ---- the user's words as options (F28)
E = ["e1", "e2", "e3", "e4"]
off, on = agg(rows("step5-echo-off", E)), agg(rows("step5-echo-on", E))
put("EchoOffCalls", float(off["calls"]), 0); put("EchoOnCalls", float(on["calls"]), 0)
put("EchoOffConf", float(off["conf_mean"]), 2); put("EchoOnConf", float(on["conf_mean"]), 2)
put("EchoOffRate", float(off["echo_rate"]), 2); put("EchoOnRate", float(on["echo_rate"]), 2)

# ---- sticky winner vs spread distribution (F29, F30): repeated sentence starts, same 8 prompts
EIGHT = E + ["v4", "v11", "v12", "s7"]
for tag, name in (("StartBefore", "step5-echo-on"), ("StartWinner", "step4c-about"), ("StartSpread", "step4d-beam-about")):
    put(tag, float(agg(rows(name, EIGHT))["same_start"]), 2)

# ---- grain of the ending (F37): replies cut by the word cap, elaborate prompts
THREE = ["v11", "s1", "s7"]
f3, a4 = rows("final-6a97002", THREE), rows("A4-reply", THREE)
put("CapBefore", sum(1 for r in f3 if r.get("trims"))); put("CapBeforeN", len(f3))
put("CapAfter", sum(1 for r in a4 if r.get("trims"))); put("CapAfterN", len(a4))
put("EndCallsBefore", float(agg(f3)["calls"]), 0); put("EndCallsAfter", float(agg(a4)["calls"]), 0)

# ---- retries (F50): every recorded run
ret = [r for f in glob.glob(R + "*.jsonl") for r in load(f) if len(r.get("critique") or []) > 1 and r.get("calls")]
up = [r for r in ret if max(r["critique"][1:]) > r["critique"][0]]
same = [r for r in ret if max(r["critique"][1:]) == r["critique"][0]]
put("RetryN", len(ret)); put("RetryUp", pct(len(up), len(ret))); put("RetrySame", pct(len(same), len(ret)))
put("RetryDown", 100 - pct(len(up), len(ret)) - pct(len(same), len(ret)))
four = [r for r in ret if r["critique"][0] == 4]; low = [r for r in ret if r["critique"][0] <= 3]
put("RetryFromFourShare", pct(len(four), len(ret)))
put("RetryFromFourUp", pct(sum(1 for r in four if r in up), len(four)))
put("RetryFromLowUp", pct(sum(1 for r in low if r in up), len(low)))

# ---- review fixes (2026-09-22): gate-independent retry count and sentences per reply, per stage
for tag, name in (("Base", "baseline-2b4fd54"), ("Zoom", "step3-zoom-v2"), ("Move", "step4-move"), ("Final", "final-6a97002"), ("Dep", "deployed-1528882")):
    a = agg(rows(name, ORIG)); put(tag + "Below", a["first_below5"]); put(tag + "Sent", float(a["sentences/reply"]), 1)

# ---- the random-choice control (review, objection 1)
if os.path.exists(R + "control-random.jsonl"):
    c = rows("control-random", ORIG)
    put("CtrlN", len(c)); put("CtrlCrit", sum(r["critique_final"] for r in c) / len(c), 2)
    put("CtrlEmpty", sum(1 for r in c if not r["text"])); put("CtrlWords", sum(r["words"] for r in c) / len(c), 1)

# ---- F1 re-run: accuracy vs option count, 20 prompts (review B5)
if os.path.exists(R + "f1-options.jsonl"):
    import json as _json
    f1 = [_json.loads(l) for l in open(R + "f1-options.jsonl")]
    sizes = sorted({r["size"] for r in f1})
    put("FoneN", len({r["prompt"] for r in f1})); put("FoneMaxSize", max(sizes)); put("FoneCalls", len(f1))
    for style in ("bare", "rich"):
        accs = [sum(r["correct"] for r in f1 if r["size"] == z and r["style"] == style) / sum(1 for r in f1 if r["size"] == z and r["style"] == style) for z in sizes]
        put("Fone" + style.capitalize() + "Min", pct(min(accs), 1)); put("Fone" + style.capitalize() + "Max", pct(max(accs), 1))

# ---- twenty sums (review M10)
if os.path.exists(R + "sums.jsonl"):
    sm = rows("sums"); put("SumsN", len(sm)); put("SumsCorrect", sum(1 for r in sm if r["correct"]))
    put("SumsNumber", sum(1 for r in sm if r["any_number"])); put("SumsCalls", float(agg(sm)["calls"] or sum(r["calls"] for r in sm) / len(sm)), 0)

# ---- the pass mark measured: deployed code, pass mark 5, same 24 prompts (review B2)
if os.path.exists(R + "deployed-pass5.jsonl"):
    p5 = agg(rows("deployed-pass5", ORIG)); d4 = agg(rows("deployed-1528882", ORIG))
    put("PassFiveCalls", float(p5["calls"]), 0); put("PassFiveCrit", float(p5["critique"]), 2); put("PassFiveRetries", p5["retries"])
    put("PassFiveCallsDrop", pct(float(p5["calls"]) - float(d4["calls"]), float(p5["calls"])))

# ---- held-out prompts (review M8)
if os.path.exists(R + "heldout.jsonl"):
    h = agg(rows("heldout")); put("HeldN", h["replies"]); put("HeldCrit", float(h["critique"]), 2); put("HeldCalls", float(h["calls"]), 0)
    put("HeldEmpty", h["empty"]); put("HeldBelow", h["first_below5"]); put("HeldCPW", cpw(rows("heldout")), 1)

# ---- the capability list recited (review M9): tool off on four prompts
if os.path.exists(R + "tooloff.jsonl"):
    FOUR = ["v13", "v11", "s7", "v4"]
    t_on, t_off = agg(rows("deployed-1528882", FOUR)), agg(rows("tooloff", FOUR))
    put("ToolOnCrit", float(t_on["critique"]), 2); put("ToolOffCrit", float(t_off["critique"]), 2)
    put("ToolOnCalls", float(t_on["calls"]), 0); put("ToolOffCalls", float(t_off["calls"]), 0)

# ---- the instrument
import server
put("GrammarWords", len(server.GRAMMAR_SLOT_WORDS)); put("Categories", len(server.TREE_INDEX["categories"]))
put("ContentCategories", sum(1 for c in server.TREE_INDEX["categories"] if c["id"] != "grammar"))  # the word question skips grammar
leaves = set()
def walk(n):
    if n.get("type") == "leaf": leaves.add(n["word"])
    for c in n.get("children", []): walk(c)
for c in server.TREE_CATS.values(): walk(c)
put("VocabWords", f"{len(leaves):,}".replace(",", "{,}"))
put("Moods", len(server.MOODS)); put("Moves", len(server.MOVES)); put("Tools", len(server.TOOLS))

# ---- the human ratings: the sessions saved by /review, copied from the deployment as one file
if os.path.exists("paper/human-reviews.json"):
    import json
    h = server.review_results(json.load(open("paper/human-reviews.json")))
    put("HumanSessions", h["sessions"]); put("HumanRatings", h["ratings"]); put("HumanCards", h["cards"])
    for name, tag in (("deployed", "Dep"), ("baseline", "Base"), ("random", "Rand")):
        s = h["systems"].get(name)
        if s:
            put(f"Human{tag}N", s["n"]); put(f"Human{tag}Nat", float(s["naturalness"]), 2)
            put(f"Human{tag}Rel", float(s["relevance"]), 2)
            put(f"Human{tag}Mean", (s["naturalness"] + s["relevance"]) / 2, 2)
    if h["human_vs_critique_r"] is not None: put("HumanCritR", float(h["human_vs_critique_r"]), 2)
    if h["rater_spread"] is not None: put("HumanSpread", float(h["rater_spread"]), 2)
    p = h["pairs"]; put("PrefDep", p["deployed"]); put("PrefBase", p["baseline"]); put("PrefSame", p["same"])
    if sum(p.values()): put("PrefDepPct", pct(p["deployed"], sum(p.values())))

with open("paper/numbers.tex", "w") as fh:
    fh.write("% generated by paper/make_numbers.py; do not edit\n")
    for k in sorted(M):
        fh.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print(f"{len(M)} macros written")
for k in sorted(M): print(f"  {k} = {M[k]}")

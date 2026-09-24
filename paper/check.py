"""Mechanical checks of paper.tex. The build fails if any of them fails.
  1. every reply quoted with \\jev{..} or \\J{..} exists verbatim in bench/runs or transcripts.txt
  2. every number typed in the text is on the short allowlist below (dates, limits, the prompt's own
     numbers, facts with a named source); every experimental number must be a macro from numbers.tex
  3. every macro used is defined; every \\cite key is in refs.bib and every entry is cited
  4. facts taken from findings.md are still there
  5. no placeholder and no hype word
Run from the repo root: python paper/check.py"""
import glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
tex = open("paper/paper.tex").read()
body = tex.split("\\begin{document}", 1)[1]
body = re.sub(r"(?<!\\)%.*", "", body)
errors = []


def norm(s):
    s = s.replace("\\ldots", "...").replace("~", " ").replace("``", '"').replace("''", '"')
    return " ".join(s.lower().split())


# ---- 1. quotes
corpus = [norm(open("paper/transcripts.txt").read())]
for f in glob.glob("bench/runs/*.jsonl"):
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        corpus.append(norm(r.get("text", "")))
        corpus += [norm(t) for t in r.get("turns") or []]
quotes = re.findall(r"\\(?:jev|J)\{((?:[^{}]|\{[^{}]*\})*)\}", body)
for q in quotes:
    if not any(norm(q) in c for c in corpus):
        errors.append(f"quote not found in any run or transcript: {q[:80]!r}")

# ---- 2. typed numbers
ALLOWED = {
    "2026": "dates", "18": "18 September", "20": "20 September", "21": "21 September", "22": "22 September",
    "255": "the API's option cap (findings.md)", "200": "option-count probe (findings.md)",
    "0.61": "findings.md", "0.73": "findings.md", "0.96": "findings.md", "0.98": "findings.md",
    "12": "the prompt '12 plus 30'", "30": "the prompt", "42": "inside a quoted reply / the sum",
    "0": "the 0 to 6 scale", "6": "the 0 to 6 scale", "5": "the pass mark / 5 options", "3": "pass mark / three",
    "4": "a score of 4", "24": "the 24 original prompts", "1": "contribution (1)", "2": "contribution (2)",
    "32": "inside a quoted reply",
}
scan = re.sub(r"\\(?:ref|label|cite[pt]?|url|bibliography|bibliographystyle|input|textcolor|hangindent)\*?(\[[^\]]*\])?\{[^}]*\}", " ", body)
scan = re.sub(r"\\begin\{(?:tabularx|minipage|figure\*?|table)\}(\[[^\]]*\])?(\{[^}]*\})*", " ", scan)
scan = re.sub(r"\\[A-Za-z]+", " ", scan)  # macros, including the number macros
scan = re.sub(r"black!\d+|\d+(?:\.\d+)?(?:em|pt|mm|cm|in|\\textwidth)|rotate=\d+|!\d+(?:\.\d+)?!", " ", scan)  # lengths, colour mixes, TikZ geometry
for n in set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", scan)):
    if n not in ALLOWED:
        errors.append(f"typed number {n!r} is not on the allowlist: make it a macro or justify it in check.py")

# ---- 3. macros and citations
defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", open("paper/numbers.tex").read()))
preamble_defs = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex))
for m in set(re.findall(r"\\([A-Z][A-Za-z]+)", body)):
    if m not in defined and m not in preamble_defs and m not in {"U", "J", "Moves", "Moods"} | defined:
        if m not in {"LaTeX"}:
            errors.append(f"macro \\{m} is used but not defined in numbers.tex")
bib = set(re.findall(r"@\w+\{([^,]+),", open("paper/refs.bib").read()))
cited = {k.strip() for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", body) for k in group.split(",")}
errors += [f"cited but not in refs.bib: {k}" for k in sorted(cited - bib)]
errors += [f"in refs.bib but never cited: {k}" for k in sorted(bib - cited)]

# ---- 4. facts with a source outside the run files
findings = open("findings.md").read()
for needle in ("61%", "73%", "96%", "98%", "no degradation up to 200 options", "max 255 options"):
    if needle not in findings:
        errors.append(f"fact source changed: {needle!r} no longer in findings.md")

# ---- 5. placeholders and hype
for bad in ("TODO", "TBD", "XXX", "??", "lorem", "PAPERDOI"):
    if bad in tex:
        errors.append(f"placeholder {bad!r} in the text")
for word in ("novel", "groundbreaking", "revolutionary", "state-of-the-art", "paradigm", "delve", "game-chang",
             "unprecedented", "remarkabl", "crucial", "leverage", "seamless"):
    if re.search(rf"\b{word}", body, re.I):
        errors.append(f"hype word {word!r} in the text")
if "\u2014" in body or "---" in body:
    errors.append("em dash in the text")

print(f"checked {len(quotes)} quotes, {len(cited)} citations, {len(defined)} number macros")
if errors:
    print("\n".join("FAIL: " + e for e in errors))
    sys.exit(1)
print("paper checks ok")

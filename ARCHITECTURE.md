# Architecture — Generation is Classification

One primitive: `decide(state, questions) → {choice, probabilities}` per question.
Every component is one or more calls to this. Nothing else decides.

---

## Full flow

```
┌───────────────────────────────────────────────────────────────────────┐
│                        PERSISTENT SELF                                │
│ mood · they_seem · told                                               │
│ Written by REFLECT at the end of each turn; loaded at the start       │
│ of the next.                                                          │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ loaded into mind
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ 1. PERCEIVE                                   plan_response     │  │
│  │ One batched decide call.                                        │  │
│  │   intent   — greeting / question / opinion / request /          │  │
│  │              emotional / playful / sharing / hostile /          │  │
│  │              compound                                           │  │
│  │   feel     — 16 emotions (curious ... bored)                    │  │
│  │   tone     — casual / warm / direct / playful /                 │  │
│  │              thoughtful / serious / gentle / firm               │  │
│  │   attend   — which span matters most (multi-part msgs only)     │  │
│  │   recall_X — one yes/no per tool: who_i_am, the_time,           │  │
│  │              my_machine, what_i_can_do                          │  │
│  │                                                                 │  │
│  │ Output: Mind with all decisions. Tool text not yet read.        │  │
│  └────────────────────────────┬────────────────────────────────────┘  │
│                               │                                       │
│                               ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ 1b. SETUP (no decide calls)                                     │  │
│  │   · Load Self from previous turn (mood, they_seem, told)        │  │
│  │   · Load user's local time                                      │  │
│  │   · Read tool text for each recall_X that said yes              │  │
│  └────────────────────────────┬────────────────────────────────────┘  │
│                               │                                       │
│                               ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ 1c. PLAN LENGTH                                plan_length      │  │
│  │ One decide call, after tool text is in the state.               │  │
│  │   length — short / medium / detailed / elaborate                │  │
│  └────────────────────────────┬────────────────────────────────────┘  │
│                               │                                       │
│                               ▼                                       │
│  ╔═════════════════════════════════════════════════════════════════╗  │
│  ║ 2. GENERATE                           generate_response_tree    ║  │
│  ║ Up to MAX_CRITIQUE_ATTEMPTS (default 2) generate-critique       ║  │
│  ║ cycles. If the first attempt passes, no retry.                  ║  │
│  ║                                                                 ║  │
│  ║ ┌─────────────────────────────────────────────────────────────┐ ║  │
│  ║ │ 2a. BUILD REPLY                                             │ ║  │
│  ║ │                                                             │ ║  │
│  ║ │ ╔═════════════════════════════════════════════════════════╗ │ ║  │
│  ║ │ ║ SENTENCE LOOP (while not done, under max_words)         ║ │ ║  │
│  ║ │ ║                                                         ║ │ ║  │
│  ║ │ ║ A. PLAN MOVE                            _plan_move      ║ │ ║  │
│  ║ │ ║    One batched decide call:                             ║ │ ║  │
│  ║ │ ║      move   — answer / reason / example / personal      ║ │ ║  │
│  ║ │ ║               / caveat / new_angle / ask_back           ║ │ ║  │
│  ║ │ ║               / reassure / close / admit / disagree     ║ │ ║  │
│  ║ │ ║      about  — it / me / them / people / one             ║ │ ║  │
│  ║ │ ║      weight — quick / full / long                       ║ │ ║  │
│  ║ │ ║      reply  — complete / more                           ║ │ ║  │
│  ║ │ ║               (only at sentence boundaries, not         ║ │ ║  │
│  ║ │ ║                before the first sentence)               ║ │ ║  │
│  ║ │ ║    If reply == complete: stop.                          ║ │ ║  │
│  ║ │ ║                         │                               ║ │ ║  │
│  ║ │ ║ B. BRANCH                           _branch_combos      ║ │ ║  │
│  ║ │ ║    No decide call. The i-th branch takes the i-th       ║ │ ║  │
│  ║ │ ║    ranked option of move × about × weight.              ║ │ ║  │
│  ║ │ ║    K = beam_width(zoom, mind), default 3.               ║ │ ║  │
│  ║ │ ║                         │ K branches in parallel        ║ │ ║  │
│  ║ │ ║                         ▼                               ║ │ ║  │
│  ║ │ ║ C. WORD LOOP ×K                _generate_sentence       ║ │ ║  │
│  ║ │ ║    (see Word Loop below)                                ║ │ ║  │
│  ║ │ ║    Returns (text, quality, word_count, zoom)            ║ │ ║  │
│  ║ │ ║                         │ K results                     ║ │ ║  │
│  ║ │ ║                         ▼                               ║ │ ║  │
│  ║ │ ║ D. BEAM SELECT                      _beam_select        ║ │ ║  │
│  ║ │ ║    Up to one decide call: "which is most natural        ║ │ ║  │
│  ║ │ ║    and adds the most new information?"                  ║ │ ║  │
│  ║ │ ║    Skipped when only one valid branch survives.         ║ │ ║  │
│  ║ │ ║    Filters out broken, repeating, lost, short.          ║ │ ║  │
│  ║ │ ║                                                         ║ │ ║  │
│  ║ │ ║    No survivor (first sentence only)?                   ║ │ ║  │
│  ║ │ ║      → reopen: mind.with_previous, re-plan, retry       ║ │ ║  │
│  ║ │ ║      → still nothing: judge among all dead branches     ║ │ ║  │
│  ║ │ ║    No survivor (later sentences)? Reply ends here.      ║ │ ║  │
│  ║ │ ║                         │                               ║ │ ║  │
│  ║ │ ║    Append winner to partial. Loop back to A.            ║ │ ║  │
│  ║ │ ╚═════════════════════════════════════════════════════════╝ │ ║  │
│  ║ │                                                             │ ║  │
│  ║ │ Post: trim trailing fragment (no decide call).              │ ║  │
│  ║ └─────────────────────────────────────────────────────────────┘ ║  │
│  ║                            │                                    ║  │
│  ║                            ▼                                    ║  │
│  ║ ┌─────────────────────────────────────────────────────────────┐ ║  │
│  ║ │ 2b. CRITIQUE                        critique_response       │ ║  │
│  ║ │ One batched decide call:                                    │ ║  │
│  ║ │   grammar   — correct / minor / broken                      │ ║  │
│  ║ │   relevance — relevant / partial / off_topic                │ ║  │
│  ║ │   natural   — natural / awkward / robotic                   │ ║  │
│  ║ │   weakest   — which sentence to redo (if >1 sentence)       │ ║  │
│  ║ │ Score = 2 per best + 1 per middle + 0 per worst. Max 6.     │ ║  │
│  ║ │ Pass threshold: 4.                                          │ ║  │
│  ║ └─────────────────────────────────────────────────────────────┘ ║  │
│  ║                            │                                    ║  │
│  ║    score >= 4: done.       │                                    ║  │
│  ║    score < 4:              │                                    ║  │
│  ║                            ▼                                    ║  │
│  ║ ┌─────────────────────────────────────────────────────────────┐ ║  │
│  ║ │ 2c. PAIRWISE PICK                          _better_of       │ ║  │
│  ║ │ Only on the second attempt (on the first, there is          │ ║  │
│  ║ │ nothing to compare against).                                │ ║  │
│  ║ │ One decide call: "which reply is better?"                   │ ║  │
│  ║ │ Keep the better of old best and new attempt.                │ ║  │
│  ║ │ Restart from weakest sentence. Loop back to 2a.             │ ║  │
│  ║ └─────────────────────────────────────────────────────────────┘ ║  │
│  ╚═════════════════════════════════════════════════════════════════╝  │
│                               │                                       │
│                               ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ 3. REFLECT                                        reflect       │  │
│  │ One batched decide call:                                        │  │
│  │   mood      — 20 options (cheerful ... confused)                │  │
│  │   they_seem — company / answers / comfort / fun / test /        │  │
│  │               know_me / vent / argue / space                    │  │
│  │   told      — which of my sentences said something about me     │  │
│  │               (picks from actual reply sentences, or "none")    │  │
│  │ Output: updated Self, persisted for next turn.                  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│ Output: complete reply text, sent as one event after all              │
│ retries. Construction is word-by-word; the user sees the              │
│ finished text.                                                        │
└───────────────────────────────────────────────────────────────────────┘
```

---

## The word loop

Inside each sentence (`_generate_sentence`). Each iteration produces one word.

```
┌───────────────────────────────────────────────────────────────┐
│ STATE                                                         │
│ format_state(messages, partial, mind)                         │
│ = conversation history + mind.render() + said so far          │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│ PREDICT NEXT WORD                         predict_next_word   │
│ One batched decide call, two questions:                       │
│                                                               │
│ Q1: CATEGORY                                                  │
│   "you're building: 'i love ___'. what word comes next?"      │
│   Options (~200, one flat list):                              │
│     · 152 grammar words (i, the, a, is, to, ...)              │
│       each as phrase criterion: "i love [word]"               │
│     · echo words from user's message + tool header            │
│     · 35 content categories (feeling, action, place, ...)     │
│     · END_SENTENCE — "end with a period"                      │
│     · DIGITS — "a number in digits"                           │
│   Output: winner + probability distribution.                  │
│                                                               │
│ Q2: LISTEN                                                    │
│   "how is this response coming along?"                        │
│   Options:                                                    │
│     · flowing  — "needs more words"                           │
│     · complete — "stop here" (only when partial non-empty)    │
│     · lost     — "repeating, nonsensical, or off track"       │
│                                                               │
│ ZOOM UPDATE: zoom = update_zoom(zoom, top_prob)               │
│   unsure = top_prob < 0.65                                    │
│   width  = lookahead_width(zoom) if unsure else 0             │
└───────────────────────────────┬───────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
     listen says:          listen says:          listen says:
     complete/lost           flowing,              flowing,
          │                 not unsure              unsure
          │                    │                      │
          ▼                    ▼                      ▼
         STOP               RESOLVE              LOOKAHEAD JUDGE
                            CATEGORY             then EMIT
                            then EMIT
```

### Lookahead judge

Fires when the classifier is unsure (top probability < 0.65).

```
1. Take top-K categories by probability (K up to 5).
2. RESOLVE each to a word — K parallel calls, each 0–2 decide
   calls (0 for grammar/echo, 1–2 for content via _tree_navigate).
3. FORWARD each word one step — K parallel predict_next_word
   calls (each is 1 batched decide + 0–2 resolve; result used
   only for the two-word phrase).
4. JUDGE — one decide call: "which two-word continuation sounds
   most natural?" Options are the phrases, e.g.
   "love dogs", "love music", "love being".
5. Output: the winning word, overriding the original category.
```

### Resolve category

Turns a category winner into a word. Used both in the main path
(when confident) and inside the lookahead (step 2 above).

```
grammar word (i, the, is, ...)  → return directly, no call
echo word (from user's message) → return directly, no call
END_SENTENCE                    → return ".", no call
DIGITS                          → _say_number: loop of decide
                                  calls, one per digit, up to 12
content category                → _tree_navigate: 1–2 decide
                                  calls walking the vocab tree
                                  (36 categories, 4756 words,
                                  depth ≤ 2)
```

### Emit word

```
· Apply article: a/an by _wants_an (no decide call)
· Punctuation: attached directly (no space before . , ! ?)
· Append to partial
· Loop detection: last 4 words ABAB or AAAA → stop
· Word is . ! ? → sentence complete, return
· Otherwise → loop back to PREDICT NEXT WORD
```

---

## Calls per reply

Typical, with BEAM_WIDTH=3.

| Phase | Calls | Notes |
|---|---|---|
| PERCEIVE | 1 | batched: intent, feel, tone, attend, recall_X |
| PLAN LENGTH | 1 | separate call, after tool text loaded |
| PLAN MOVE (×sentences) | 1 | batched: move, about, weight, reply |
| WORD LOOP (×words ×K) | 1 | batched: category + listen |
| TREE NAVIGATE (×content words) | 1–2 | walk to leaf |
| LOOKAHEAD (×unsure words) | K+1 to 5K+1 | K resolves (0–2 each) + K forwards (1–3 each) + 1 judge |
| BEAM SELECT (×sentences) | 0–1 | pick best of K (skipped if only 1 valid) |
| CRITIQUE | 1 | batched: grammar, relevance, natural, weakest |
| PAIRWISE (if retry) | 1 | which reply is better |
| REFLECT | 1 | batched: mood, they_seem, told |

Measured average: ~12 calls per word in the deployed version.

---

## State composition

`format_state(messages, partial, mind)` builds the text every decide call sees.

```
[about me: <recalled tool texts, if any>]

user: <message>
jev: <previous reply>
user: <current message>

[thinking: <intent>. <attend>. i feel <feel>. i want it to
come across <tone>. i've been feeling <mood>. they seem to
want <they_seem>. i already told them: '<told>'. <length>.
next: <move>, <about>, <weight>. said so far: <covered
content words>.]

(On a retry after critique or reopen, also includes:
 "my last try was '...' and it was <issue>, so this time
 differently".)

jev: <partial — said so far>
```

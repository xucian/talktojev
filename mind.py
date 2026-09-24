"""The mind: decisions written back into the state as one first-person block.

Every classifier answer becomes a Decision. A Mind holds the turn's decisions and
renders them, from the option descriptions the classifier chose between, into the
block that every later question reads. Nothing here is free text.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

SOFT_WITHIN = 0.2
MIND_SOFT = os.environ.get("MIND_SOFT", "1") == "1"
ZOOM_START = 0.3
SOFT_NAMES = frozenset({"feel", "tone"})
# What the length decision implies as a count, rendered as "this is sentence n of about m" so the
# listen can say complete instead of the word cap ending the reply (sep21-final-changes.md, A).
EXPECTED_SENTENCES = {"short": 1, "medium": 2, "detailed": 3, "elaborate": 4}  # 4 x ~9 words fits under the cap


def update_zoom(zoom: float, confidence: float) -> float:
    """One smoothed number in [0, 1]: how unsure the last decisions left us."""
    return 0.6 * zoom + 0.4 * (1.0 - confidence)


@dataclass(frozen=True)
class Decision:
    name: str
    winner: str
    probs: dict[str, float]
    options: dict[str, str]

    @classmethod
    def from_answer(cls, name: str, answer: dict, options: dict[str, str]) -> Decision:
        probs = dict(answer.get("probabilities") or {})
        winner = answer.get("choice") or (max(probs, key=probs.get) if probs else "")
        return cls(name, winner, probs, options)

    @property
    def confidence(self) -> float:
        if not self.probs:
            return 1.0
        return self.probs.get(self.winner, 0.0)

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.probs.items(), key=lambda kv: -kv[1])

    def top(self, k: int) -> list[str]:
        ids = [oid for oid, _ in self.ranked()][:k]
        return ids or [self.winner]

    def runner_up(self) -> tuple[str, float] | None:
        for oid, p in self.ranked():
            if oid != self.winner:
                return oid, p
        return None

    def desc(self, option_id: str | None = None) -> str:
        oid = self.winner if option_id is None else option_id
        return self.options.get(oid, oid)

    def render(self, soft: bool = False) -> str:
        ru = self.runner_up() if soft else None
        if ru and self.confidence - ru[1] <= SOFT_WITHIN:
            return f"mostly {self.desc()}, a little {self.desc(ru[0])}"
        return self.desc()


@dataclass(frozen=True)
class Self:
    """What carries from one turn to the next: how i feel after the exchange, what they seem to
    want, and what i have told them about myself (my own sentences, quoted, the last few)."""
    mood: Decision | None = None
    they_seem: Decision | None = None
    told: tuple[str, ...] = ()

    def parts(self) -> list[str]:
        out = []
        if self.mood:
            out.append(f"i've been feeling {self.mood.render(soft=MIND_SOFT)}")
        if self.they_seem:
            out.append(f"they seem to want {self.they_seem.render()}")
        if self.told:
            out.append("i already told them: " + "; ".join(f"'{t}'" for t in self.told))
        return out


@dataclass(frozen=True)
class Mind:
    plan: dict[str, Decision] = field(default_factory=dict)
    self_: Self | None = None
    now: str | None = None  # the user's local date and time, as their browser sent it
    recalled: tuple[tuple[str, str], ...] = ()  # (tool name, what it read) for the tools the plan chose
    # The next sentence's decisions, in render order: move (what it does), about (who or what it is
    # about, F26/F30), weight (how much it carries). Each is (decision, chosen option id).
    sentence: dict[str, tuple[Decision, str]] = field(default_factory=dict)
    previous: tuple[str, str] | None = None
    retracted: tuple[str, ...] = ()

    def with_sentence(self, **chosen: tuple[Decision, str] | None) -> Mind:
        return replace(self, sentence={k: v for k, v in chosen.items() if v is not None})

    @property
    def move(self) -> Decision | None:
        return self.sentence["move"][0] if "move" in self.sentence else None

    def with_previous(self, text: str, issue: str) -> Mind:
        return replace(self, previous=(text, issue))

    def with_self(self, self_: Self | None) -> Mind:
        return replace(self, self_=self_)

    def with_now(self, now: str | None) -> Mind:
        return replace(self, now=now)

    def with_recalled(self, recalled: tuple[tuple[str, str], ...]) -> Mind:
        return replace(self, recalled=recalled)

    def with_plan(self, decisions: dict[str, Decision]) -> Mind:
        """More plan decisions, made after the first call; existing ones are kept."""
        return replace(self, plan={**self.plan, **decisions})

    def with_retracted(self, word: str) -> Mind:
        return replace(self, retracted=self.retracted + (word,))

    @property
    def length(self) -> str:
        d = self.plan.get("length")
        return d.winner if d else "short"

    def lowest_confidence(self) -> float:
        """Over hard decisions only: a soft decision's spread is nuance, not doubt, and a close
        call between two parts of the message ("attend") means both matter."""
        ds = [d for name, d in self.plan.items()
              if name not in SOFT_NAMES and name != "attend" and not name.startswith("recall_")]
        if self.move:
            ds.append(self.move)
        return min((d.confidence for d in ds), default=1.0)

    def _move_part(self) -> str | None:
        if not self.sentence:
            return None
        return "next: " + ", ".join(d.desc(oid) for d, oid in self.sentence.values())

    def _length_part(self, sentence_no: int | None) -> str | None:
        d = self.plan.get("length")
        if not d:
            return None
        part = d.render()
        expected = EXPECTED_SENTENCES.get(d.winner)
        if sentence_no and expected and expected > 1:
            part += f". this is sentence {sentence_no} of about {expected}"
        return part

    def render(self, covered: list[str] = (), sentence_no: int | None = None) -> str:
        p = self.plan
        parts = []
        if "intent" in p:
            parts.append(p["intent"].render())
        if "attend" in p:
            parts.append(f"they said several things; the part that matters most: '{p['attend'].desc()}'")
        if "feel" in p:
            parts.append(f"i feel {p['feel'].render(soft=MIND_SOFT)}")
        if "tone" in p:
            parts.append(f"i want it to come across {p['tone'].render(soft=MIND_SOFT)}")
        if self.self_:
            parts.extend(self.self_.parts())
        if self._length_part(sentence_no):
            parts.append(self._length_part(sentence_no))
        if self._move_part():
            parts.append(self._move_part())
        if self.previous:
            parts.append(f"my last try was '{self.previous[0]}' and it was {self.previous[1]}, so this time differently")
        if self.retracted:
            parts.append(f"i took back: {', '.join(self.retracted)}")
        if covered:
            parts.append(f"said so far: {', '.join(covered)}")
        return f"[thinking: {'. '.join(parts)}.]\n" if parts else ""

    def brief(self, covered: list[str] = (), with_move: bool = True) -> str:
        """The parts that change most, quoted into question instructions."""
        parts = []
        if "intent" in self.plan:
            parts.append(self.plan["intent"].render())
        if "attend" in self.plan:
            parts.append(f"they said several things; the part that matters most: '{self.plan['attend'].desc()}'")
        if with_move and self._move_part():
            parts.append(self._move_part())
        if self.self_ and self.self_.told:
            parts.append("i already told them: " + "; ".join(f"'{t}'" for t in self.self_.told))
        if covered:
            parts.append(f"said so far: {', '.join(covered)}")
        return f"(you're thinking: {'. '.join(parts)}.) " if parts else ""

# talk to jev

A chatbot with no language model in it.

Every word Jev says was picked from a list by a classifier. The only model is
[Jev](https://openrouter.ai/typesafe/jev-1.13), TypeSafe AI's "System One" decision model, which can
do exactly one thing: given some text and a set of options, choose one. It cannot write a token.
This project is the claim that choosing is enough: **generation is classification, repeated, with a
memory in between.**

Try it: https://talktojev.com · Paper: https://doi.org/10.5281/zenodo.22940945 (also at https://talktojev.com/paper.pdf; source: https://github.com/xucian/generation-is-classification) · This code, archived: https://doi.org/10.5281/zenodo.22895303 · Jev on X: https://x.com/jevprime · Video: _soon_

This is the public copy of the code, `github.com/xucian/talktojev`.

```
> who are you
jev: i am jev
> i am your boss
jev: see that you are rude
> tell me a full joke
jev: i see that you are rude but i will do it anyway because i am helpful. a man was walking
     down the street and he stopped to wait for a bus because he was lazy.
```

## The idea

A speaker never generates; a speaker chooses. If that is true, a model that can only choose should
be able to speak, provided the choices are put to it well. "Thinking" is then the same primitive
asked at coarser grains, with each answer written back into the text the next question reads.

| grain | the question put to the classifier |
|---|---|
| reply | what are they doing? how much should i say? how do i feel about it? should i look something up? |
| sentence | what does the next sentence do, who is it about, how much does it carry? is the reply complete? |
| word | which word comes next: one of ~150 function words, one of their words, or a kind of word? |
| word, when unsure | of the top candidates, imagined one word further, which continuation is best? |
| letter, for numbers | which digit comes next, or is the number complete? |
| after the reply | is it grammatical, relevant, natural? which sentence is weakest? how do i feel now? |

Nothing in the loop is a template or a phrase bank, and nothing is forbidden: when output goes wrong,
the fix is more information in the state or a better description of an option, never a rule.

This repository is **one implementation** of that idea, written quickly and iteratively. It is not
a one-to-one rendering of the paper, and it is not tidy. The paper is the idea; the code is a proof
that it runs.

## What it can do

Hold a conversation, keep its mood and what it said about itself across turns, say the user's own
words back (names included), do small sums digit by digit, and use tools the way it does everything
else: by being asked a yes/no question per tool ("would it help to look up the time?") and choosing.

## Run it

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env              # put an OpenRouter key in it
./.venv/bin/python server.py      # http://localhost:8787
```

`python bench.py run --only v11,s7` runs bench prompts and records per-reply metrics.
The public site runs this same code behind `gate.py`: a message length cap, per-minute and per-day
limits, a communal daily budget, or the visitor's own key.

Cost and speed, honestly: a decision takes about 0.3 s and a few thousand input tokens; a short
reply is a dozen decisions and two seconds, a five-sentence reply is several hundred decisions and
up to a minute.

## Credits

- **Jev**, the model that does all the choosing: built by Diogo Almeida at TypeSafe AI
  ([Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)),
  reached through OpenRouter's Decisions API.
- WordNet (Miller, 1995) through NLTK (Bird, Klein & Loper, 2009); word frequencies from `wordfreq`
  (Speer, 2022).
- The claim, the direction and the principles: the author. The implementation and the experiments were
  done with Claude (Anthropic) as the engineering collaborator, the human deciding at every step.

## Cite

```
@misc{lazar2026generation,
  title     = {Generation is Classification: a conversational agent built from a model that can only choose},
  author    = {Lazar, Lucian},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22940945},
  note      = {Demo: https://talktojev.com. Code: https://github.com/xucian/talktojev (doi:10.5281/zenodo.22895303)}
}
```

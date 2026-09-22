# talk to jev

**Generation is Classification: a chatbot with no language model in it.**

Every word Jev says is picked from a list by a classifier. The only model is
[Jev](https://openrouter.ai/typesafe/jev-1.13), TypeSafe AI's decision model, which can do exactly one
thing: given some text and a set of options, choose one. It cannot write a token. The claim is that
choosing is enough: generation is classification, repeated, with a memory in between.

- **Try it:** https://talktojev.com
- **Paper:** _link added when it is up_
- **Code:** this repository will hold the full implementation (the generation loop, the vocabulary,
  the bench and every run file the paper's numbers come from). It is being cleaned up for release
  and will be published here; the demo already runs on it.

Until then, the paper describes the method in enough detail to rebuild it, and the demo shows what it
does. 

## Cite

```
@misc{lazar2026generation,
  title  = {Generation is Classification: language from a model that can only choose},
  author = {Lazar, Lucian},
  year   = {2026},
  note   = {Demo: https://talktojev.com. Code: https://github.com/xucian/talktojev}
}
```

License: MIT (see LICENSE).

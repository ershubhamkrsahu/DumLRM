# My View

A journal of the substantial work on DumLRM — capabilities, design decisions, and improvements worth remembering. I skip the boilerplate and the fixes; those belong in the commit history.

## 2026-11-11 — Chapter 2: getting Qwen3's tokenizer under my fingers

I pulled down the small Qwen3 base tokenizer and ran real text through it before touching any model code. Encoding "Explain the large language models." gave me seven clean tokens, so I at least know the vocab and the encode/decode path agree with each other.

I also grabbed the 0.6B base checkpoint locally so I can load weights without re-downloading. I'm deliberately keeping both the tokenizer file and the checkpoint out of Git — they're reproducible downloads, not source, and 71 MB of binary in history is a mistake I don't want to make on day one.

Next I want the checkpoint loaded in the Qwen3 class and a forward pass verified before I trust anything downstream.


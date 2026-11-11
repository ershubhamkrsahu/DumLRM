# My View

A journal of the substantial work on DumLRM — capabilities, design decisions, and improvements worth remembering. I skip the boilerplate and the fixes; those belong in the commit history.

## 2026-11-11 — Chapter 2: getting Qwen3's tokenizer under my fingers

I pulled down the small Qwen3 base tokenizer and ran real text through it before touching any model code. Encoding "Explain the large language models." gave me seven clean tokens, so I at least know the vocab and the encode/decode path agree with each other.

I also grabbed the 0.6B base checkpoint locally so I can load weights without re-downloading. I'm deliberately keeping both the tokenizer file and the checkpoint out of Git — they're reproducible downloads, not source, and 71 MB of binary in history is a mistake I don't want to make on day one.

Next I want the checkpoint loaded in the Qwen3 class and a forward pass verified before I trust anything downstream.

## 2026-11-11 — Chapter 2: tokenizer verified, weights on disk, device picked

Encoding and decoding now agree, so the tokenizer side of chapter 2 is solid. I made the notebook print each token with the decoded text wrapped in backticks — a small change, but it makes whitespace visible, and whitespace is exactly where I'd expect BPE to bite me later.

Then I pulled the full 0.6B base checkpoint (1.4 GiB) and added the device-selection helper. My 3060 runs CUDA 12.4 with torch 2.4.1, so the helper takes the legacy TF32 branch; newer torch versions moved that setting to the fp32_precision API. I'm enabling TF32 because the matmul speedup is worth the tiny precision loss while I'm still getting things working.

Next: load the checkpoint into the Qwen3 model and validate the forward pass. That's the first real test of whether the architecture code is actually correct.

## 2026-11-11 — Chapter 2: the pretrained checkpoint loads clean

The Qwen3 0.6B weights went into my `Qwen3Model` with a strict state-dict match on the first attempt. That's the signal I was waiting for: the config and parameter names line up with the real model, so the architecture I'm building matches what these weights expect.

Loading is necessary but not sufficient, though. Next I need an actual forward pass and a look at the logits before I trust the model for generation.

## 2026-11-11 — Chapter 2: first forward pass

The model ran end to end: six prompt tokens in, a `[6, 151936]` logits tensor out. That vocabulary dimension matching Qwen3's 151,936 tokens is the concrete proof that the output head and the checkpoint actually agree, not just that the parameter names lined up.

The remaining bridge is turning those logits into predicted tokens — that's where an off-by-one or a wrong transpose would finally show up.

## 2026-11-11 — Chapter 2: it generates text

I wired up a streaming generator — greedy argmax on the last position, yield each token, stop on EOS — and it produced coherent English on the first run. Watching my own forward pass turn into readable output is the point where chapter 2 stopped feeling theoretical.

The implementation is deliberately naive: it re-runs the whole sequence for every new token, so it gets slower the longer the output. I'm leaving that alone for now; the KV-cache version is where the real optimization work starts, and I'd rather have a correct baseline to compare against.

## 2026-11-11 — Chapter 2: baseline numbers

I stopped guessing and measured the naive generator: 25 new tokens in 1.08 s, about 23 tokens/sec, with 1.47 GB peak CUDA memory. That's the baseline every optimization from here has to beat.

The detail that matters is the measurement itself — synchronizing CUDA before reading the clock and resetting peak memory afterwards. Without the sync the timing is meaningless, because the kernel launches return before the GPU finishes. Now the KV-cache rewrite gets a fair comparison instead of a flattering one.

## 2026-11-11 — Chapter 2: the KV cache is correct, but not faster

I added a cached generator that keeps keys and values across steps and only feeds the newest token. It produces the same text as the naive loop, so the mechanism is right — but the benchmark says 24 tokens/sec against 27 for the version that recomputes everything, with memory essentially unchanged (2.90 vs 2.94 GB).

The reason is in the cache itself: this implementation concatenates the cached K/V with the new ones on every step, so each token copies an ever-growing buffer through all 24 layers. The saved compute doesn't outrun that copy at these lengths. It also means the "optimization" is measured honestly instead of assumed.

I'm keeping both generators side by side. The next step is the real fix — preallocating the cache and writing into a fixed slot — and I want the before/after numbers to stay in the notebook so the speedup is provable.


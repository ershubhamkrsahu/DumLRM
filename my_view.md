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


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

## 2026-11-11 — Chapter 2: skipping torch.compile

I tried wrapping the generator in `torch.compile` and it failed with "Cannot find a working triton installation". On Windows that's a toolchain problem rather than a code problem, and I'm not going to chase it right now — the manual KV-cache rewrite is the thing I actually want to understand. Once the cache is preallocated and I have clean before/after numbers, I'll come back and see whether compile stacks on top.

## 2026-11-11 — Chapter 3: a real prompt, and the base model does math

I split the notebooks by chapter — chapter 2 is now `ch02_llm_basics`, and chapter 3 begins in `ch03_measuring_performance`. The first thing I wanted was a reusable `load_model_and_tokenizer` helper so I stop copy-pasting model setup between notebooks; it handles both the base and reasoning checkpoints, including the reasoning tokenizer's chat and thinking flags.

Then I gave the base model an actual math problem: given `a+b=3` and `ab=13/6`, find `a²+b²`. It ran for up to 2048 tokens, wrote out the full derivation, and landed on `14/3` — correctly. Rendering the generated LaTeX with `display(Math(...))` also showed me how much of the "reasoning" is just the model writing the algebra out step by step.

Chapter 3 is about measuring this properly, so the next step is timing long generations like this one instead of eyeballing them.

## 2026-11-11 — Chapter 3: the answer grader works

I finally have a grader I trust enough to build on. `split_into_parts` handles tuple and list answers, and `grade_answer` normalizes both sides and compares each part with the sympy check. Sixteen cases — fractions, roots, decimals, tuples, percentages, `\text{}`, `\cdot`, degrees, superscripts — all land on the expected result, including the tuple case that returned False a few cells earlier.

The cases that are supposed to fail also fail: wrong order, wrong length, `50%` versus `1/2`, and `0.333...` versus `1/3`. That matters more than the passes — a grader that says yes to everything would be worse than having none.

Next is running this over a real evaluation set so I get an actual accuracy number instead of a vibe.

## 2026-11-11 — Chapter 3: the first real accuracy number

I ran the base model over the first 10 MATH-500 problems with a "you are a math assistant" template and got 30% (3/10) in 21 seconds. That's the first honest number I've had, and it's lower than the earlier standalone math prompt made things feel — the same model that produced `14/3` answered `10` to that exact problem once it was wrapped in the assistant template.

That's useful rather than disappointing. It tells me the base model is very sensitive to prompt shape, and that the earlier correct answer wasn't representative of anything. The evaluation loop writes every generation to a JSONL file, so I can go read the failures instead of only counting them.

Next: inspect that file for systematic failure modes before changing anything, then run the full 500.

## 2026-11-11 — Chapter 4: inference scaling begins

Chapter 4 starts with a small but important change to the generation wrapper: `generate_text_stream_concat_flex` takes a `generate_func` plus extra keyword arguments, so the sampling and multi-candidate strategies later in the chapter can plug into the same loop instead of me rewriting it each time.

The first sample I ran makes the chapter's case for it. Asked to solve `(3x-9)/2 = x+37`, the base model answered `20`; the actual answer is `83`. One greedy sample is not reliable, which is exactly why the next sections are about generating more than one and choosing between them.

I'd rather build that on a clean interface now than bolt it on after the fact.

## 2026-11-11 — Chapter 4: "explain step by step" fixes the wrong answer

Appending "Explain step by step." to the same prompt turned that `20` into a full derivation ending in `\boxed{83}` — correct. Same model, same greedy decoding, same problem; the only change is asking it to show its work.

That reframes the chapter for me. The model already knew the algebra, it just wasn't committing to the intermediate steps. Sampling a pile of guesses may still help later, but this is the cheapest accuracy win available, and it tells me the later techniques are adding something on top of a model that can already reason when prompted properly.

I'm keeping the prompt change explicit in the notebook so it's clear which single variable moved the result.

## 2026-11-12 — Chapter 4: temperature turns the model into a sampler

I added temperature scaling to the cached generator: divide the logits by `T`, softmax them, then draw with `torch.multinomial` instead of taking the argmax. Temperature 0 stays on the exact greedy path, so I always have deterministic output to fall back on.

Seeing the distribution made it click. At low temperature the "Berlin" completion still dominates but is no longer guaranteed — 1000 samples at `T=0.35` gave roughly 443 ` berlin`, 292 newline, and 116 ` Berlin`. At `T=5` the logits flatten into near-uniform noise, which is exactly the failure mode sampling can introduce.

That's the tradeoff I'm buying into: diversity is what lets me try several answers, but it only helps if I have a way to pick the good one. Next is generating multiple candidates and choosing between them.

## 2026-11-12 — Chapter 4: top-p keeps sampling from wandering off

I wired the nucleus filter into a real generator: `top_p_filter` sorts probabilities, keeps the smallest set whose mass crosses `p`, and renormalizes. `generate_text_top_p_stream_cache` applies it after temperature scaling, with `top_p=None` meaning unfiltered.

Running the algebra prompt at `temperature=1.1, top_p=0.8` still reached `83`, but through a different derivation than the greedy run. That's the behavior I wanted from inference scaling — different reasoning paths, not just different wording.

Top-p also fixes the failure mode I saw with high temperature alone: it cuts the flat, low-probability tail that produces nonsense while leaving real choice among the plausible tokens.

## 2026-11-12 — Chapter 4: self-consistency turns samples into a decision

Sampling several answers does nothing on its own; I still needed a way to choose between them. `self_consistency_vote` generates `num_samples` answers, extracts the final candidate from each, and takes the majority. Ties return `None` instead of letting the first answer win arbitrarily.

The first run was the tie case I needed to see: five samples at temperature 0.8 and top-p 0.9 gave `83` twice, `11` once, and two empty extractions, so there was no winner. That's a real limitation — a sample the extractor can't parse is a vote thrown away, and two of them is enough to deadlock a small sample.

Running the same five samples batched gave three `83` and settled on `83` in 17.53s, against 39.92s for the sequential pass. Roughly 2.3× faster for the same work, which makes batching the version worth keeping. The empty extractions are the next thing to fix.

## 2026-11-12 — Chapter 5: scoring answers without a reference

Self-refinement needs a way to tell whether a revision actually helped, and at inference time there is no ground truth to check against. So `heuristic_score` rewards what I can verify without one: a properly boxed final answer (+2), a fallback extracted number (+1), and a brevity bonus of `1.5 * exp(-len/500)` so a padded derivation cannot win just by being long.

On the algebra prompt the conservative sample (temperature 0.5, top-p 0.5) scored 2.340 against 2.170 for the looser one (0.9/0.7). Both reached `83`, so the gap is entirely formatting and length — which is all a reference-free score can see, and also its danger: nothing in it stops a shorter answer from being a wrong one.

## 2026-11-12 — Chapter 5: a sequence score that survives length

Ranking whole answers needs a likelihood over the sequence, not one token. Multiplying per-token probabilities is the textbook definition, but in bfloat16 a five-token sentence underflowed to exactly `0.0000` — and any real answer would too, so the product cannot discriminate between candidates at all.

Summing log-probabilities fixes it. The same sentence ending in "Berlin" scored `-16.75`, against `-30.00` ending in "Bridge" — a real gap, and still meaningful when the answer runs to hundreds of tokens. That is the score I'll use to compare refined drafts.

The catch is that raw log-likelihood rewards shorter sequences for free, which is the brevity bonus all over again but hidden inside the math. I'll have to decide whether to normalize by length or accept the bias deliberately.

## 2026-11-12 — Chapter 5: averaging per-token scores fixes the length bias

That length problem had a straightforward fix: score only the answer tokens and take the mean instead of the sum. `avg_logprob_answer` encodes the prompt and the answer separately so it knows exactly where the answer starts, then averages the log-probabilities of those tokens only.

The result is a scale I can compare across answers of different lengths. "The capital of Germany is Berlin." scores `-0.2100`, while the same sentence ending in "Bridge." scores `-3.9062`. Both candidate derivations of the algebra problem land at `-0.069` — which is at least plausible, since both are the model's own samples conditioned on its own prompt.

I'm treating this as my confidence measure now, with one caveat: the second comparison cell prints a stale `score` variable instead of recomputing it, so that duplicated `-0.069` is not actually evidence yet. Fix that before trusting any comparison.

## 2026-11-12 — Chapter 5: the refinement loop works, the score does not

The full loop is in place: draft an answer, ask the model to critique it, ask it to revise using that critique, and keep the revision only if the score does not drop. On the algebra problem both iterations kept `83` — but the log-probability score rejected both revisions, `-1.375` and then `-0.301` against `-0.053` for the original draft.

That is the first result that contradicts the plan. The revisions are not worse answers; they were written in response to a critique, which is a narrower and more constrained context than the original question, so their per-token likelihood is naturally lower. Scored this way the loop can only ever keep the first draft — a self-refinement loop that refuses every refinement.

The obvious fixes all have costs: compare the extracted answers rather than the text, normalize the score against the refinement prompt, or accept only when the answer changes. I want to pick one deliberately instead of grabbing whichever makes the numbers go up.

## 2026-11-12 — Chapter 6: a reward that only counts boxed answers

Rollouts now come from a dedicated `sample_response` that keeps the KV cache and returns the token ids plus the prompt length, because GRPO needs the whole sequence and the boundary between prompt and answer. Rewards are binary: `reward_rlvr` extracts the final candidate and hands it to the grader from chapter 3, returning `1.0` or `0.0`.

The test rollouts show the rule's sharp edge. `\boxed{83}` and "The correct answer is `\boxed{83}`" both score 1.0 and `\boxed{38}` scores 0.0 — but so does "The final answer is 83". With `fallback=None` the extractor deliberately refuses anything unboxed, so the model has to learn the format along with the reasoning. That's a choice, not an accident, and I want to see whether it costs me answers that were right.

Turning the rewards into signal is just group normalization: `[1, 1, 0, 0]` becomes advantages of `±0.8659`. That is the GRPO idea at this scale — compare each rollout against its siblings instead of against an absolute score.

## 2026-11-12 — Chapter 6: the score has to keep its gradients

Sequence scoring is back, but this time it has to be differentiable. GRPO updates the policy using the log-probability of what the model actually generated, so `sequence_logprob` runs without `inference_mode`, casts the logits to float, and gathers the sampled token ids. The result still carries a `SumBackward0` node, which is the whole point.

The number agrees with my earlier long-hand estimate: mean log-prob times token count gives `-15.75`, the direct sum gives `-15.7349`. That's a useful cross-check that the chapter 5 implementation wasn't subtly wrong.

The rollout scores also show that likelihood does not track correctness: `\boxed{83}` scores `-8.24`, "The correct answer is `\boxed{83}`" scores `-19.83`, and the unboxed "The final answer is 83" scores `-16.16`. The model is least comfortable with the exact format the reward cares most about, which is precisely the gap the training loop is meant to close.

## 2026-11-12 — Chapter 6: the first GRPO step learns nothing, correctly

The loss is in place: advantage times the log-probability of each rollout, negated and averaged. `compute_grpo_loss_batched` wraps sampling, rewarding, scoring, and that loss into a single call, and short-circuits when every advantage is zero instead of returning a meaningless gradient.

The first real run hit that branch: two rollouts, both scored 0.0, advantages `[0.0, 0.0]`, loss `-0.0`. On the Sam problem one rollout quit after four tokens with `<|endoftext|>` and the other was cut off mid-sentence with no box, so neither could be graded. Nothing was learned, and nothing should have been.

That is the real constraint on this training loop: with only two rollouts most groups will be all-wrong or all-right, and neither teaches anything. I need more samples per problem so the group actually contains a mix, and I should skip the examples where it doesn't.

## 2026-11-12 — Chapter 6: stopping before the real run

The training loop is written and it works: `train_rlvr_grpo` zeroes the gradients, computes the batched GRPO loss, backpropagates, clips them, steps the optimizer, writes loss/reward/response length to a CSV, and checkpoints along the way. I started a 50-step run and stopped it after three — step 3 was the first step with any signal at all (loss 13.43, average reward 0.5) — and kept the interrupt checkpoint.

I'm deliberately not running this to completion, and I'm not running the evaluation that comes after it. The machinery is in place but the setup is too thin to learn anything I'd trust: two rollouts per problem, most groups with no reward spread, and a reward that ignores correct answers that forgot to box. Training it as-is would mostly teach the model formatting noise.

So chapter 6 stops here, with the loop, logging, and checkpointing ready. I'm going to improve the GRPO itself in the next chapter before spending GPU hours on a full run — more rollouts per group, a reward that isn't so brittle, and an evaluation that can actually tell whether any of it helped.

## 2026-11-12 — Chapter 7: measuring GRPO before improving it

Improving the training loop without measurements is guesswork, so chapter 7 starts by deciding what to watch. `compute_advantage_stats` returns the advantages along with their mean and spread, which tells me whether a group actually contained any signal.

The other number is entropy. Per-token entropy is `-p log p` summed over the vocabulary: `1.37` for a peaked toy distribution and `11.93` at the uniform maximum for 151,900 tokens. `sequence_logprob_and_entropy` now returns both the answer log-probability and the mean per-step entropy of the answer tokens, so I can see both during training.

Entropy is the one I expect to matter most. If it collapses, the policy has stopped exploring and GRPO has nothing left to learn from; if it sits near the uniform maximum the model is guessing. Either way I want that visible instead of inferred from a loss curve.


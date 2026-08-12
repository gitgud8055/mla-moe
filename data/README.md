---
task_categories:
- text-generation
language:
- en
size_categories:
- n<1K
tags:
- inference-benchmark
- throughput
- evaluation
---

# mla-moe public evaluation dataset

Public (participant-facing) input set for the **MLA-MoE inference throughput** exercise.
Two target models, **512 prompts each**, with the model's own greedy continuation as the
correctness reference:

| subset | target model |
|---|---|
| `glm47/` | [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) |
| `dsv2lite/` | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) |

Grading is done on a **held-out private set** with the same size and length
distribution, generated the same way but from disjoint source text. Tune against the
public set; do not tune against a specific prompt.

## Files (per subset)

| file | content |
|---|---|
| `requests.txt` | the input. Line 1 = request count (`512`), then **one prompt per line**, verbatim single-line text. |
| `prompts.i32.txt` | prompt token ids, one line per prompt, space-separated (`add_special_tokens=False`). |
| `completions.i32.txt` | **reference output** — the model's greedy continuation, token ids, one line per prompt, ≤ 64 tokens. |
| `reference.json` | per-request `{prompt_len, completion_len, hf_nll, hf_ntok}` plus provenance (`model_dir`, `max_new`, `dtype`, `transformers_version`). |
| `manifest.json` | prompt-generation parameters: seed, length buckets, source book ids, length statistics, per-prompt target lengths. |

Token ids are produced by **each model's own tokenizer**, so `glm47/` and `dsv2lite/`
ids are not interchangeable — the prompt *text* differs too (independent length
trimming per tokenizer).

## Length characteristics

Prompt token lengths: **min 64, median 256, max 512** (mean 241), drawn from buckets
64 / 128 / 256 / 384 / 512. Longest `prompt + completion` is **576** tokens, well inside
the 4096-token budget the harness assumes.

## How the reference was produced

Greedy decoding of the target model itself — HF `transformers` 5.12.1, **bf16**,
`max_new_tokens=64`, no sampling. `hf_nll` is the teacher-forced negative
log-likelihood of the reference continuation, `hf_ntok` the number of scored tokens.

Because greedy decoding is a hard argmax, **any** small numeric difference (dtype,
kernel, quantization) can make a completion diverge from some token onward. Correctness
is therefore scored with fuzzy metrics (METEOR + BERTScore-F1) against this reference,
not exact match.

## Prompt provenance

Prompts are sampled from 12 **public-domain Project Gutenberg** books, cleaned into
single-line paragraphs, then trimmed to a target token length with the model's own
tokenizer. Generation is deterministic (seed recorded in `manifest.json`).
Source book ids: 76, 84, 98, 174, 768, 1232, 1260, 1342, 1400, 1661, 2554, 2701.

The prompt text is public domain. The reference completions are outputs of
GLM-4.7-Flash and DeepSeek-V2-Lite and are subject to those models' own licenses.

## Loading

```python
from huggingface_hub import snapshot_download

path = snapshot_download("thanhnx12/mla-moe-dataset-public", repo_type="dataset")

with open(f"{path}/glm47/requests.txt") as f:
    n = int(f.readline())
    prompts = [f.readline().rstrip("\n") for _ in range(n)]

with open(f"{path}/glm47/completions.i32.txt") as f:
    reference = [[int(t) for t in line.split()] for line in f]

assert len(prompts) == len(reference) == 512
```

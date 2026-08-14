# Token-budget batching benchmark

## Scheduling behavior

- Input requests keep their original order. There is no sorting or bucketing by prompt length.
- Prefill packs consecutive requests until the configured input-token budget is full. Rows are packed without padding.
- Decode uses an independent generated-token budget. Each active sequence contributes one token per decode step.
- Prefill and decode orchestration and phase-specific kernels live in separate namespaces.

## Configuration

```bash
GETP_PREFILL_BATCH_TOKENS=65520
GETP_DECODE_BATCH_TOKENS=512
```

These are also the code defaults. The runtime clamps the prefill budget to available VRAM and clamps decode to the KV-cache capacity. `GETP_BATCH_SIZE` remains a legacy fallback for the decode budget.

## DSV2-Lite public evaluation

Dataset: 512 prompts, 123,392 input tokens, 64 requested output tokens per prompt.

| Prefill token budget | Decode token budget | Prefill chunks | End-to-end throughput |
|---:|---:|---:|---:|
| 65,520 | 256 | 2 | 493.36 token/s |
| 65,520 | 512 | 2 | **522.95 token/s** |

Best run details:

- Prefill: 43.635 s, 2,827.82 input token/s
- Decode: 18.713 s, 1,751.05 generated token/s
- End-to-end: 62.557 s, 522.95 generated token/s
- METEOR: 0.7166 (minimum: 0.3)
- BERTScore F1: 0.9463 (minimum: 0.8)

The outputs from prefill budgets 4,096 and 65,520, and decode budgets 256 and 512, were byte-identical. Increasing these budgets changed scheduling and throughput, not model output.

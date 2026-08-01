# Ming-Omni-TTS

[Ming-Omni-TTS-16.8B-A3B](https://huggingface.co/inclusionAI/Ming-omni-tts-16.8B-A3B)
is a mixture-of-experts audio generation model from inclusionAI. The current SGLang-Omni
serving path supports **text-to-speech** and **zero-shot voice cloning** through the
OpenAI-compatible `/v1/audio/speech` endpoint and produces **44.1 kHz** audio.

![Ming-Omni-TTS model architecture](https://github.com/inclusionAI/Ming-omni-tts/raw/main/figures/ming_omni_tts.png)

The serving pipeline keeps the SGLang autoregressive backbone and the Ming acoustic feedback
loop in one generation stage:

```text
preprocessing -> reference_encode -> tts_engine -> audio_decode
                                      |       ^
                                      +-------+
                                       latent feedback
```

`reference_encode` is a no-op for text-only requests. For voice cloning it extracts the speaker
embedding and prompt latents before the autoregressive loop starts. `tts_engine` runs the
SGLang backbone, FlowLoss/CFM acoustic tail, stop head, and feedback projection. `audio_decode`
converts the generated latent sequence into the final waveform with the Ming AudioVAE.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download
the checkpoint:

```bash
hf download inclusionAI/Ming-omni-tts-16.8B-A3B
```

The provided configuration is the recommended TP1 deployment and uses GPU 0.

## Server Configuration

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-omni-tts-16.8B-A3B \
  --config examples/configs/ming_omni_tts.yaml \
  --port 8000
```

The provided configuration enables the AR and acoustic-tail CUDA graphs. AudioVAE decode remains
eager. Streaming and prompt radix caching remain opt-in: requests are non-streaming unless
`stream` is set, and `disable_radix_cache` is `true` in the TTS engine configuration. To reuse
matching text or reference-conditioned prompt prefixes, set `disable_radix_cache` to `false`.
Generated acoustic history is never inserted into the radix tree. Reference encoding has a
separate content cache enabled by default; it is independent of the AR prompt radix cache.

## Synthesizing Speech

### Text Only

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni-tts",
    "input": "SGLang-Omni is a great project!",
    "response_format": "wav"
  }' \
  --output output.wav
```

### Voice Cloning

Ming-Omni-TTS currently accepts one local reference clip and requires its transcript. Start the
server with access to the directory containing the clip:

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-omni-tts-16.8B-A3B \
  --config examples/configs/ming_omni_tts.yaml \
  --allowed-local-media-path /path/to/references \
  --port 8000
```

Then submit the reference as a `file://` URL:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni-tts",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "file:///path/to/references/prompt.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "response_format": "wav"
  }' \
  --output cloned.wav
```

`ref_audio` and `ref_text` are accepted as shorthand for the single `references` item.

### Streaming

Streaming responses use raw signed 16-bit 44.1 kHz PCM. The decoder first consumes one latent
patch to prime AudioVAE lookahead, which normally emits no user-visible audio. It then decodes
groups of `audio_vae_steady_chunk_patches` patches; the provided configuration uses two. The
terminal step flushes any remaining patches immediately. This cadence is configured for the
pipeline rather than per request.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni-tts",
    "input": "SGLang-Omni supports streaming speech generation.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Non-empty text to synthesize |
| `references` | `null` | At most one local reference clip with non-empty `text` |
| `ref_audio` / `ref_text` | `null` | Shorthand for the reference clip and transcript |
| `max_new_tokens` | `200` | Maximum acoustic generation steps; the provided config caps this at `256` |
| `temperature` | `0.0` | Non-negative SDE temperature used by the FlowLoss sampler |
| `response_format` | `wav` | Use `pcm` when `stream` is enabled; `wav` is used by the reference benchmark |
| `stream` | `false` | Streams raw PCM audio when enabled |
| `voice` | `default` | Only the default voice selector is accepted |
| `speed` | `1.0` | Other speed values are not supported |

Advanced FlowLoss controls can be passed through `stage_params.tts_engine`:

```json
{
  "stage_params": {
    "tts_engine": {
      "cfg": 2.0,
      "sigma": 0.25,
      "temperature": 0.0
    }
  }
}
```

`cfg` must be positive and cannot equal `1.0`; `sigma` and `temperature` must be
non-negative.

## Benchmarking

The reference serving configuration uses Seed-TTS-Eval with concurrency 8. Run generation
against the existing Ming-TTS server and save the audio for a separate ASR pass:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url http://127.0.0.1:8000 \
  --model ming-omni-tts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir results/ming_tts_reference_en \
  --lang en --ref-format references \
  --max-new-tokens 256 --max-concurrency 8 --warmup 8
```

Use `--no-ref-audio` for text-only synthesis. Use `--lang zh` and a different output directory
for the Chinese split. Release the TTS server GPUs before starting the ASR server, then compute
WER from the saved audio:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --port 8100

python -m benchmarks.eval.benchmark_tts_seedtts \
  --transcribe-only --use-existing-server \
  --host 127.0.0.1 --port 8100 \
  --model ming-omni-tts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir results/ming_tts_reference_en \
  --lang en --ref-format references \
  --max-new-tokens 256
```

## Benchmark Results

### Recommended Single-H200 TP1

The recommended TP1 configuration was evaluated on **1× H200 141 GB** with concurrency 8,
eight warmup requests, and the full Seed-TTS-Eval EN and ZH splits. AR and acoustic-tail CUDA
graphs were enabled, while AudioVAE decode remained eager and prompt radix caching was disabled.

Streaming:

| Slice | Lang | Samples | Failed | Corpus WER | RTF Mean | Latency Mean (s) | First Audio Mean (s) | Throughput (qps) | Audio s/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 1088 | 0 | 0.93% | 0.3377 | 1.543 | 0.5285 | 5.174 | 24.428 |
| text-only | ZH | 2020 | 0 | 0.70% | 0.3368 | 1.659 | 0.5250 | 4.818 | 24.199 |
| reference | EN | 1088 | 0 | 1.03% | 0.3770 | 1.650 | 0.6327 | 4.844 | 22.038 |
| reference | ZH | 2020 | 0 | 0.71% | 0.3467 | 1.961 | 0.6188 | 4.076 | 23.365 |

Non-streaming:

| Slice | Lang | Samples | Failed | Corpus WER | RTF Mean | Latency Mean (s) | Throughput (qps) | Audio s/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 1088 | 0 | 0.95% | 0.2155 | 1.011 | 7.898 | 37.220 |
| text-only | ZH | 2020 | 0 | 0.69% | 0.2084 | 1.042 | 7.669 | 38.478 |
| reference | EN | 1088 | 0 | 1.17% | 0.2342 | 1.048 | 7.623 | 34.592 |
| reference | ZH | 2020 | 0 | 0.72% | 0.2005 | 1.148 | 6.958 | 39.986 |

All 12,432 requests completed successfully. Reference corpus WER includes a small
near-silent tail from unseeded acoustic sampling. Streaming returned its first audio
payload in 0.53-0.63 seconds, while non-streaming retained higher complete-response
throughput.

## Known Limitations

- **Serving optimizations.** Prompt radix caching is supported but disabled by default. When
  enabled, reuse is limited to the original prompt; generated acoustic history is not inserted
  into the cache. `torch.compile` has not yet been validated and remains disabled in the provided
  configuration.
- **Reference inputs.** The current request adapter accepts one local reference audio file with a
  non-empty transcript; remote URLs, data URLs, precomputed prompt latents, and speaker embeddings
  are not yet exposed.
- **Generation controls.** Request-local `seed`, logits sampling fields (`top_p`, `top_k`,
  `repetition_penalty`), named voices, explicit language selection, instructions, and duration
  control are not yet exposed. `initial_codec_chunk_frames` is rejected because AudioVAE cadence
  is a pipeline-level setting.
- **Checkpoint coverage.** The provided configuration targets the 16.8B-A3B checkpoint. A
  configuration for the 0.5B checkpoint has not yet been added.

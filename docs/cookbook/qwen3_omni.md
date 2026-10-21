# Qwen3-Omni

[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) is a multi-modal model
that accepts text, image, audio, and video input and can produce text-only or text + audio output.
This page covers every supported server configuration — use the generator to get the exact launch
command for your hardware, then check the tables to confirm your combination is supported.

## Prerequisites

```bash
docker pull hongccc/sglang-omni:dev
docker run -it --shm-size 32g --gpus all hongccc/sglang-omni:dev /bin/zsh
```

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install --prerelease=allow "sglang-omni==0.1.8"
```

See [Installation](../get_started/installation.md) for Docker digests and source installs.

## Server Configuration

Use the selector below to generate the exact launch command for your configuration.

```{raw} html
<div id="sgl-server-gen-mount"></div>
```

## Compatibility Matrix

Colocated topology requires `--config examples/configs/qwen3_omni_colocated_h20.yaml`
(or `qwen3_omni_colocated_h200.yaml` on H200) to set per-stage GPU memory budgets.

| Mode | Topology | Thinker TP | Precision | Status |
|---|---|---|---|---|
| Thinker-only | — | — | BF16 | ✅ |
| Thinker-only | — | — | FP8 | ✅ |
| Thinker-only | — | — | AutoRound INT4 | ✅ |
| Thinker-Talker | Disaggregated | TP=1 | BF16 | ✅ |
| Thinker-Talker | Disaggregated | TP=1 | FP8 | ✅ |
| Thinker-Talker | Disaggregated | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |
| Thinker-Talker | Disaggregated | TP=2 | BF16 | ✅ |
| Thinker-Talker | Disaggregated | TP=2 | FP8 | ✅ |
| Thinker-Talker | Disaggregated | TP=2 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |
| Thinker-Talker | Colocated | TP=1 | BF16 | ✅ |
| Thinker-Talker | Colocated | TP=1 | FP8 | ✅ |
| Thinker-Talker | Colocated | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |

## Input / Output Modalities

All input modality combinations work with both text-only and speech servers.
`modalities: ["text", "audio"]` requires a **speech-mode server** (omit `--text-only`).

| Input | Output | Speech server | Minimal request body | Notes |
|---|---|---|---|---|
| Text | Text | No | `{"messages": [{"role": "user", "content": "..."}], "modalities": ["text"]}` | — |
| Image + text | Text | No | `{"messages": [{"role": "user", "content": "..."}], "images": ["path/or/url"], "modalities": ["text"]}` | — |
| Audio | Text | No | `{"messages": [{"role": "user", "content": ""}], "audios": ["path/or/url"], "modalities": ["text"]}` | content must be "" when the query is spoken |
| Image + audio | Text | No | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text"]}` | content must be "" when the query is spoken |
| Image | Text | No | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "modalities": ["text"]}` | content must be "" when query comes from image |
| Video + text | Text | No | `{"messages": [{"role": "user", "content": "..."}], "videos": ["path/or/url"], "modalities": ["text"]}` | — |
| Video + audio | Text | No | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text"]}` | content must be "" when the query is spoken |
| Video | Text | No | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "modalities": ["text"]}` | content must be "" when query comes from video |
| Text | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": "..."}], "modalities": ["text", "audio"]}` | — |
| Image + text | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": "..."}], "images": ["path/or/url"], "modalities": ["text", "audio"]}` | — |
| Audio | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when the query is spoken |
| Image + audio | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when the query is spoken |
| Image | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when query comes from image |
| Video + text | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": "..."}], "videos": ["path/or/url"], "modalities": ["text", "audio"]}` | — |
| Video + audio | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when the query is spoken |
| Video | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when query comes from video |

### Sampling Parameters

Standard sampling parameters apply to the thinker stage. When `modalities` includes `"audio"`, the additional talker-specific parameters below control the speech generation independently.

| Parameter | Type | Default | Applies to |
|---|---|---|---|
| `temperature` | float | `0.0` (greedy) | Thinker |
| `top_p` | float | `1.0` | Thinker |
| `top_k` | int | `-1` | Thinker |
| `min_p` | float | `0.0` | Thinker |
| `repetition_penalty` | float | `1.0` | Thinker |
| `max_tokens` | int | `2048` | Thinker |
| `max_completion_tokens` | int | `null` | Thinker; OpenAI-compatible alias for `max_tokens` |
| `stop` | str \| list | `null` | Thinker |
| `seed` | int | `null` | Thinker |
| `stream` | bool | `false` | Both |
| `audio` | dict | `null` | Speech response format config, e.g. `{"format": "wav"}` |
| `talker_temperature` | float | `0.9` | Talker (audio output only) |
| `talker_top_p` | float | `1.0` | Talker (audio output only) |
| `talker_top_k` | int | `50` | Talker (audio output only) |
| `talker_repetition_penalty` | float | `1.05` | Talker (audio output only) |
| `talker_max_new_tokens` | int | `4096` | Talker (audio output only) |
| `stage_sampling` | dict | `null` | Per-stage sampling override |
| `stage_params` | dict | `null` | Per-stage non-sampling params |
| `video_fps` | float | `null` | Frame sampling rate for video input (uses server default if unset) |
| `video_max_frames` | int | `null` | Maximum number of frames sampled from a video |
| `video_min_pixels` | int | `null` | Minimum pixels per video frame |
| `video_max_pixels` | int | `null` | Maximum pixels per video frame |
| `video_total_pixels` | int | `null` | Total pixel budget across all video frames |
| `use_audio_in_video` | bool | `null` | Set to `true` to process embedded audio. Requests where all videos lack audio use video-only processing; mixing videos with and without audio is rejected. Audio decoding errors are reported. |

Non-streaming chat requests return HTTP 400 for invalid media, HTTP 503 when the request queue is full, and HTTP 500 for other
server failures. With `stream=true`, errors detected after the event stream
starts are sent as `data: {"error": {"message": "...", "type": "invalid_request_error", "code": 400}}`,
followed by `data: [DONE]`. A full request queue uses `type: "server_error"` and `code: 503`. Other server failures use `type: "server_error"` and
`code: 500`. The HTTP status remains 200 once streaming has started; clients
must check for an `error` event, including after partial output.

### Known Limitations

- **Multiple videos must have the same sampled frame rate.** The processor accepts one frame rate for video token timestamps. Requests with different sampled frame rates follow the invalid-input error contract above.

- **`modalities: ["text", "audio"]` has no effect on a text-only server.** No error is raised — the response simply contains no audio. Use a speech-mode server (without `--text-only`) to get audio output.
- **`content` must be `""` when the query is entirely in `audios`, `videos`, or `images`.** Leaving a text query in `content` alongside audio causes the model to process both, which is usually not what you want.
- **Colocated topology does not support `--thinker.tp_size 2`.** The server raises a `ValueError` at startup ("Qwen Phase 1 colocation does not support thinker TP"). Use disaggregated topology for TP=2.
- **Requests that exceed the model's context length are rejected with an error.** The preprocessor raises a `ValueError` when the prompt token count alone meets or exceeds `max_seq_len`, or when `prompt tokens + max_new_tokens ≥ max_seq_len`. Reduce input length or lower `max_tokens` to stay within the limit. Without `--preprocessing.factory.max_seq_len` and `--thinker.factory.max_seq_len`, the limit is the thinker's context in the checkpoint config (65,536 tokens for the Instruct checkpoint). Setting both to a lower value caps it. The talker's context is likewise the checkpoint's unless `--talker_ar.factory.max_seq_len` sets it. When a long spoken prompt leaves less room than `talker_max_new_tokens`, the talker generates only the frames that fit, so the audio can end before the text.

## Speculative Decoding

The thinker can verify tokens proposed by SpeechSpec's DFlash draft [`yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8`](https://huggingface.co/yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8) or EAGLE3 draft [`yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio`](https://huggingface.co/yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio).

```bash
# DFlash: one 8-token block per step
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --thinker.engine.speculative_algorithm DFLASH \
  --thinker.engine.speculative_draft_model_path yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8 \
  --thinker.engine.speculative_num_draft_tokens 8 \
  --port 8000

# EAGLE3: a 3-step chain
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --thinker.engine.speculative_algorithm EAGLE3 \
  --thinker.engine.speculative_draft_model_path yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio \
  --thinker.engine.speculative_num_steps 3 \
  --thinker.engine.speculative_eagle_topk 1 \
  --thinker.engine.speculative_num_draft_tokens 4 \
  --port 8000
```

- **Context.** The thinker serves within the draft's context (8,192 tokens for both drafts), and a request whose prompt plus `max_tokens` exceeds it, less a few tokens each step reserves, is rejected with an error. Setting `--preprocessing.factory.max_seq_len 8192` rejects most of them before the thinker.
- **Sampling.** Requests with `repetition_penalty`, `min_p` or logprobs are rejected, because the draft workers verify without them. Rollout `/generate` requests default to `return_logprob: true`, so they must send `false`.
- **Scheduling.** Async decode is turned off, and setting `enable_mixed_chunk` is rejected. EAGLE3 also runs prefill without CUDA graphs, since its draft extends on the composed multimodal embeddings. Sessions and weight updates are refused while a draft is loaded.

Text-only ASR on LibriSpeech test-clean (2,620 utterances, `--text-only`, thinker on one H100 80GB with FA3, greedy, `max_tokens 200`), in requests per second:

| Concurrency | Baseline | DFlash | EAGLE3 |
|---|---|---|---|
| 1 | 7.09 | 11.49 (1.62x) | 8.55 (1.21x) |
| 4 | 14.47 | 21.80 (1.51x) | 14.71 (1.02x) |
| 8 | 20.65 | 33.56 (1.63x) | 21.60 (1.05x) |

DFlash accepts 4.7 tokens per verify and EAGLE3 3.4. WER stays between 1.65% and 1.68%.

With speech output (Seed-TTS-Eval English, 1,088 prompts, concurrency 16, streaming; thinker on one H100, talker and code2wav on a second), DFlash raises throughput from 23.7 to 26.9 req/s (1.13x) and cuts mean latency from 0.67 to 0.59 s and time to first audio from 0.25 to 0.23 s, with WER unchanged (1.88% → 1.87%). The gain is smaller because the talker and code2wav, which the draft does not speed up, produce the audio.

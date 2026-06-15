# Agent runbook — pre-calibration environment verification

**Read this file before any `tune.py run`.** The repro container is already
running; this document covers **in-container** checks only (no Docker startup).

Branch `ci-host-profiles` (vs `main`) added:

| Change | Purpose |
|--------|---------|
| `hosts/<name>.yaml` | In-container `repo_root`, `venv_python`, cache paths |
| `tune.py --host` / autodetect | Maps `physical.*` → `HF_HOME`, `SEEDTTS_SIM_CACHE_DIR`, … |
| `tune.py hosts-list` | List host profiles |
| `precheck` speaker_sim line | Verifies WavLM assets before TTS similarity / Omni utmos |
| `default_venv_python` | `/data/chenyang/.python/omni/bin/python` on H20 profile |

**No symlinks.** If `auto env: HF_HOME=…` in precheck output matches
`physical.hf_hub` in the host YAML, paths are wired correctly.

---

## 0. Scope — what are you calibrating?

Confirm with the user (or `handoff:` in host YAML) before spending GPU hours:

| Goal | `--model` | Stages | Repeats |
|------|-----------|--------|---------|
| TTS CI only | `tts` | `ALL` | 5 (default) |
| Qwen3-Omni CI only | `qwen3-omni-v1` | `ALL` | 5 |
| **Full CI** | run **both** sequentially | each `ALL` | 5 each |

Full CI order: **TTS first**, then **Qwen3-Omni** (matches omni-ci DAG).

---

## 1. Load host profile (mandatory)

```bash
cd /data/chenyang/sglang-omni   # or host profile repo_root

python .claude/skills/tune-ci-thresholds/tune.py hosts-list
hostname   # should match a profile, e.g. sglang-h20-ci
```

**Selection:** `$TUNE_HOST` → `--host <name>` → autodetect by `hostname`.

Shipped profile **`sglang-h20-ci`** (`hosts/sglang-h20-ci.yaml`):

| Key | Path |
|-----|------|
| `repo_root` | `/data/chenyang/sglang-omni` |
| `venv_python` | `/data/chenyang/.python/omni/bin/python` |
| `physical.hf_hub` | `/root/.cache/huggingface` |
| `physical.speaker_sim` | `/root/.cache/huggingface/speaker_sim` |
| `physical.omni_ci_home` | `/github/home/calibration` |

User paths given in chat **override** the YAML.

---

## 2. Manual sanity checks (before `precheck`)

Run these once; fix only what fails. **Report gaps to the user first**
(`agent_policy.env_check: report_missing_first`) unless they asked you to fix
a specific item.

### 2.1 Repo + venv

```bash
HOST_ROOT=/data/chenyang/sglang-omni
VENV=/data/chenyang/.python/omni/bin/python

test -f "$HOST_ROOT/pyproject.toml" && echo OK repo
test -x "$VENV" && echo OK venv
$VENV -c "import torch, sglang, flashinfer; print('torch', torch.__version__); print('sglang', sglang.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
$VENV -c "import sglang_omni; print('sglang_omni OK')"
```

Expected pins (from `pyproject.toml`): **torch 2.11.0**, **sglang 0.5.12.post1**.

Sync code only (do **not** rebuild venv unless precheck proves corrupt):

```bash
source $($VENV -c "import sys; print(sys.executable)") 2>/dev/null || true
# or: source $(dirname $VENV)/activate
cd "$HOST_ROOT" && uv pip install -e .
```

### 2.2 OMNI slice directory (FlashInfer / torchinductor)

```bash
mkdir -p /github/home/calibration
$VENV -c "
import os
assert os.environ.get('HOME') or True  # tune.py sets HOME at runtime
omni = '/github/home/calibration'
for sub in ('.cache', '.torchinductor'):
    os.makedirs(f'{omni}/{sub}', exist_ok=True)
print('OK omni_ci_home slice dirs')
"
```

At **`tune.py run`** time, also:

```bash
source "$HOST_ROOT/.github/scripts/ci_env.sh"
$VENV -c "import os; assert os.environ['TORCHINDUCTOR_CACHE_DIR'].startswith(os.environ['OMNI_CI_HOME'])"
```

### 2.3 GPUs idle

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
# Calibration needs 2× H20, each ≤ 2048 MiB before each pytest (tune.py enforces at run)
```

Precheck does **not** kill GPU processes — user must free GPUs if busy.

### 2.4 HuggingFace hub cache (models + datasets)

Host profile sets `HF_HOME=/root/.cache/huggingface`. Quick listing:

```bash
ls /root/.cache/huggingface/hub | rg -i 'qwen3|higgs|seed-tts|video|mmmu|mmsu|marksverdhei' | head -20
```

**Required for `--model tts` precheck (all stages):**

| Kind | Repo ID |
|------|---------|
| model | `boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999` |
| model | `Qwen/Qwen3-ASR-1.7B` |
| dataset | `zhaochenyang20/seed-tts-eval-arrow` |

**Additional for `--model qwen3-omni-v1` precheck (all stages):**

| Kind | Repo ID |
|------|---------|
| model | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| model | `marksverdhei/Qwen3-Omni-30B-A3B-FP8` |
| dataset | `zhaochenyang20/mmsu-ci-2000` |
| dataset | `zhaochenyang20/mmmu-ci-50` |
| dataset | `zhaochenyang20/seed-tts-eval-50-arrow` |
| dataset | `zhaochenyang20/Video_MME_ci` |
| dataset | `zhaochenyang20/Video_AMME_ci` |

(Qwen3 also uses `Qwen/Qwen3-ASR-1.7B` for talker WER stages.)

If precheck prints ✗, run **only** the `huggingface-cli download …` lines precheck
prints — one repo at a time. Use `HF_ENDPOINT=https://hf-mirror.com`. For
private models, `source ~/.zshrc` first (`HF_TOKEN`).

### 2.5 Speaker Similarity assets (TTS + Qwen3)

Directory: `/root/.cache/huggingface/speaker_sim`

```bash
SIM=/root/.cache/huggingface/speaker_sim
ls -lah "$SIM/wavlm_large.pt" "$SIM/wavlm_large_finetune.pth" "$SIM/.complete"
$VENV -m benchmarks.metrics.speaker_similarity_assets --warm-cache
# Must end with: cache HIT at .../speaker_sim
```

Warm-cache env (also in host YAML):

```bash
source ~/.zshrc
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export SEEDTTS_SIM_CACHE_DIR=/root/.cache/huggingface/speaker_sim
cd /data/chenyang/sglang-omni
$VENV -m benchmarks.metrics.speaker_similarity_assets --warm-cache
```

### 2.6 Stage 11 only — `CAP_SYS_PTRACE` (runtime check)

Only needed for `videoamme_talker_tp2`. Skip for TTS-only calibration.

```bash
grep -q cap_sys_ptrace /proc/self/status && echo ptrace_ok || echo ptrace_MISSING
# Or: capsh --print | rg -i ptrace
```

If missing, stage 11 requests 500 — calibrate other stages first; user fixes
container capability outside this skill.

---

## 3. Official precheck (mandatory gate)

Run **both** models if doing full CI:

```bash
cd /data/chenyang/sglang-omni
export TUNE_HOST=sglang-h20-ci   # optional if hostname autodetects

python .claude/skills/tune-ci-thresholds/tune.py --model tts precheck \
  --output-dir /tmp/precheck_tts

python .claude/skills/tune-ci-thresholds/tune.py --model qwen3-omni-v1 precheck \
  --output-dir /tmp/precheck_qwen3
```

**Pass criteria — every line must be good:**

```
host: sglang-h20-ci (repo=...)
venv_python: ... [ok]
  sglang: 0.5.12.post1 (pin ...) [ok]
  torch: 2.11.0+cu130 (pin ...) [ok]
  auto env: HF_HOME=/root/.cache/huggingface          ← must match physical.hf_hub
  auto env: SEEDTTS_SIM_CACHE_DIR=/root/.cache/huggingface/speaker_sim
    ✓ model: ...
    ✓ dataset: ...
    ✓ speaker_sim: ... (wavlm_large.pt + wavlm_large_finetune.pth)
  GPUs: 2× NVIDIA H20 — 2/2 free

precheck OK
```

**Common false alarms:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| HF ✗ but files exist under `/root/.cache/huggingface` | Host profile not loaded | `--host sglang-h20-ci` or fix hostname |
| HF ✗, wrong `HF_HOME` in auto env | Same | Same |
| speaker_sim ✗ | Missing warm-cache | §2.5 |
| GPU busy | Other jobs | User frees GPUs |

Do **not** run `prepare_omni_venv.sh` or bulk `ensure_hf_models.sh` when precheck
is green or only reports one missing repo.

---

## 4. Optional smoke test (after precheck OK)

```bash
cd /data/chenyang/sglang-omni
source /data/chenyang/.python/omni/bin/activate
source .github/scripts/ci_env.sh
export GITHUB_ACTIONS=true RUNNER_TEMP=/tmp PYTHONPATH=$PWD
export NO_PROXY=localhost,127.0.0.1,::1
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_videomme_ci.py -v -s -x
```

Router/worker startup **< ~60s** after FlashInfer cold compile — if much slower,
fix env (`XDG_CACHE_HOME`, `HOME`) before calibration.

---

## 5. Ready for calibration

When §3 passes for every `--model` you will run:

1. Create run dir: `.tune-runs/<timestamp>_<model>_all_r5/`
2. Spawn **Tab A** (`tail_calibration_pytest.sh`) then **Tab B** (`tune.py run`)
3. Poll ≤120s; show **strict audit** ✓ counts, not only `ok/total`
4. Do not start `run` until user confirmed scope (unless already in `handoff`)

Update `handoff:` in `hosts/sglang-h20-ci.yaml` when pausing mid-calibration.

---

## 6. Forbidden before / during calibration

- Document or run `docker run` inside this skill
- Create symlinks for `HF_HOME` / `SEEDTTS_SIM_CACHE_DIR` when host profile is active
- `prepare_omni_venv.sh` / bulk downloads when precheck only shows specific ✗
- `tune.py run` with pytest `-x`
- Proceed while strict audit has △/✗ repeats
- Fix env without reporting to user first (unless user explicitly asked)

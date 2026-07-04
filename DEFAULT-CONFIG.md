# DEFAULT CONFIG — DS4-Flash + DSpark, verified live 2026-07-04 (2× DGX Spark)

This is the **canonical, verified-working config** — captured live from the running
deployment before teardown. Reproduce this exactly to get the benchmarked result.

**Verified:** TP=2 on Asusi (rank0/head) + Spark4 (rank1/worker), served `:8888`,
clean output (no garble), honest speed **~49 tok/s mixed / 54–60 structured-agentic**.

---

## Model + image
- **Model:** `fraserprice/DeepSeek-V4-Flash-DSpark` (in HF cache: `/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`)
- **Image (as deployed):** `vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b`
  (drowzeys/Keys overlay w/ Patch 1/2/2b concurrency). vLLM 0.21.1rc1, venv `/opt/env`, ENTRYPOINT=`["bash"]`.
- **Canonical image:** `vllm-dspark-runtime:dspark-nvfp4-stage-c` built via `./build-dspark-vllm-runtime.sh`
  — its overlay ALREADY contains Patch 3 (commit e83606a), so a fresh stage-c build needs NO bind-mount.
- **Patch 3 (roady001, issue #3 — cold-start garble root fix):** in a fresh stage-c build it's baked in.
  As-deployed on probe-c-p2b (which predates Patch 3) it was injected via bind-mount:
  `-v /var/tmp/patch3-scheduler.py:/opt/env/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:ro`
  (source = repo `recipe/overlay/vllm/v1/core/sched/scheduler.py`).

## Exact vLLM command (byte-for-byte, as running)
```
/opt/env/bin/vllm serve /cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark \
  --served-model-name deepseek-v4-flash-dspark \
  --host 0.0.0.0 --port 8888 \
  --trust-remote-code \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --kv-cache-dtype nvfp4_ds_mla \
  --block-size 256 \
  --max-model-len 350000 \
  --max-num-seqs 12 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 12 \
  --gpu-memory-utilization 0.80 \
  --enable-prefix-caching \
  --async-scheduling \
  --enable-chunked-prefill \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --default-chat-template-kwargs '{"thinking":false}' \
  --generation-config vllm \
  --enable-flashinfer-autotune \
  --nnodes 2 --node-rank <0|1> --master-addr 192.168.192.3 --master-port 25440
```
Key spec-decode choices (verified): **`num_speculative_tokens: 5`** (not 3 — Patch 3 makes the
higher depth garble-safe; +24% over 3) and **`draft_sample_method: probabilistic`** (BEATS greedy
for DSpark's calibrated draft heads: 49 vs 32 tok/s mixed — do NOT switch to greedy). No
`--override-generation-config`. `--max-cudagraph-capture-size` MUST equal `--max-num-seqs`.

## Runtime env (compose defaults — all verified)
```
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1  VLLM_TRITON_MLA_SPARSE=1  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0  VLLM_SKIP_INIT_MEMORY_CHECK=1  VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_USE_B12X_MOE=1  VLLM_USE_B12X_WO_PROJECTION=1  VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=0
VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=16  B12X_W4A16_TC_DECODE=0
VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0  VLLM_DSPARK_CONFIDENCE_SCHEDULER=off  VLLM_DSPARK_LOCAL_ARGMAX=1
VLLM_DSPARK_REPLICATE_MARKOV_W1=1  VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0  VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0  VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1
VLLM_DSV4_B12X_COMPRESSED_MLA=0  VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0  VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT=0
TORCH_CUDA_ARCH_LIST=12.1a  FLASHINFER_CUDA_ARCH_LIST=12.1a  FLASHINFER_DISABLE_VERSION_CHECK=1
TILELANG_CLEANUP_TEMP_FILES=1  DG_JIT_USE_NVRTC=0  DG_JIT_NVCC_COMPILER=/opt/env/bin/nvcc
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_NET=IB  NCCL_IB_DISABLE=0  NCCL_IB_HCA=rocep1s0f0  NCCL_SOCKET_IFNAME=enp1s0f0np0
NCCL_IB_GID_INDEX=3  NCCL_CROSS_NIC=1  NCCL_CUMEM_ENABLE=0  NCCL_IGNORE_CPU_AFFINITY=1
NCCL_DEBUG=WARN  NCCL_NVLS_ENABLE=0
HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1  HF_HUB_DISABLE_XET=1  HF_HOME=/cache/huggingface  VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache
```

## Node / network (2× DGX Spark over CRS812 switched fabric 192.168.192.0/24)
| role | node | user | fabric IP | tailnet | GID | HF cache |
|---|---|---|---|---|---|---|
| head (rank0) | Asusi | tonyspark3 | 192.168.192.3 | 100.90.25.78 | 3 | /home/tonyspark3/.cache/huggingface |
| worker (rank1, --headless) | Spark4 | tonyspark4 | 192.168.192.4 | 100.121.11.91 | 3 | /home/tonyspark4/.cache/huggingface |

MASTER_ADDR=192.168.192.3, MASTER_PORT=25440. Container: `network_mode: host`, `ipc: host`,
`shm_size: 64gb`, `gpus: all`, `-v /dev/infiniband:/dev/infiniband`, memlock unlimited.

## Reproduction (worker-first)
1. Ensure image on both nodes + model in HF cache (or build stage-c via `./build-dspark-vllm-runtime.sh`).
2. If using a pre-Patch-3 image (probe-c-p2b), stage `patch3-scheduler.py` at `/var/tmp/` on both nodes
   (`cp recipe/overlay/vllm/v1/core/sched/scheduler.py /var/tmp/patch3-scheduler.py`) + add the bind-mount.
   A fresh stage-c build skips this (Patch 3 already in the overlay).
3. Worker (Spark4): `cd <repo> && COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=1 HEADLESS=1 HF_CACHE=/home/tonyspark4/.cache/huggingface VLLM_HOST_IP=192.168.192.4 docker compose --env-file .env.dspark -f docker-compose.dspark.yml up -d`
4. Head (Asusi): same with `NODE_RANK=0 HEADLESS= VLLM_HOST_IP=192.168.192.3`.
5. ~4-5 min: weight-load + 2-node NCCL + KV profiling + CUDA-graph capture → serves `:8888`.
   Smoke: `curl :8888/v1/chat/completions ... "Reply with exactly: NVFP4 DSPARK OK"`.

## Honest benchmark (temp 0, authoritative completion_tokens/wall, 5 varied prompts)
| category | tok/s |
|---|---|
| Math | 60.1 |
| JSON | 54.0 |
| Code | 53.8 |
| Communication | 42.0 |
| Narrative | 33.7 |
| **mixed avg** | **48.7** |

Structured/agentic (JSON/code/math — the supervisor workload) = 54–60 tok/s. Deterministic
best-case ~75 (not representative). DSpark draft acceptance ~92% on structured, ~40% on creative.

## openclaw wiring (Mac)
A/B proxy `workspace/ds4-lb-proxy/proxy.py` A-backend → `http://100.90.25.78:8888`. Supervisors
flipped via `workspace/spark-swap/route-supervisors.py ds4` (target = `ds4-lb-proxy/deepseek-v4-flash-dspark`),
gateway `ai.openclaw.gateway` kickstarted. Fallback = 3090 27B.

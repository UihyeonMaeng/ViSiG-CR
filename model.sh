#!/usr/bin/env bash
# ViSiG-CR single-host/two-GPU model helper.
#
# Docker is expected to expose host GPUs 0 and 3:
#   docker run --gpus '"device=0,3"' ...
# The selected devices are normally enumerated in the container as cuda:0 and
# cuda:1. Always verify with `bash model.sh check-gpus`.
#
# Recommended role split:
#   cuda:0 : ViSiG-CR pipeline / MedSigLIP
#   cuda:1 : one vLLM server at a time (ontology OR attribute selector OR reader)
#
# A single config.yaml is used for every model variant. All ontology variants
# expose the API model name `ontology-extractor` on port 8001, and all reader
# variants expose `frozen-reader` on port 8002. Attribute Selector variants also
# use port 8002 but expose `attribute-selector`; stop the selector before starting
# a reader. Switching 8B/30B selector checkpoints never requires a config change.
#
# Examples:
#   bash model.sh check-gpus
#   bash model.sh download-main
#   bash model.sh start-ontology-extractor
#   bash model.sh stop
#   bash model.sh start-attribute-selector8
#   bash model.sh stop
#   bash model.sh start-reader30
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
MODEL_GPU="${MODEL_GPU:-1}"             # default container-local GPU ordinal
SERVICE_GPU_LIST="${MODEL_GPU}"          # comma-separated container-local ordinals
LLAMA_TP_GPU_LIST="${LLAMA_TP_GPU_LIST:-0,1}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
ONTOLOGY_PORT="${ONTOLOGY_PORT:-8001}"
READER_PORT="${READER_PORT:-8002}"
ONTOLOGY_SERVED_MODEL="${ONTOLOGY_SERVED_MODEL:-ontology-extractor}"
ATTRIBUTE_SELECTOR_SERVED_MODEL="${ATTRIBUTE_SELECTOR_SERVED_MODEL:-attribute-selector}"
READER_SERVED_MODEL="${READER_SERVED_MODEL:-frozen-reader}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/run/model_server}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/model_server}"
PID_FILE="${RUN_ROOT}/vllm.pid"
META_FILE="${RUN_ROOT}/vllm.meta"

ONTOLOGY_GPU_UTIL="${ONTOLOGY_GPU_UTIL:-0.90}"
ATTRIBUTE_SELECTOR30_GPU_UTIL="${ATTRIBUTE_SELECTOR30_GPU_UTIL:-0.92}"
ATTRIBUTE_SELECTOR8_GPU_UTIL="${ATTRIBUTE_SELECTOR8_GPU_UTIL:-0.88}"
READER30_GPU_UTIL="${READER30_GPU_UTIL:-0.92}"
READER8_GPU_UTIL="${READER8_GPU_UTIL:-0.88}"
LLAVA_MED_GPU_UTIL="${LLAVA_MED_GPU_UTIL:-0.88}"
LLAVA_MED_ALLOW_LEGACY_VLLM="${LLAVA_MED_ALLOW_LEGACY_VLLM:-0}"
ONTOLOGY_BACKEND="${ONTOLOGY_BACKEND:-qwen-vl8}"
ATTRIBUTE_SELECTOR_BACKEND="${ATTRIBUTE_SELECTOR_BACKEND:-qwen-vl8}"
ALLOW_EXPERIMENTAL_QWEN35="${ALLOW_EXPERIMENTAL_QWEN35:-0}"

if [[ "${MODEL_ROOT}" != /* ]]; then
  echo "ERROR: MODEL_ROOT must be an absolute path: ${MODEL_ROOT}" >&2
  exit 2
fi
MODEL_ROOT="$(realpath -m "${MODEL_ROOT}")"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

QWEN_TEXT_DIR="${MODEL_ROOT}/Qwen/Qwen3.5-27B"
QWEN_TEXT_INT4_DIR="${MODEL_ROOT}/Qwen/Qwen3.5-27B-GPTQ-Int4"
LLAMA_TEXT_DIR="${MODEL_ROOT}/meta-llama/Llama-3.3-70B-Instruct"
QWEN_VL30_DIR="${MODEL_ROOT}/Qwen/Qwen3-VL-30B-A3B-Instruct"
QWEN_VL8_DIR="${MODEL_ROOT}/Qwen/Qwen3-VL-8B-Instruct"
MEDSIGLIP_DIR="${MODEL_ROOT}/google/medsiglip-448"
LLAVA_MED_REPO_ID="microsoft/llava-med-v1.5-mistral-7b"
LLAVA_MED_OFFICIAL_DIR="${MODEL_ROOT}/${LLAVA_MED_REPO_ID}"
LLAVA_MED_DIR="${LLAVA_MED_DIR:-${LLAVA_MED_OFFICIAL_DIR}}"

if [[ "${LLAVA_MED_DIR}" != /* ]]; then
  echo "ERROR: LLAVA_MED_DIR must be an absolute path: ${LLAVA_MED_DIR}" >&2
  exit 2
fi
LLAVA_MED_DIR="$(realpath -m "${LLAVA_MED_DIR}")"

VLLM_BIN="${VLLM_BIN:-}"
MODEL_CMD=()
SERVICE_NAME=""
SERVICE_PORT=""
SERVICE_MODEL_DIR=""
SERVICE_API_MODEL=""
SERVICE_GPU_LIST="${MODEL_GPU}"

log() {
  printf '[model.sh] %s\n' "$*"
}

fail() {
  printf '[model.sh] ERROR: %s\n' "$*" >&2
  exit 2
}

resolve_vllm_bin() {
  if [[ -n "${VLLM_BIN}" ]]; then
    [[ -x "${VLLM_BIN}" ]] || fail "VLLM_BIN is not executable: ${VLLM_BIN}"
    return
  fi

  if [[ -x /opt/conda/envs/visig-cr-serve/bin/vllm ]]; then
    VLLM_BIN=/opt/conda/envs/visig-cr-serve/bin/vllm
  elif command -v vllm >/dev/null 2>&1; then
    VLLM_BIN="$(command -v vllm)"
  else
    fail "vllm executable not found. Create the visig-cr-serve environment or set VLLM_BIN."
  fi
}

require_hf_cli() {
  local hf_bin
  hf_bin="$(command -v hf 2>&1 || true)"
  [[ -n "${hf_bin}" ]] || fail "Install huggingface_hub first: python -m pip install -U huggingface_hub"
}

require_model_dir() {
  local path="$1"
  local name="$2"
  [[ -d "${path}" ]] || fail "${name} checkpoint directory not found: ${path}"
  [[ -f "${path}/config.json" ]] || fail "${name} config.json not found: ${path}/config.json"
}

validate_service_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is not available inside the container."

  local count
  count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${count}" =~ ^[0-9]+$ ]] || fail "Could not determine the number of visible GPUs."
  (( count > 0 )) || fail "No NVIDIA GPU is visible in this container."

  local gpu
  IFS=',' read -r -a gpu_list <<< "${SERVICE_GPU_LIST}"
  (( ${#gpu_list[@]} > 0 )) || fail "SERVICE_GPU_LIST is empty."
  for gpu in "${gpu_list[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || fail "Invalid container-local GPU ordinal: ${gpu}"
    (( gpu < count )) || fail "GPU ordinal ${gpu} is invalid; visible GPU count is ${count}."
  done
}

check_gpus() {
  echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo
  nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total --format=csv
  echo
  python - <<'PY' || true
try:
    import torch
except Exception as exc:
    print(f"PyTorch GPU check skipped: {exc}")
else:
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i} -> {torch.cuda.get_device_name(i)}")
PY
  echo
  echo "Expected role mapping for --gpus '\"device=0,3\"':"
  echo "  container cuda:0 -> host physical GPU 0  (pipeline / MedSigLIP)"
  echo "  container cuda:1 -> host physical GPU 3  (vLLM model server)"
}

show_config() {
  cat <<CFG
REPO_ROOT=${REPO_ROOT}
MODEL_ROOT=${MODEL_ROOT}
MODEL_GPU=${MODEL_GPU}    # default container-local ordinal
LLAMA_TP_GPU_LIST=${LLAMA_TP_GPU_LIST}
SERVER_HOST=${SERVER_HOST}
ONTOLOGY_PORT=${ONTOLOGY_PORT}
READER_PORT=${READER_PORT}
ONTOLOGY_SERVED_MODEL=${ONTOLOGY_SERVED_MODEL}
ATTRIBUTE_SELECTOR_SERVED_MODEL=${ATTRIBUTE_SELECTOR_SERVED_MODEL}
READER_SERVED_MODEL=${READER_SERVED_MODEL}
LLAVA_MED_OFFICIAL_DIR=${LLAVA_MED_OFFICIAL_DIR}
LLAVA_MED_DIR=${LLAVA_MED_DIR}
LLAVA_MED_ALLOW_LEGACY_VLLM=${LLAVA_MED_ALLOW_LEGACY_VLLM}
ONTOLOGY_BACKEND=${ONTOLOGY_BACKEND}
ATTRIBUTE_SELECTOR_BACKEND=${ATTRIBUTE_SELECTOR_BACKEND}
ALLOW_EXPERIMENTAL_QWEN35=${ALLOW_EXPERIMENTAL_QWEN35}
RUN_ROOT=${RUN_ROOT}
LOG_ROOT=${LOG_ROOT}
VLLM_BIN=${VLLM_BIN:-<auto-detect>}
CFG
}

llava_med_config_kind() {
  local config_path="$1"
  python - "${config_path}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        config = json.load(handle)
except (OSError, ValueError):
    print("invalid")
    raise SystemExit(0)

model_type = str(config.get("model_type", ""))
architectures = config.get("architectures", [])
if not isinstance(architectures, list):
    architectures = []

if model_type == "llava" and "LlavaForConditionalGeneration" in architectures:
    print("hf-llava")
elif model_type == "llava_mistral" or "LlavaMistralForCausalLM" in architectures:
    print("legacy-llava-mistral")
else:
    print("unknown")
PY
}

check_llava_med() {
  local config_path="${LLAVA_MED_DIR}/config.json"
  if [[ ! -f "${config_path}" ]]; then
    printf 'MISS %-36s %s\n' "LLaVA-Med-v1.5-Mistral-7B" "${LLAVA_MED_DIR}"
    return 1
  fi

  local kind
  kind="$(llava_med_config_kind "${config_path}")"
  case "${kind}" in
    hf-llava)
      printf 'OK   %-36s %s (HF LlavaForConditionalGeneration)\n' \
        "LLaVA-Med-v1.5-Mistral-7B" "${LLAVA_MED_DIR}"
      ;;
    legacy-llava-mistral)
      printf 'LEGACY %-32s %s\n' "LLaVA-Med-v1.5-Mistral-7B" "${LLAVA_MED_DIR}"
      printf '%s\n' \
        "The official checkpoint uses llava_mistral/LlavaMistralForCausalLM;" \
        "the pinned vLLM 0.11 reader requires an HF-converted LlavaForConditionalGeneration checkpoint."
      return 1
      ;;
    *)
      printf 'BAD  %-36s %s (unrecognized or invalid config.json)\n' \
        "LLaVA-Med-v1.5-Mistral-7B" "${LLAVA_MED_DIR}"
      return 1
      ;;
  esac
}

require_llava_med_vllm_compatibility() {
  local kind
  kind="$(llava_med_config_kind "${LLAVA_MED_DIR}/config.json")"
  if [[ "${kind}" == "hf-llava" ]]; then
    return 0
  fi
  if [[ "${kind}" == "legacy-llava-mistral" && "${LLAVA_MED_ALLOW_LEGACY_VLLM}" == "1" ]]; then
    log "WARNING: bypassing the legacy LLaVA-Med vLLM compatibility guard by explicit request."
    return 0
  fi

  if [[ "${kind}" == "legacy-llava-mistral" ]]; then
    fail "The official ${LLAVA_MED_REPO_ID} checkpoint uses the unsupported legacy llava_mistral/LlavaMistralForCausalLM layout. Set LLAVA_MED_DIR to a verified HF-converted LlavaForConditionalGeneration checkpoint. LLAVA_MED_ALLOW_LEGACY_VLLM=1 bypasses this guard only for an independently verified vLLM build."
  fi
  fail "LLaVA-Med config is not a supported HF LlavaForConditionalGeneration checkpoint: ${LLAVA_MED_DIR}/config.json"
}

check_models() {
  local failed=0
  local item path name shards
  for item in \
    "${LLAMA_TEXT_DIR}|Llama-3.3-70B-Instruct" \
    "${QWEN_VL30_DIR}|Qwen3-VL-30B-A3B-Instruct" \
    "${QWEN_VL8_DIR}|Qwen3-VL-8B-Instruct" \
    "${MEDSIGLIP_DIR}|MedSigLIP-448"
  do
    path="${item%%|*}"
    name="${item##*|}"
    if [[ -f "${path}/config.json" ]]; then
      shards="$(find "${path}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) | wc -l | tr -d ' ')"
      printf 'OK   %-36s %s (weight files: %s)\n' "${name}" "${path}" "${shards}"
    else
      printf 'MISS %-36s %s\n' "${name}" "${path}"
      failed=1
    fi
  done
  return "${failed}"
}

download_main_models() {
  require_hf_cli
  log "Run 'hf auth login' first; gated repositories require accepted access terms."
  mkdir -p "${MODEL_ROOT}/Qwen" "${MODEL_ROOT}/google" "${MODEL_ROOT}/meta-llama"

  hf download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir "${QWEN_VL30_DIR}"
  hf download Qwen/Qwen3-VL-8B-Instruct --local-dir "${QWEN_VL8_DIR}"
  hf download meta-llama/Llama-3.3-70B-Instruct --local-dir "${LLAMA_TEXT_DIR}"
  hf download google/medsiglip-448 --local-dir "${MEDSIGLIP_DIR}"
}

download_qwen35_experimental() {
  require_hf_cli
  log "Qwen3.5 requires a separate current/nightly vLLM environment; it is not part of the pinned vLLM 0.11.0 path."
  mkdir -p "${MODEL_ROOT}/Qwen"
  hf download Qwen/Qwen3.5-27B --local-dir "${QWEN_TEXT_DIR}"
  hf download Qwen/Qwen3.5-27B-GPTQ-Int4 --local-dir "${QWEN_TEXT_INT4_DIR}"
}

download_all_models() {
  download_main_models
  download_qwen35_experimental
}

download_llava_med() {
  require_hf_cli
  log "Downloading only the official LLaVA-Med-v1.5-Mistral-7B checkpoint."
  mkdir -p "$(dirname "${LLAVA_MED_OFFICIAL_DIR}")"
  hf download "${LLAVA_MED_REPO_ID}" --local-dir "${LLAVA_MED_OFFICIAL_DIR}"
  log "Downloaded to ${LLAVA_MED_OFFICIAL_DIR}"
  log "Note: the official legacy checkpoint must be HF-converted before the pinned vLLM 0.11 reader can serve it."
}

require_experimental_qwen35() {
  resolve_vllm_bin
  if [[ "${ALLOW_EXPERIMENTAL_QWEN35}" != "1" ]]; then
    fail "Qwen3.5 is disabled in the pinned vLLM 0.11.0 environment because that release does not register the Qwen3.5 architecture. Use ONTOLOGY_BACKEND=qwen-vl8 (default), qwen-vl30, llama-4bit, or llama-tp2. To use Qwen3.5, create a separate current/nightly vLLM environment and set ALLOW_EXPERIMENTAL_QWEN35=1 explicitly."
  fi
  log "WARNING: launching Qwen3.5 through an explicitly acknowledged experimental vLLM environment."
}

prepare_qwen_vl8_ontology() {
  SERVICE_GPU_LIST="${MODEL_GPU}"
  resolve_vllm_bin
  require_model_dir "${QWEN_VL8_DIR}" "Qwen3-VL-8B-Instruct"
  SERVICE_NAME=ontology-qwen-vl8
  SERVICE_PORT="${ONTOLOGY_PORT}"
  SERVICE_MODEL_DIR="${QWEN_VL8_DIR}"
  SERVICE_API_MODEL="${ONTOLOGY_SERVED_MODEL}"
  MODEL_CMD=(
    "${VLLM_BIN}" serve "${QWEN_VL8_DIR}"
    --served-model-name "${ONTOLOGY_SERVED_MODEL}"
    --host "${SERVER_HOST}" --port "${ONTOLOGY_PORT}"
    --tensor-parallel-size 1
    --dtype bfloat16
    --max-model-len 16384
    --max-num-seqs 4
    --gpu-memory-utilization "${ONTOLOGY_GPU_UTIL}"
    --limit-mm-per-prompt '{"image":0,"video":0}'
    --structured-outputs-config \
    '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}'
  )
}

prepare_qwen_vl30_ontology() {
  SERVICE_GPU_LIST="${MODEL_GPU}"
  resolve_vllm_bin
  require_model_dir "${QWEN_VL30_DIR}" "Qwen3-VL-30B-A3B-Instruct"
  SERVICE_NAME=ontology-qwen-vl30
  SERVICE_PORT="${ONTOLOGY_PORT}"
  SERVICE_MODEL_DIR="${QWEN_VL30_DIR}"
  SERVICE_API_MODEL="${ONTOLOGY_SERVED_MODEL}"
  MODEL_CMD=(
    "${VLLM_BIN}" serve "${QWEN_VL30_DIR}"
    --served-model-name "${ONTOLOGY_SERVED_MODEL}"
    --host "${SERVER_HOST}" --port "${ONTOLOGY_PORT}"
    --tensor-parallel-size 1
    --dtype bfloat16
    --max-model-len 16384
    --max-num-seqs 2
    --gpu-memory-utilization "${ONTOLOGY_GPU_UTIL}"
    --limit-mm-per-prompt '{"image":0,"video":0}'
    --structured-outputs-config \
    '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}'
  )
}

prepare_default_ontology() {
  case "${ONTOLOGY_BACKEND}" in
    qwen-vl8|qwen_vl8|qwen8) prepare_qwen_vl8_ontology ;;
    qwen-vl30|qwen_vl30|qwen30) prepare_qwen_vl30_ontology ;;
    llama-4bit|llama_4bit|llama) prepare_llama_ontology_4bit ;;
    llama-tp2|llama_tp2) prepare_llama_ontology_tp2 ;;
    qwen35|qwen3.5|qwen-3.5)
      require_experimental_qwen35
      prepare_qwen_ontology
      ;;
    qwen35-int4|qwen3.5-int4)
      require_experimental_qwen35
      prepare_qwen_ontology_int4
      ;;
    *)
      fail "Unknown ONTOLOGY_BACKEND=${ONTOLOGY_BACKEND}. Use qwen-vl8, qwen-vl30, llama-4bit, llama-tp2, or an explicitly enabled qwen35 variant."
      ;;
  esac
}

prepare_qwen_ontology() {
  require_experimental_qwen35
  SERVICE_GPU_LIST="${MODEL_GPU}"
  resolve_vllm_bin
  require_model_dir "${QWEN_TEXT_DIR}" "Qwen3.5-27B"
  SERVICE_NAME=ontology-qwen
  SERVICE_PORT="${ONTOLOGY_PORT}"
  SERVICE_MODEL_DIR="${QWEN_TEXT_DIR}"
  SERVICE_API_MODEL="${ONTOLOGY_SERVED_MODEL}"
  MODEL_CMD=(
    "${VLLM_BIN}" serve "${QWEN_TEXT_DIR}"
    --served-model-name "${ONTOLOGY_SERVED_MODEL}"
    --host "${SERVER_HOST}" --port "${ONTOLOGY_PORT}"
    --tensor-parallel-size 1
    --dtype bfloat16
    --max-model-len 16384
    --max-num-seqs 4
    --gpu-memory-utilization "${ONTOLOGY_GPU_UTIL}"
    --reasoning-parser qwen3
    # --default-chat-template-kwargs '{"enable_thinking":false}'
    # --language-model-only
    --limit-mm-per-prompt '{"image":0,"video":0}'
    --structured-outputs-config \
    '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}'
  )
}

prepare_qwen_ontology_int4() {
  require_experimental_qwen35
  SERVICE_GPU_LIST="${MODEL_GPU}"
  resolve_vllm_bin
  require_model_dir "${QWEN_TEXT_INT4_DIR}" "Qwen3.5-27B-GPTQ-Int4"
  SERVICE_NAME=ontology-qwen-int4
  SERVICE_PORT="${ONTOLOGY_PORT}"
  SERVICE_MODEL_DIR="${QWEN_TEXT_INT4_DIR}"
  SERVICE_API_MODEL="${ONTOLOGY_SERVED_MODEL}"
  MODEL_CMD=(
    "${VLLM_BIN}" serve "${QWEN_TEXT_INT4_DIR}"
    --served-model-name "${ONTOLOGY_SERVED_MODEL}"
    --host "${SERVER_HOST}" --port "${ONTOLOGY_PORT}"
    --tensor-parallel-size 1
    --dtype bfloat16
    --max-model-len 16384
    --max-num-seqs 2
    --gpu-memory-utilization "${ONTOLOGY_GPU_UTIL}"
    --reasoning-parser qwen3
    # --default-chat-template-kwargs '{"enable_thinking":false}'
    # --language-model-only
    --limit-mm-per-prompt '{"image":0,"video":0}'
    --structured-outputs-config \
    '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}'
  )
}

prepare_llama_ontology_4bit() {
  SERVICE_GPU_LIST="${MODEL_GPU}"
  resolve_vllm_bin
  require_model_dir "${LLAMA_TEXT_DIR}" "Llama-3.3-70B-Instruct"
  SERVICE_NAME=ontology-llama
  SERVICE_PORT="${ONTOLOGY_PORT}"
  SERVICE_MODEL_DIR="${LLAMA_TEXT_DIR}"
  SERVICE_API_MODEL="${ONTOLOGY_SERVED_MODEL}"
  MODEL_CMD=(
    "${VLLM_BIN}" serve "${LLAMA_TEXT_DIR}"
    --served-model-name "${ONTOLOGY_SERVED_MODEL}"
    --host "${SERVER_HOST}" --port "${ONTOLOGY_PORT}"
    --tensor-parallel-size 1
    --dtype bfloat16
    --quantization bitsandbytes
    --max-model-len 16384
    --max-num-seqs 2
    --gpu-memory-utilization "${ONTOLOGY_GPU_UTIL}"
    --structured-outputs-config \
    '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}'
  )
}

prepare_llama_ontology_tp2() {
  SERVICE_GPU_LIST="${LLAMA_TP_GPU_LIST}"
  resolve_vllm_bin
  require_model_dir "${LLAMA_TEXT_DIR}" "Llama-3.3-70B-Instruct"
  SERVICE_NAME=ontology-llama-tp2
  SERVICE_PORT="${ONTOLOGY_PORT}"
  SERVICE_MODEL_DIR="${LLAMA_TEXT_DIR}"
  SERVICE_API_MODEL="${ONTOLOGY_SERVED_MODEL}"
  MODEL_CMD=(
    "${VLLM_BIN}" serve "${LLAMA_TEXT_DIR}"
    --served-model-name "${ONTOLOGY_SERVED_MODEL}"
    --host "${SERVER_HOST}" --port "${ONTOLOGY_PORT}"
    --tensor-parallel-size 2
    --dtype bfloat16
    --max-model-len 16384
    --max-num-seqs 1
    --gpu-memory-utilization 0.92
    --structured-outputs-config \
    '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}'
  )
}

prepare_attribute_selector8() {
  SERVICE_GPU_LIST="${MODEL_GPU}"
  resolve_vllm_bin
  require_model_dir "${QWEN_VL8_DIR}" "Qwen3-VL-8B-Instruct"
  SERVICE_NAME=attribute-selector-qwen8
  SERVICE_PORT="${READER_PORT}"
  SERVICE_MODEL_DIR="${QWEN_VL8_DIR}"
  SERVICE_API_MODEL="${ATTRIBUTE_SELECTOR_SERVED_MODEL}"
  MODEL_CMD=(
    "${VLLM_BIN}" serve "${QWEN_VL8_DIR}"
    --served-model-name "${ATTRIBUTE_SELECTOR_SERVED_MODEL}"
    --host "${SERVER_HOST}" --port "${READER_PORT}"
    --tensor-parallel-size 1
    --dtype bfloat16
    --max-model-len 16384
    --max-num-seqs 4
    --gpu-memory-utilization "${ATTRIBUTE_SELECTOR8_GPU_UTIL}"
    --limit-mm-per-prompt '{"image":0,"video":0}'
    --structured-outputs-config \
    '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}'
  )
}

prepare_attribute_selector30() {
  SERVICE_GPU_LIST="${MODEL_GPU}"
  resolve_vllm_bin
  require_model_dir "${QWEN_VL30_DIR}" "Qwen3-VL-30B-A3B-Instruct"
  SERVICE_NAME=attribute-selector-qwen30
  SERVICE_PORT="${READER_PORT}"
  SERVICE_MODEL_DIR="${QWEN_VL30_DIR}"
  SERVICE_API_MODEL="${ATTRIBUTE_SELECTOR_SERVED_MODEL}"
  MODEL_CMD=(
    "${VLLM_BIN}" serve "${QWEN_VL30_DIR}"
    --served-model-name "${ATTRIBUTE_SELECTOR_SERVED_MODEL}"
    --host "${SERVER_HOST}" --port "${READER_PORT}"
    --tensor-parallel-size 1
    --dtype bfloat16
    --max-model-len 16384
    --max-num-seqs 2
    --gpu-memory-utilization "${ATTRIBUTE_SELECTOR30_GPU_UTIL}"
    --limit-mm-per-prompt '{"image":0,"video":0}'
    --structured-outputs-config \
    '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}'
  )
}

prepare_default_attribute_selector() {
  case "${ATTRIBUTE_SELECTOR_BACKEND}" in
    qwen-vl8|qwen_vl8|qwen8|8|8b) prepare_attribute_selector8 ;;
    qwen-vl30|qwen_vl30|qwen30|30|30b) prepare_attribute_selector30 ;;
    *)
      fail "Unknown ATTRIBUTE_SELECTOR_BACKEND=${ATTRIBUTE_SELECTOR_BACKEND}. Use qwen-vl8 or qwen-vl30."
      ;;
  esac
}

prepare_reader30() {
  SERVICE_GPU_LIST="${MODEL_GPU}"
  resolve_vllm_bin
  require_model_dir "${QWEN_VL30_DIR}" "Qwen3-VL-30B-A3B-Instruct"
  SERVICE_NAME=reader-qwen30
  SERVICE_PORT="${READER_PORT}"
  SERVICE_MODEL_DIR="${QWEN_VL30_DIR}"
  SERVICE_API_MODEL="${READER_SERVED_MODEL}"
  MODEL_CMD=(
    "${VLLM_BIN}" serve "${QWEN_VL30_DIR}"
    --served-model-name "${READER_SERVED_MODEL}"
    --host "${SERVER_HOST}" --port "${READER_PORT}"
    --tensor-parallel-size 1
    --dtype bfloat16
    --max-model-len 16384
    --max-num-seqs 1
    --gpu-memory-utilization "${READER30_GPU_UTIL}"
    --limit-mm-per-prompt '{"image":7,"video":0}'
    --mm-processor-cache-gb 0
    --structured-outputs-config \
    '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}'
  )
}

prepare_reader8() {
  SERVICE_GPU_LIST="${MODEL_GPU}"
  resolve_vllm_bin
  require_model_dir "${QWEN_VL8_DIR}" "Qwen3-VL-8B-Instruct"
  SERVICE_NAME=reader-qwen8
  SERVICE_PORT="${READER_PORT}"
  SERVICE_MODEL_DIR="${QWEN_VL8_DIR}"
  SERVICE_API_MODEL="${READER_SERVED_MODEL}"
  MODEL_CMD=(
    "${VLLM_BIN}" serve "${QWEN_VL8_DIR}"
    --served-model-name "${READER_SERVED_MODEL}"
    --host "${SERVER_HOST}" --port "${READER_PORT}"
    --tensor-parallel-size 1
    --dtype bfloat16
    --max-model-len 16384
    --max-num-seqs 1
    --gpu-memory-utilization "${READER8_GPU_UTIL}"
    --limit-mm-per-prompt '{"image":7,"video":0}'
    --mm-processor-cache-gb 0
    --structured-outputs-config \
    '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}'
  )
}

prepare_reader_llava_med() {
  SERVICE_GPU_LIST="${MODEL_GPU}"
  resolve_vllm_bin
  require_model_dir "${LLAVA_MED_DIR}" "LLaVA-Med-v1.5-Mistral-7B"
  require_llava_med_vllm_compatibility
  SERVICE_NAME=reader-llava-med
  SERVICE_PORT="${READER_PORT}"
  SERVICE_MODEL_DIR="${LLAVA_MED_DIR}"
  SERVICE_API_MODEL="${READER_SERVED_MODEL}"
  MODEL_CMD=(
    "${VLLM_BIN}" serve "${LLAVA_MED_DIR}"
    --served-model-name "${READER_SERVED_MODEL}"
    --host "${SERVER_HOST}" --port "${READER_PORT}"
    --tensor-parallel-size 1
    --dtype bfloat16
    --max-model-len 8192
    --max-num-seqs 1
    --gpu-memory-utilization "${LLAVA_MED_GPU_UTIL}"
    --limit-mm-per-prompt '{"image":7,"video":0}'
    --structured-outputs-config \
    '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}'
  )
}

pid_is_running() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

load_meta() {
  if [[ -f "${META_FILE}" ]]; then
    # This file is generated only by this script with shell-escaped values.
    # shellcheck disable=SC1090
    source "${META_FILE}"
  fi
}

stop_model() {
  if ! pid_is_running; then
    rm -f "${PID_FILE}" "${META_FILE}"
    log "No managed vLLM server is running."
    return 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  load_meta
  log "Stopping ${SERVICE_NAME:-vLLM} (pid=${pid}) ..."

  # The server is started in its own session, so terminate the whole process group.
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  for _ in $(seq 1 60); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${PID_FILE}" "${META_FILE}"
      log "Stopped."
      return 0
    fi
    sleep 1
  done

  log "Graceful stop timed out; sending SIGKILL."
  kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  rm -f "${PID_FILE}" "${META_FILE}"
}

health_host() {
  if [[ "${SERVER_HOST}" == "0.0.0.0" || "${SERVER_HOST}" == "::" ]]; then
    printf '127.0.0.1'
  else
    printf '%s' "${SERVER_HOST}"
  fi
}

wait_for_server() {
  local pid="$1"
  local port="$2"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  local host
  host="$(health_host)"
  local url="http://${host}:${port}/v1/models"

  log "Waiting up to ${STARTUP_TIMEOUT}s for ${url} ..."
  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 1
    fi
    if python - "${url}" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
url = sys.argv[1]
with urllib.request.urlopen(url, timeout=3) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 5
  done
  return 1
}

write_meta() {
  local pid="$1"
  local log_file="$2"
  {
    printf 'SERVICE_NAME=%q\n' "${SERVICE_NAME}"
    printf 'SERVICE_PORT=%q\n' "${SERVICE_PORT}"
    printf 'SERVICE_MODEL_DIR=%q\n' "${SERVICE_MODEL_DIR}"
    printf 'SERVICE_API_MODEL=%q\n' "${SERVICE_API_MODEL}"
    printf 'MODEL_GPU=%q\n' "${MODEL_GPU}"
    printf 'SERVICE_GPU_LIST=%q\n' "${SERVICE_GPU_LIST}"
    printf 'LOG_FILE=%q\n' "${log_file}"
    printf 'STARTED_AT=%q\n' "$(date -Iseconds)"
    printf 'MANAGED_PID=%q\n' "${pid}"
  } > "${META_FILE}"
}

start_prepared_model() {
  validate_service_gpus

  # Preserve the newly prepared command. Loading metadata for the currently
  # running service would otherwise overwrite SERVICE_NAME/SERVICE_PORT.
  local next_service_name="${SERVICE_NAME}"
  local next_service_port="${SERVICE_PORT}"
  local next_service_model_dir="${SERVICE_MODEL_DIR}"
  local next_service_api_model="${SERVICE_API_MODEL}"
  local next_service_gpu_list="${SERVICE_GPU_LIST}"
  local -a next_model_cmd=("${MODEL_CMD[@]}")

  if pid_is_running; then
    load_meta
    log "A managed server (${SERVICE_NAME:-unknown}) is already running; switching models."
    stop_model
  else
    rm -f "${PID_FILE}" "${META_FILE}"
  fi

  SERVICE_NAME="${next_service_name}"
  SERVICE_PORT="${next_service_port}"
  SERVICE_MODEL_DIR="${next_service_model_dir}"
  SERVICE_API_MODEL="${next_service_api_model}"
  SERVICE_GPU_LIST="${next_service_gpu_list}"
  MODEL_CMD=("${next_model_cmd[@]}")

  local log_file="${LOG_ROOT}/${SERVICE_NAME}.log"
  log "Starting ${SERVICE_NAME} on container GPU list ${SERVICE_GPU_LIST} (port ${SERVICE_PORT})"
  log "Log: ${log_file}"

  # setsid creates a process group so stop can clean up vLLM worker processes too.
  nohup setsid env CUDA_VISIBLE_DEVICES="${SERVICE_GPU_LIST}" "${MODEL_CMD[@]}" >"${log_file}" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" > "${PID_FILE}"
  write_meta "${pid}" "${log_file}"

  if wait_for_server "${pid}" "${SERVICE_PORT}"; then
    log "Ready: http://127.0.0.1:${SERVICE_PORT}/v1/models"
  else
    echo >&2
    log "Server failed to become ready. Last log lines:"
    tail -n 120 "${log_file}" >&2 || true
    stop_model || true
    return 1
  fi
}

serve_prepared_foreground() {
  validate_service_gpus
  log "Foreground mode: ${SERVICE_NAME} on container GPU list ${SERVICE_GPU_LIST}"
  exec env CUDA_VISIBLE_DEVICES="${SERVICE_GPU_LIST}" "${MODEL_CMD[@]}"
}

status_model() {
  if pid_is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    load_meta
    cat <<STATUS
status=running
service=${SERVICE_NAME:-unknown}
pid=${pid}
port=${SERVICE_PORT:-unknown}
model_gpu=${MODEL_GPU:-unknown}
service_gpu_list=${SERVICE_GPU_LIST:-unknown}
model_dir=${SERVICE_MODEL_DIR:-unknown}
api_model=${SERVICE_API_MODEL:-unknown}
log=${LOG_FILE:-unknown}
started_at=${STARTED_AT:-unknown}
STATUS
    echo
    ps -o pid,ppid,stat,etime,cmd -p "${pid}" || true
  else
    rm -f "${PID_FILE}" "${META_FILE}"
    echo "status=stopped"
  fi
}

show_logs() {
  load_meta
  local log_file="${LOG_FILE:-}"
  if [[ -z "${log_file}" || ! -f "${log_file}" ]]; then
    local newest
    newest="$(find "${LOG_ROOT}" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
    log_file="${newest}"
  fi
  [[ -n "${log_file}" && -f "${log_file}" ]] || fail "No model-server log file found in ${LOG_ROOT}"
  log "Following ${log_file}; press Ctrl-C to exit."
  tail -n 200 -f "${log_file}"
}

health_check() {
  local port="${1:-}"
  if [[ -z "${port}" ]]; then
    load_meta
    port="${SERVICE_PORT:-}"
  fi
  [[ -n "${port}" ]] || fail "No port supplied and no managed service metadata found."
  local host
  host="$(health_host)"
  python - "http://${host}:${port}/v1/models" <<'PY'
import json
import sys
import urllib.request
url = sys.argv[1]
with urllib.request.urlopen(url, timeout=10) as response:
    payload = json.load(response)
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
}

usage() {
  cat <<'USAGE'
Usage: bash model.sh <command>

Inspection
  check-gpus                  Show GPUs visible inside the container
  show-config                 Show resolved paths/ports/GPU assignment
  check-models                Verify main local checkpoint directories
  check-llava-med             Check LLaVA-Med presence and vLLM layout
  status                      Show managed vLLM process status
  health [port]               Query /v1/models
  logs                        Follow the current/latest server log

Download
  download-main               Llama-3.3-70B, Qwen3-VL 30B/8B, MedSigLIP
  download-all                Main models plus experimental Qwen3.5 checkpoints
  download-qwen35-experimental Qwen3.5 full/INT4 for a separate nightly vLLM env
  download-llava-med          Only official LLaVA-Med-v1.5-Mistral-7B

Background server (automatically stops the previous managed server)
  start-ontology-extractor    Use ONTOLOGY_BACKEND (default: qwen-vl8)
  start-qwen-vl8-ontology     Qwen3-VL-8B text-only extraction on port 8001
  start-qwen-vl30-ontology    Qwen3-VL-30B text-only extraction on port 8001
  start-llama-ontology-4bit   Llama-3.3-70B bitsandbytes 4-bit on one GPU
  start-llama-ontology-tp2    Llama-3.3-70B BF16 across cuda:0,1 (exclusive)
  start-qwen35-ontology       Experimental only; requires separate current/nightly vLLM
  start-attribute-selector    Use ATTRIBUTE_SELECTOR_BACKEND (default: qwen-vl8)
  start-attribute-selector8   Qwen3-VL-8B text-only selector as attribute-selector@8002
  start-attribute-selector30  Qwen3-VL-30B text-only selector as attribute-selector@8002
  start-reader30              Qwen3-VL-30B as frozen-reader on port 8002
  start-reader8               Qwen3-VL-8B as frozen-reader on port 8002
  start-reader-llava-med      LLaVA-Med HF conversion as frozen-reader
  start-reader [30|8|llava-med] Generic reader selector; default is 30
  stop                        Stop the managed vLLM server and workers

Foreground server (debugging)
  serve-ontology-extractor
  serve-qwen-vl8-ontology
  serve-qwen-vl30-ontology
  serve-llama-ontology-4bit
  serve-llama-ontology-tp2
  serve-qwen35-ontology
  serve-attribute-selector
  serve-attribute-selector8
  serve-attribute-selector30
  serve-reader30
  serve-reader8
  serve-reader-llava-med
  serve-reader [30|8|llava-med]

Important:
  - MODEL_GPU is a container-local ordinal. With Docker device=0,3, use
    MODEL_GPU=1 for the second visible device; verify the actual mapping first.
  - Every reader variant uses the same API identity: frozen-reader@127.0.0.1:8002.
  - Attribute Selector uses attribute-selector@127.0.0.1:8002 and is text-only.
    It shares the managed GPU/port with the reader, so stop it before reader inference.
  - The official LLaVA-Med download has a legacy llava_mistral layout. The
    vLLM 0.11 path requires an explicitly supplied HF-compatible conversion.
  - Every ontology variant uses ontology-extractor@127.0.0.1:8001.
  - Pass the same backend identity to `run_pipeline.py extract-ontology
    --ontology-backend ...`; the shared API model name cannot prove which
    checkpoint generated an existing events file.
  - Pinned vLLM 0.11.0 supports Qwen3-VL/Llama/LLaVA architectures but not
    Qwen3.5. Qwen3.5 is guarded unless ALLOW_EXPERIMENTAL_QWEN35=1 is set in a
    separately verified current/nightly vLLM environment.
  - ONTOLOGY_BACKEND accepts qwen-vl8, qwen-vl30, llama-4bit, or llama-tp2.
  - ATTRIBUTE_SELECTOR_BACKEND accepts qwen-vl8 (default) or qwen-vl30.
  - Llama 4-bit requires bitsandbytes and still needs enough GPU memory for KV
    cache/runtime overhead; Llama BF16 TP2 uses both cuda:0 and cuda:1.
USAGE
}

command="${1:-}"
case "${command}" in
  check-gpus) check_gpus ;;
  show-config) show_config ;;
  check-models) check_models ;;
  check-llava-med) check_llava_med ;;
  download-main) download_main_models ;;
  download-llava-med) download_llava_med ;;
  download-qwen35-experimental) download_qwen35_experimental ;;
  download|download-all) download_all_models ;;
  status) status_model ;;
  health) health_check "${2:-}" ;;
  logs) show_logs ;;
  stop) stop_model ;;

  start-ontology-extractor|start-ontology)
    prepare_default_ontology
    start_prepared_model
    ;;
  start-qwen-vl8-ontology|start-ontology-qwen-vl8)
    prepare_qwen_vl8_ontology
    start_prepared_model
    ;;
  start-qwen-vl30-ontology|start-ontology-qwen-vl30)
    prepare_qwen_vl30_ontology
    start_prepared_model
    ;;
  start-qwen35-ontology|start-qwen-ontology)
    prepare_qwen_ontology
    start_prepared_model
    ;;
  start-qwen35-ontology-int4|start-ontology-int4|start-qwen-ontology-int4)
    prepare_qwen_ontology_int4
    start_prepared_model
    ;;
  start-llama-ontology-4bit)
    prepare_llama_ontology_4bit
    start_prepared_model
    ;;
  start-llama-ontology-tp2)
    prepare_llama_ontology_tp2
    start_prepared_model
    ;;
  start-attribute-selector)
    prepare_default_attribute_selector
    start_prepared_model
    ;;
  start-attribute-selector8|start-qwen-vl8-attribute-selector)
    prepare_attribute_selector8
    start_prepared_model
    ;;
  start-attribute-selector30|start-qwen-vl30-attribute-selector)
    prepare_attribute_selector30
    start_prepared_model
    ;;
  start-reader)
    case "${2:-30}" in
      30|30b|qwen30) prepare_reader30 ;;
      8|8b|qwen8) prepare_reader8 ;;
      llava|llava-med|7|7b) prepare_reader_llava_med ;;
      *) fail "Unknown reader variant: ${2}. Use 30, 8, or llava-med." ;;
    esac
    start_prepared_model
    ;;
  start-reader30|start-qwen-vl30)
    prepare_reader30
    start_prepared_model
    ;;
  start-reader8|start-qwen-vl8)
    prepare_reader8
    start_prepared_model
    ;;
  start-reader-llava-med|start-llava-med-reader)
    prepare_reader_llava_med
    start_prepared_model
    ;;

  serve-ontology-extractor|serve-ontology)
    prepare_default_ontology
    serve_prepared_foreground
    ;;
  serve-qwen-vl8-ontology|serve-ontology-qwen-vl8)
    prepare_qwen_vl8_ontology
    serve_prepared_foreground
    ;;
  serve-qwen-vl30-ontology|serve-ontology-qwen-vl30)
    prepare_qwen_vl30_ontology
    serve_prepared_foreground
    ;;
  serve-qwen35-ontology|serve-qwen-ontology)
    prepare_qwen_ontology
    serve_prepared_foreground
    ;;
  serve-qwen35-ontology-int4|serve-ontology-int4|serve-qwen-ontology-int4)
    prepare_qwen_ontology_int4
    serve_prepared_foreground
    ;;
  serve-llama-ontology-4bit)
    prepare_llama_ontology_4bit
    serve_prepared_foreground
    ;;
  serve-llama-ontology-tp2)
    prepare_llama_ontology_tp2
    serve_prepared_foreground
    ;;
  serve-attribute-selector)
    prepare_default_attribute_selector
    serve_prepared_foreground
    ;;
  serve-attribute-selector8|serve-qwen-vl8-attribute-selector)
    prepare_attribute_selector8
    serve_prepared_foreground
    ;;
  serve-attribute-selector30|serve-qwen-vl30-attribute-selector)
    prepare_attribute_selector30
    serve_prepared_foreground
    ;;
  serve-reader)
    case "${2:-30}" in
      30|30b|qwen30) prepare_reader30 ;;
      8|8b|qwen8) prepare_reader8 ;;
      llava|llava-med|7|7b) prepare_reader_llava_med ;;
      *) fail "Unknown reader variant: ${2}. Use 30, 8, or llava-med." ;;
    esac
    serve_prepared_foreground
    ;;
  serve-reader30|serve-qwen-vl30)
    prepare_reader30
    serve_prepared_foreground
    ;;
  serve-reader8|serve-qwen-vl8)
    prepare_reader8
    serve_prepared_foreground
    ;;
  serve-reader-llava-med|serve-llava-med-reader)
    prepare_reader_llava_med
    serve_prepared_foreground
    ;;

  -h|--help|help|"") usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac

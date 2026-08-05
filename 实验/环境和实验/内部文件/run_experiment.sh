#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${BUNDLE_DIR}/../.." && pwd)"
PROJECT_DIR="${REPO_ROOT}"
RUNTIME_DIR="${BUNDLE_DIR}/.runtime"
RESULT_ROOT="$(CDPATH= cd -- "${BUNDLE_DIR}/.." && pwd)/结果"
LOG_DIR="${RUNTIME_DIR}/logs"
RUN_STAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="${LOG_DIR}/${RUN_STAMP}.log"
ACTIVE_PID=""
LAST_PROGRESS=-1

show_progress() {
    local percent="$1"
    if (( percent < 0 )); then
        percent=0
    elif (( percent > 100 )); then
        percent=100
    fi
    if (( percent > LAST_PROGRESS )); then
        LAST_PROGRESS="${percent}"
        printf '\r实验总进度：%s%%' "${percent}" >&2
    fi
}

finish_success() {
    printf '\n结束实验\n' >&2
    exit 0
}

fail_experiment() {
    local reason="$1"
    local exit_code="${2:-1}"
    if [[ -n "${ACTIVE_PID}" ]]; then
        kill "${ACTIVE_PID}" >/dev/null 2>&1 || true
        wait "${ACTIVE_PID}" >/dev/null 2>&1 || true
        ACTIVE_PID=""
    fi
    printf '\n实验失败：%s（详见运行日志）\n' "${reason}" >&2
    printf '结束实验\n' >&2
    exit "${exit_code}"
}

run_with_heartbeat() {
    local start_percent="$1"
    local cap_percent="$2"
    shift 2
    local current="${start_percent}"
    show_progress "${current}"
    "$@" >>"${LOG_FILE}" 2>&1 &
    ACTIVE_PID="$!"
    while kill -0 "${ACTIVE_PID}" >/dev/null 2>&1; do
        sleep 2
        if (( current < cap_percent )); then
            current=$((current + 1))
            show_progress "${current}"
        fi
    done
    wait "${ACTIVE_PID}"
    local status="$?"
    ACTIVE_PID=""
    return "${status}"
}

download_miniconda() {
    local installer="$1"
    local url="$2"
    if command -v curl >/dev/null 2>&1; then
        run_with_heartbeat 3 7 \
            curl -fL --retry 3 --connect-timeout 30 \
            -o "${installer}" "${url}"
        return $?
    fi
    if command -v wget >/dev/null 2>&1; then
        run_with_heartbeat 3 7 \
            wget --tries=3 --timeout=30 -O "${installer}" "${url}"
        return $?
    fi
    return 127
}

trap 'fail_experiment "实验被中断" 130' INT TERM

printf '开始实验\n' >&2
show_progress 0

if ! mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}/downloads" \
        "${RUNTIME_DIR}/envs" "${RUNTIME_DIR}/conda-pkgs" \
        "${RESULT_ROOT}" >/dev/null 2>&1; then
    fail_experiment "无法创建运行目录" 2
fi
touch "${LOG_FILE}" >/dev/null 2>&1 || \
    fail_experiment "无法创建运行日志" 2

{
    printf '[%s] bundle=%s\n' "$(date '+%F %T')" "${BUNDLE_DIR}"
    printf '[%s] project=%s\n' "$(date '+%F %T')" "${PROJECT_DIR}"
} >>"${LOG_FILE}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    fail_experiment "仅支持 Linux x86_64" 3
fi
if ! command -v sha256sum >/dev/null 2>&1; then
    fail_experiment "系统缺少 sha256sum" 3
fi
if [[ ! -f "${PROJECT_DIR}/requirements.txt" \
        || ! -f "${PROJECT_DIR}/invert.py" ]]; then
    fail_experiment "项目文件不完整" 3
fi

REQUIREMENTS_HASH="$(sha256sum "${PROJECT_DIR}/requirements.txt" \
    | awk '{print substr($1,1,12)}')"
ENV_RUNTIME_DIR="${RUNTIME_DIR}"
CASE_PROBE_DIR="${RUNTIME_DIR}/case-probe-${RUN_STAMP}"
if ! mkdir -p "${CASE_PROBE_DIR}" \
        || ! : >"${CASE_PROBE_DIR}/probe" \
        || ! : >"${CASE_PROBE_DIR}/PROBE"; then
    fail_experiment "无法检查运行目录文件系统" 3
fi
CASE_ENTRY_COUNT="$(find "${CASE_PROBE_DIR}" -maxdepth 1 -type f \
    | wc -l | tr -d '[:space:]')"
rm -f "${CASE_PROBE_DIR}/probe" "${CASE_PROBE_DIR}/PROBE"
rmdir "${CASE_PROBE_DIR}" >/dev/null 2>&1 || true
if [[ "${CASE_ENTRY_COUNT}" != "2" ]]; then
    PROJECT_HASH="$(printf '%s' "${PROJECT_DIR}" | sha256sum \
        | awk '{print substr($1,1,12)}')"
    ENV_RUNTIME_DIR="${HOME}/.cache/deml-one-click/${PROJECT_HASH}-${REQUIREMENTS_HASH}"
fi
if ! mkdir -p "${ENV_RUNTIME_DIR}/envs" \
        "${ENV_RUNTIME_DIR}/conda-pkgs" >/dev/null 2>&1; then
    fail_experiment "无法创建隔离环境目录" 3
fi
printf '[%s] environment_runtime=%s\n' "$(date '+%F %T')" \
    "${ENV_RUNTIME_DIR}" >>"${LOG_FILE}"
ENV_PREFIX="${ENV_RUNTIME_DIR}/envs/deml-${REQUIREMENTS_HASH}"
ENV_PYTHON="${ENV_PREFIX}/bin/python"
MODEL_CACHE_DIR="${RUNTIME_DIR}/hf-cache/hub/models--Qwen--Qwen2.5-1.5B"
AVAILABLE_KB="$(df -Pk "${BUNDLE_DIR}" 2>>"${LOG_FILE}" \
    | awk 'NR==2 {print $4}')"
ENV_AVAILABLE_KB="$(df -Pk "${ENV_RUNTIME_DIR}" 2>>"${LOG_FILE}" \
    | awk 'NR==2 {print $4}')"
if [[ ! "${AVAILABLE_KB}" =~ ^[0-9]+$ \
        || ! "${ENV_AVAILABLE_KB}" =~ ^[0-9]+$ ]]; then
    fail_experiment "无法检查磁盘空间" 3
fi
if [[ ! -x "${ENV_PYTHON}" || ! -d "${MODEL_CACHE_DIR}" ]]; then
    REQUIRED_KB=$((15 * 1024 * 1024))
else
    REQUIRED_KB=$((1 * 1024 * 1024))
fi
if (( AVAILABLE_KB < REQUIRED_KB || ENV_AVAILABLE_KB < REQUIRED_KB )); then
    fail_experiment "磁盘空间不足" 3
fi
show_progress 2

MINICONDA_NAME="Miniconda3-py310_26.5.3-1-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_NAME}"
MINICONDA_SHA256="4a82fe0a50a28e8a9406b3ed8e465b7009aa7d0225566802c3370df96b10d834"
MINICONDA_INSTALLER="${RUNTIME_DIR}/downloads/${MINICONDA_NAME}"
LOCAL_CONDA="${ENV_RUNTIME_DIR}/miniconda-py310-26.5.3"

if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
elif [[ -x "${LOCAL_CONDA}/bin/conda" ]]; then
    CONDA_EXE="${LOCAL_CONDA}/bin/conda"
else
    if [[ -d "${LOCAL_CONDA}" ]]; then
        if ! mv "${LOCAL_CONDA}" \
                "${LOCAL_CONDA}.incomplete-${RUN_STAMP}" \
                >>"${LOG_FILE}" 2>&1; then
            fail_experiment "无法隔离未完成的 Miniconda" 4
        fi
    fi
    if [[ ! -f "${MINICONDA_INSTALLER}" ]]; then
        if ! download_miniconda "${MINICONDA_INSTALLER}" \
                "${MINICONDA_URL}"; then
            fail_experiment "Miniconda 下载失败" 4
        fi
    fi
    ACTUAL_SHA256="$(sha256sum "${MINICONDA_INSTALLER}" \
        | awk '{print $1}')"
    if [[ "${ACTUAL_SHA256}" != "${MINICONDA_SHA256}" ]]; then
        fail_experiment "Miniconda 校验失败" 4
    fi
    if ! run_with_heartbeat 8 11 \
            bash "${MINICONDA_INSTALLER}" -b -p "${LOCAL_CONDA}"; then
        fail_experiment "Miniconda 安装失败" 4
    fi
    CONDA_EXE="${LOCAL_CONDA}/bin/conda"
fi
show_progress 12

export CONDA_PKGS_DIRS="${ENV_RUNTIME_DIR}/conda-pkgs"
export PYTHONNOUSERSITE=1
export PIP_NO_INPUT=1
export HF_HOME="${RUNTIME_DIR}/hf-cache"
GPU_ID="${DEML_GPU_ID:-0}"
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    fail_experiment "DEML_GPU_ID 必须是单个非负 GPU 编号" 7
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
printf '[%s] physical_gpu=%s\n' "$(date '+%F %T')" "${GPU_ID}" \
    >>"${LOG_FILE}"

ENV_READY=0
if [[ -x "${ENV_PYTHON}" ]]; then
    if "${ENV_PYTHON}" "${SCRIPT_DIR}/runner.py" check-env \
            --project "${PROJECT_DIR}" >>"${LOG_FILE}" 2>&1; then
        ENV_READY=1
    fi
fi

if (( ENV_READY == 0 )); then
    if [[ -x "${ENV_PYTHON}" ]]; then
        if ! run_with_heartbeat 12 14 \
                "${CONDA_EXE}" install -y \
                --override-channels --channel conda-forge \
                -p "${ENV_PREFIX}" \
                python=3.10.20 pip=26.0.1; then
            fail_experiment "Python 环境修复失败" 5
        fi
    else
        if [[ -d "${ENV_PREFIX}" ]]; then
            if ! mv "${ENV_PREFIX}" \
                    "${ENV_PREFIX}.incomplete-${RUN_STAMP}" \
                    >>"${LOG_FILE}" 2>&1; then
                fail_experiment "无法隔离未完成的 Python 环境" 5
            fi
        fi
        if ! run_with_heartbeat 12 14 \
                "${CONDA_EXE}" create -y \
                --override-channels --channel conda-forge \
                -p "${ENV_PREFIX}" \
                python=3.10.20 pip=26.0.1; then
            fail_experiment "Python 环境创建失败" 5
        fi
    fi
    show_progress 15
    if ! run_with_heartbeat 15 34 \
            "${ENV_PYTHON}" -m pip install \
            --disable-pip-version-check --no-input \
            --progress-bar off -r "${PROJECT_DIR}/requirements.txt"; then
        fail_experiment "依赖安装失败" 6
    fi
fi

if ! "${ENV_PYTHON}" "${SCRIPT_DIR}/runner.py" check-env \
        --project "${PROJECT_DIR}" >>"${LOG_FILE}" 2>&1; then
    fail_experiment "环境版本校验失败" 6
fi
if ! "${ENV_PYTHON}" -m pip check >>"${LOG_FILE}" 2>&1; then
    fail_experiment "依赖完整性校验失败" 6
fi
if ! "${ENV_PYTHON}" -c \
        'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
        >>"${LOG_FILE}" 2>&1; then
    fail_experiment "CUDA 不可用" 7
fi
show_progress 35

CONDA_BASE="$("${CONDA_EXE}" info --base 2>>"${LOG_FILE}")"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    fail_experiment "Conda 激活脚本缺失" 7
fi
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh" >>"${LOG_FILE}" 2>&1
ACTIVATE_STATUS="$?"
set -u
if (( ACTIVATE_STATUS != 0 )); then
    fail_experiment "Conda 初始化失败" 7
fi
if ! conda activate "${ENV_PREFIX}" >>"${LOG_FILE}" 2>&1; then
    fail_experiment "环境切换失败" 7
fi
if [[ "${CONDA_PREFIX:-}" != "${ENV_PREFIX}" ]]; then
    fail_experiment "环境切换校验失败" 7
fi

RUNNER_ARGS=(
    run
    --project "${PROJECT_DIR}"
    --runtime "${RUNTIME_DIR}"
    --result-root "${RESULT_ROOT}"
    --log-file "${LOG_FILE}"
)
if [[ "${DEML_SMOKE_TEST:-0}" == "1" ]]; then
    RUNNER_ARGS+=(--smoke-test)
fi
"${ENV_PYTHON}" "${SCRIPT_DIR}/runner.py" "${RUNNER_ARGS[@]}"
RUNNER_STATUS="$?"

case "${RUNNER_STATUS}" in
    0)
        finish_success
        ;;
    20)
        fail_experiment "模型下载失败" 20
        ;;
    30)
        fail_experiment "实验运行失败" 30
        ;;
    40)
        fail_experiment "结果复制或校验失败" 40
        ;;
    130)
        fail_experiment "实验被中断" 130
        ;;
    *)
        fail_experiment "运行器发生异常" "${RUNNER_STATUS}"
        ;;
esac

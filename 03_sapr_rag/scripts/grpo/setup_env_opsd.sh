#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# SAPR-RAG GRPO 环境检查脚本（开发机 + worker 共享磁盘，环境一次配置永久持有）
#
# 默认行为：纯**只读验证**，不会动 pip。打印当前依赖矩阵 + 验证 plugin 注册。
#           worker 重新申请后跑这一条即可确认环境健康。
#
# 修复模式：仅当检查发现版本不符时，加 --repair 才会按下面的钉死矩阵 pip 安装；
#           --repair 也只动这 4 个包，绝不升级 vllm/torch。
#
# 已验证的兼容矩阵（2026-06-09 跑通 swift rlhf grpo 数据预处理 + plugin 注册）：
#   ms-swift     4.4.0.dev0 (本地 editable)
#   vllm         0.10.0      ← 必须保持，不动
#   torch        2.7.1       ← 必须保持，不动
#   trl          0.26.2      ← GRPO trainer 硬要求 >=0.26
#   transformers 4.56.2      ← trl 0.26 要 >=4.56.1；vllm 0.10.0 只要 >=4.53.2
#   tokenizers   0.22.2      ← 随 transformers 4.56 自动配套
#   datasets     4.8.4       ← core.py 需 datasets>=4.4 的 `Json` feature；上限 <4.8.5
#
# 用法：
#   bash setup_env.sh           # 只读验证（默认）
#   bash setup_env.sh --repair  # 只在版本不符时 pip 安装
# ───────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPAIR=0
[ "${1:-}" = "--repair" ] && REPAIR=1

GRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$GRPO_DIR/../../.." && pwd)"
SWIFT_ROOT="${SWIFT_ROOT:-$(cd "$PROJ_ROOT/../ms-swift" && pwd)}"
BASE_MODEL="${BASE_MODEL:-$PROJ_ROOT/03_sapr_rag/models/Qwen2.5-7B-Instruct}"
SFT_DPO_ADAPTER="${SFT_DPO_ADAPTER:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_dpo/checkpoint-395}"
DEVICE_BACKEND="${DEVICE_BACKEND:-cuda}"

echo "════════════════════════════════════════════════════════════"
echo "[check] mode       = $([ $REPAIR -eq 1 ] && echo REPAIR || echo VERIFY-ONLY)"
echo "[check] SWIFT_ROOT = $SWIFT_ROOT"
echo "[check] GRPO_DIR   = $GRPO_DIR"
echo "[check] BASE_MODEL = $BASE_MODEL"
echo "[check] SFT_DPO    = $SFT_DPO_ADAPTER"
echo "[check] backend    = $DEVICE_BACKEND"
echo "════════════════════════════════════════════════════════════"

# ── 一次性收集所有版本 + 关键 import 状态 ─────────────────────────────────
STATUS_FILE="$(mktemp)"
trap "rm -f $STATUS_FILE" EXIT

python - "$GRPO_DIR" "$STATUS_FILE" "$BASE_MODEL" "$SFT_DPO_ADAPTER" <<'PY'
import importlib
import inspect
import json
import os
import sys

grpo_dir, status_file, base_model, adapter_path = sys.argv[1:]

def vers(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return None

result = {"versions": {}, "checks": {}, "errors": []}
for n in ["vllm", "torch", "transformers", "trl", "tokenizers", "datasets", "swift", "torch_npu", "vllm_ascend"]:
    result["versions"][n] = vers(n)

def record(name, value, error=None):
    result["checks"][name] = bool(value)
    if error:
        result["errors"].append(f"{name}: {error}")

# datasets.features 的 Json/List
try:
    from datasets.features import Json, List  # noqa
    result["checks"]["datasets_json_feature"] = True
except Exception as e:
    result["checks"]["datasets_json_feature"] = False
    result["errors"].append(f"datasets.features Json/List: {e}")

# plugin 注册（三 ORM + scheduler）
try:
    sys.path.insert(0, grpo_dir)
    import plugin  # noqa
    from swift.rewards import orms
    from swift.rollout.multi_turn import multi_turns
    sapr_orms = sorted(k for k in orms if k.startswith("sapr"))
    sapr_sched = [k for k in multi_turns if "sapr" in k]
    result["checks"]["plugin_orms"] = sapr_orms
    result["checks"]["plugin_scheduler"] = sapr_sched
except Exception as e:
    result["checks"]["plugin_orms"] = None
    result["checks"]["plugin_scheduler"] = None
    result["errors"].append(f"plugin import: {e}")

# ms-swift dynamic OPSD / OPD-RL 能力检查（只读，不实例化模型）
try:
    from swift.rl_core.advantage import compute_teacher_logratio, expand_advantage_to_per_token
    from swift.rl_core.data import OnPolicySample
    from swift.rlhf_trainers.gkd_helpers import (
        resolve_dynamic_opd_self_distillation,
        should_compute_local_teacher_logps,
    )

    record("opd_compute_teacher_logratio", callable(compute_teacher_logratio))
    record(
        "opd_advantage_injection",
        "teacher_kl_coef" in inspect.signature(expand_advantage_to_per_token).parameters,
    )
    record(
        "opd_dynamic_self_distillation",
        resolve_dynamic_opd_self_distillation(
            has_teacher_explicit=False,
            is_self_distillation=True,
        ),
    )
    record(
        "opd_teacher_forward_gate",
        should_compute_local_teacher_logps(
            has_teacher_explicit=False,
            is_dynamic_self_distillation=True,
            use_teacher_api=False,
            has_opsd_batch=True,
        ),
    )
    record("opd_teacher_prompt_field", "teacher_prompt" in OnPolicySample.__dataclass_fields__)
except Exception as e:
    for key in (
        "opd_compute_teacher_logratio",
        "opd_advantage_injection",
        "opd_dynamic_self_distillation",
        "opd_teacher_forward_gate",
        "opd_teacher_prompt_field",
    ):
        record(key, False)
    result["errors"].append(f"dynamic OPSD imports: {e}")

# 当前源码契约：GKD 路径仍保留 fail-fast 守卫（NotImplementedError/ValueError）。
# 说明：新版 ms-swift 的 _check_gkd 已不再以 `multi_turn_scheduler` 字样表达该限制，
# 故此处只校验 GKD 仍有 fail-fast 守卫；我们的方案走 dynamic OPSD，不依赖 GKD。
try:
    from swift.arguments.rlhf_args import RLHFArguments

    gkd_check_src = inspect.getsource(RLHFArguments._check_gkd)
    record(
        "gkd_multiturn_failfast",
        ("NotImplementedError" in gkd_check_src) or ("raise ValueError" in gkd_check_src),
    )
except Exception as e:
    record("gkd_multiturn_failfast", False, e)

# Ascend/NPU 环境检查：只导入轻量模块，不加载模型。
try:
    import torch
    try:
        import torch_npu  # noqa
    except Exception as e:
        record("npu_torch_npu_import", False, e)
    else:
        record("npu_torch_npu_import", True)
    npu_obj = getattr(torch, "npu", None)
    npu_count = int(npu_obj.device_count()) if npu_obj is not None else 0
    result["checks"]["npu_device_count"] = npu_count
    record("npu_visible", npu_count > 0)
except Exception as e:
    result["checks"]["npu_device_count"] = 0
    record("npu_visible", False, e)

try:
    import vllm_ascend  # noqa
    record("npu_vllm_ascend_import", True)
except Exception as e:
    record("npu_vllm_ascend_import", False, e)

# 轻量模型/adapter 检查：只读配置文件，不加载模型权重。
record("base_model_path", os.path.isdir(base_model))
record(
    "base_model_config",
    all(os.path.isfile(os.path.join(base_model, name))
        for name in ("config.json", "tokenizer_config.json")),
)
record("sft_dpo_adapter_path", os.path.isdir(adapter_path))
adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
record("sft_dpo_adapter_config", os.path.isfile(adapter_config_path))
if os.path.isfile(adapter_config_path):
    try:
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        configured_base_raw = adapter_config.get("base_model_name_or_path", "")
        configured_base = os.path.realpath(configured_base_raw)
        expected_base = os.path.realpath(base_model)
        same_path = configured_base == expected_base
        same_model_name = os.path.basename(configured_base_raw.rstrip("/")) == os.path.basename(expected_base)
        compatible = same_path or same_model_name
        record("sft_dpo_base_compatible", compatible)
        if not compatible:
            result["errors"].append(
                "sft_dpo_base_compatible: "
                f"adapter base={configured_base_raw!r}, expected={expected_base!r}"
            )
    except Exception as e:
        record("sft_dpo_base_compatible", False, e)
else:
    record("sft_dpo_base_compatible", False)

with open(status_file, "w") as f:
    json.dump(result, f)
PY

# ── 版本对照（不在则记缺失，版本不匹配则记 mismatch）──────────────────────
EXPECT_TRL="0.26.2"
EXPECT_TRANSFORMERS="4.56.2"
EXPECT_TOKENIZERS_PREFIX="0.22"
EXPECT_DATASETS_MIN="4.4"
EXPECT_DATASETS_MAX_EXCL="4.8.5"

NEED_FIX=0

print_ver() {
    local name="$1" got="$2" want="$3"
    if [ -z "$got" ] || [ "$got" = "null" ]; then
        echo "  [MISSING] $name (期望 $want)"
        NEED_FIX=1
    elif [ -n "$want" ] && [ "$got" != "$want" ]; then
        echo "  [MISMATCH] $name = $got  (期望 $want)"
        NEED_FIX=1
    else
        echo "  [OK]      $name = $got"
    fi
}

JQ() { python -c "import json,sys; d=json.load(open('$STATUS_FILE')); print(d$1)" 2>/dev/null || echo ""; }

VLLM_V=$(JQ "['versions']['vllm']")
TORCH_V=$(JQ "['versions']['torch']")
TORCH_NPU_V=$(JQ "['versions']['torch_npu']")
VLLM_ASCEND_V=$(JQ "['versions']['vllm_ascend']")
TRANSFORMERS_V=$(JQ "['versions']['transformers']")
TRL_V=$(JQ "['versions']['trl']")
TOKENIZERS_V=$(JQ "['versions']['tokenizers']")
DATASETS_V=$(JQ "['versions']['datasets']")
MSSWIFT_V=$(JQ "['versions']['swift']")
JSON_OK=$(JQ "['checks']['datasets_json_feature']")
ORMS_OK=$(JQ "['checks']['plugin_orms']")
SCHED_OK=$(JQ "['checks']['plugin_scheduler']")
OPD_LOGRATIO_OK=$(JQ "['checks']['opd_compute_teacher_logratio']")
OPD_ADV_OK=$(JQ "['checks']['opd_advantage_injection']")
OPD_DYNAMIC_OK=$(JQ "['checks']['opd_dynamic_self_distillation']")
OPD_GATE_OK=$(JQ "['checks']['opd_teacher_forward_gate']")
OPD_PROMPT_OK=$(JQ "['checks']['opd_teacher_prompt_field']")
GKD_MULTITURN_FAILFAST_OK=$(JQ "['checks']['gkd_multiturn_failfast']")
NPU_TORCH_OK=$(JQ "['checks']['npu_torch_npu_import']")
NPU_VISIBLE_OK=$(JQ "['checks']['npu_visible']")
NPU_DEVICE_COUNT=$(JQ "['checks']['npu_device_count']")
NPU_VLLM_ASCEND_OK=$(JQ "['checks']['npu_vllm_ascend_import']")
BASE_PATH_OK=$(JQ "['checks']['base_model_path']")
BASE_CONFIG_OK=$(JQ "['checks']['base_model_config']")
ADAPTER_PATH_OK=$(JQ "['checks']['sft_dpo_adapter_path']")
ADAPTER_CONFIG_OK=$(JQ "['checks']['sft_dpo_adapter_config']")
ADAPTER_BASE_OK=$(JQ "['checks']['sft_dpo_base_compatible']")

echo
echo "── 重依赖（不主动安装/升级）──"
print_ver "vllm"  "$VLLM_V"  ""
print_ver "torch" "$TORCH_V" ""
[ -n "$VLLM_V" ] && [[ ! "$VLLM_V" =~ ^0\.10\. ]] \
    && echo "  [WARN] vllm $VLLM_V 不是 0.10.x，与 transformers 4.56 兼容性需自查"

echo
echo "── Ascend / NPU 依赖（DEVICE_BACKEND=npu 时为硬条件）──"
echo "  [INFO] torch_npu = ${TORCH_NPU_V:-None}"
echo "  [INFO] vllm_ascend = ${VLLM_ASCEND_V:-None}"
echo "  [INFO] npu_device_count = ${NPU_DEVICE_COUNT:-0}"
if [ "$DEVICE_BACKEND" = "npu" ]; then
    check_npu_capability() {
        local label="$1" value="$2"
        if [ "$value" = "True" ]; then
            echo "  [OK]  $label"
        else
            echo "  [FAIL] $label"
            NEED_FIX=1
        fi
    }
    check_npu_capability "torch_npu import 可用" "$NPU_TORCH_OK"
    check_npu_capability "NPU 设备可见" "$NPU_VISIBLE_OK"
    check_npu_capability "vllm_ascend import 可用" "$NPU_VLLM_ASCEND_OK"
else
    [ "$NPU_TORCH_OK" = "True" ] && echo "  [OK]  torch_npu import 可用" || echo "  [WARN] torch_npu 不可用；仅影响 DEVICE_BACKEND=npu"
    [ "$NPU_VISIBLE_OK" = "True" ] && echo "  [OK]  NPU 设备可见" || echo "  [WARN] NPU 设备不可见；仅影响 DEVICE_BACKEND=npu"
    [ "$NPU_VLLM_ASCEND_OK" = "True" ] && echo "  [OK]  vllm_ascend import 可用" || echo "  [WARN] vllm_ascend 不可用；仅影响 Ascend rollout"
fi

echo
echo "── 关键依赖（钉死矩阵）──"
print_ver "ms-swift"     "$MSSWIFT_V"      ""
print_ver "trl"          "$TRL_V"          "$EXPECT_TRL"
print_ver "transformers" "$TRANSFORMERS_V" "$EXPECT_TRANSFORMERS"
[ -n "$TOKENIZERS_V" ] && [[ ! "$TOKENIZERS_V" =~ ^${EXPECT_TOKENIZERS_PREFIX}\. ]] \
    && { echo "  [MISMATCH] tokenizers = $TOKENIZERS_V (期望 ${EXPECT_TOKENIZERS_PREFIX}.x)"; NEED_FIX=1; } \
    || echo "  [OK]      tokenizers = $TOKENIZERS_V"

if [ -n "$DATASETS_V" ]; then
    if python -c "
from packaging.version import Version
v = Version('$DATASETS_V')
ok = Version('$EXPECT_DATASETS_MIN') <= v < Version('$EXPECT_DATASETS_MAX_EXCL')
import sys; sys.exit(0 if ok else 1)" 2>/dev/null; then
        echo "  [OK]      datasets = $DATASETS_V"
    else
        echo "  [MISMATCH] datasets = $DATASETS_V (期望 >=$EXPECT_DATASETS_MIN,<$EXPECT_DATASETS_MAX_EXCL)"
        NEED_FIX=1
    fi
else
    echo "  [MISSING] datasets"
    NEED_FIX=1
fi

echo
echo "── 功能性检查 ──"
[ "$JSON_OK" = "True" ] && echo "  [OK]  datasets.features Json/List import" \
                       || { echo "  [FAIL] datasets.features Json/List import"; NEED_FIX=1; }

EXPECTED_ORMS="['sapr_f1', 'sapr_format', 'sapr_relevance']"
EXPECTED_SCHED="['sapr_rag_scheduler']"
if [ "$ORMS_OK" = "$EXPECTED_ORMS" ] && [ "$SCHED_OK" = "$EXPECTED_SCHED" ]; then
    echo "  [OK]  plugin 注册：$ORMS_OK + $SCHED_OK"
else
    echo "  [FAIL] plugin 注册异常： orms=$ORMS_OK  scheduler=$SCHED_OK"
    NEED_FIX=1
fi

check_capability() {
    local label="$1" value="$2"
    if [ "$value" = "True" ]; then
        echo "  [OK]  $label"
    else
        echo "  [FAIL] $label"
        NEED_FIX=1
    fi
}

echo
echo "── OPD / OPSD 能力检查（只读）──"
check_capability "compute_teacher_logratio 可用" "$OPD_LOGRATIO_OK"
check_capability "per-token teacher advantage 注入可用" "$OPD_ADV_OK"
check_capability "dynamic OPSD self-distillation 可用" "$OPD_DYNAMIC_OK"
check_capability "dynamic OPSD teacher forward gate 可用" "$OPD_GATE_OK"
check_capability "dataset teacher_prompt 字段可用" "$OPD_PROMPT_OK"
check_capability "GKD 路径仍有 fail-fast 守卫（我们不依赖 GKD）" "$GKD_MULTITURN_FAILFAST_OK"

echo
echo "── OPD 模型与 adapter 轻量检查 ──"
check_capability "Qwen2.5 base 路径存在" "$BASE_PATH_OK"
check_capability "Qwen2.5 config/tokenizer_config 存在" "$BASE_CONFIG_OK"
check_capability "SFT+DPO checkpoint-395 路径存在" "$ADAPTER_PATH_OK"
check_capability "SFT+DPO adapter_config.json 存在" "$ADAPTER_CONFIG_OK"
check_capability "SFT+DPO adapter 指向当前 Qwen2.5 base" "$ADAPTER_BASE_OK"

# ── 错误信息 ──
ERRS=$(python - "$STATUS_FILE" "$DEVICE_BACKEND" <<'PY'
import json
import sys

status_file, backend = sys.argv[1:]
errors = json.load(open(status_file)).get("errors", [])
if backend != "npu":
    errors = [
        e for e in errors
        if not e.startswith(("npu_torch_npu_import:", "npu_visible:", "npu_vllm_ascend_import:"))
    ]
print(json.dumps(errors, ensure_ascii=False))
PY
)
if [ "$ERRS" != "[]" ] && [ -n "$ERRS" ]; then
    echo
    echo "── 错误明细 ──"
    echo "  $ERRS"
fi

echo
echo "════════════════════════════════════════════════════════════"

# ── 出口判断 ──
if [ $NEED_FIX -eq 0 ]; then
    echo "[check] 环境就绪 ✓"
    echo
    echo "  下一步："
    echo "    1) 起检索 daemon ：GPU=7 bash run_retrieval_daemon.sh"
    echo "    2) 链路 sanity    ：python sanity_check.py"
    echo "    3) 起 rollout     ：GPU=6 bash run_rollout.sh   (记下实际监听端口)"
    echo "    4) 起训练(100条)  ：DATASET=.../hotpotqa_train_100.jsonl VLLM_PORT=<端口> DEEPSPEED=none bash run_grpo.sh"
    echo
    echo "  Ascend 910B2 示例："
    echo "    1) DEVICE_BACKEND=cpu PORT=8100 bash run_retrieval_daemon.sh"
    echo "    2) DEVICE_BACKEND=npu ROLLOUT_DEVICES=6 PORT=8000 bash run_rollout.sh"
    echo "    3) DEVICE_BACKEND=npu TRAIN_DEVICES=0,1,2,3,4,5 NPROC_PER_NODE=6 DATASET=... bash run_grpo.sh"
    echo "════════════════════════════════════════════════════════════"
    exit 0
fi

# ── 有问题：要么提示，要么修复 ──
if [ $REPAIR -eq 0 ]; then
    echo "[check] 检测到环境异常。"
    echo "  共享磁盘环境理论上不应漂移；先排查："
    echo "    - 是否被其他项目改了 transformers/trl/datasets？"
    echo "    - ~/.local 是否被清过？"
    echo "  如确需修复，跑：bash setup_env.sh --repair"
    echo "════════════════════════════════════════════════════════════"
    exit 1
fi

# ── REPAIR：只装/钉本矩阵需要的 4 个包，全程 --no-deps 或精确版本 ──
echo "[repair] 开始按钉死矩阵修复（--no-deps / 精确版本）..."

# ms-swift editable（每次重装无害；仅在缺失时执行）
if [ -z "$MSSWIFT_V" ] || [ "$MSSWIFT_V" = "null" ]; then
    echo "[repair] pip install -e $SWIFT_ROOT --no-deps"
    pip install -e "$SWIFT_ROOT" --no-deps
fi

# trl
if [ "$TRL_V" != "$EXPECT_TRL" ]; then
    echo "[repair] pip install --no-deps trl==$EXPECT_TRL"
    pip install --no-deps "trl==$EXPECT_TRL"
fi

# transformers（不加 --no-deps，让 tokenizers 配套自动到 0.22.x）
if [ "$TRANSFORMERS_V" != "$EXPECT_TRANSFORMERS" ]; then
    echo "[repair] pip install transformers==$EXPECT_TRANSFORMERS"
    pip install "transformers==$EXPECT_TRANSFORMERS"
fi

# datasets
need_datasets=0
[ -z "$DATASETS_V" ] && need_datasets=1
[ -n "$DATASETS_V" ] && python -c "
from packaging.version import Version
v = Version('$DATASETS_V')
ok = Version('$EXPECT_DATASETS_MIN') <= v < Version('$EXPECT_DATASETS_MAX_EXCL')
import sys; sys.exit(0 if ok else 1)" 2>/dev/null || need_datasets=1

if [ $need_datasets -eq 1 ]; then
    echo "[repair] pip install --no-deps 'datasets>=$EXPECT_DATASETS_MIN,<$EXPECT_DATASETS_MAX_EXCL' -U"
    pip install --no-deps "datasets>=$EXPECT_DATASETS_MIN,<$EXPECT_DATASETS_MAX_EXCL" -U
fi

echo
echo "[repair] 修复完成。重跑验证 ..."
echo "════════════════════════════════════════════════════════════"
exec bash "$0"

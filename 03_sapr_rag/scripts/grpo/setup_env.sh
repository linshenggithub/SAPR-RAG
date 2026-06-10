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

SWIFT_ROOT="${SWIFT_ROOT:-/mlx_devbox/users/mayi.summer/playground/ms-swift}"
GRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════════════════════════════════════════════════════════════"
echo "[check] mode       = $([ $REPAIR -eq 1 ] && echo REPAIR || echo VERIFY-ONLY)"
echo "[check] SWIFT_ROOT = $SWIFT_ROOT"
echo "[check] GRPO_DIR   = $GRPO_DIR"
echo "════════════════════════════════════════════════════════════"

# ── 一次性收集所有版本 + 关键 import 状态 ─────────────────────────────────
STATUS_FILE="$(mktemp)"
trap "rm -f $STATUS_FILE" EXIT

python - "$GRPO_DIR" "$STATUS_FILE" <<'PY'
import json, sys, importlib
grpo_dir, status_file = sys.argv[1], sys.argv[2]

def vers(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return None

result = {"versions": {}, "checks": {}, "errors": []}
for n in ["vllm", "torch", "transformers", "trl", "tokenizers", "datasets", "swift"]:
    result["versions"][n] = vers(n)

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
TRANSFORMERS_V=$(JQ "['versions']['transformers']")
TRL_V=$(JQ "['versions']['trl']")
TOKENIZERS_V=$(JQ "['versions']['tokenizers']")
DATASETS_V=$(JQ "['versions']['datasets']")
MSSWIFT_V=$(JQ "['versions']['swift']")
JSON_OK=$(JQ "['checks']['datasets_json_feature']")
ORMS_OK=$(JQ "['checks']['plugin_orms']")
SCHED_OK=$(JQ "['checks']['plugin_scheduler']")

echo
echo "── 重依赖（不主动安装/升级）──"
print_ver "vllm"  "$VLLM_V"  ""
print_ver "torch" "$TORCH_V" ""
[ -n "$VLLM_V" ] && [[ ! "$VLLM_V" =~ ^0\.10\. ]] \
    && echo "  [WARN] vllm $VLLM_V 不是 0.10.x，与 transformers 4.56 兼容性需自查"

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

# ── 错误信息 ──
ERRS=$(JQ "['errors']")
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

#!/usr/bin/env bash
# ===========================================================================
# SAPR-RAG 仓库清理脚本（P0 + P4）
# ---------------------------------------------------------------------------
# 用途：
#   1. P0：把 refine-logs/ 中的最新版迁移到 docs/，删除旧快照，删空 refine-logs
#   2. P4：删除 31 个 .gitkeep 占位文件，并删除 27 个空目录
#
# 前置条件（必须在跑脚本前由 AI 完成，否则脚本会拒绝执行）：
#   A) docs/history.md 已生成（旧 v1/v2/v3 阶段性总结）
#   B) gate0/GATE0_STATUS.md 已合并 refine-logs/GATE0_STATUS.md 的差异
#
# 安全保证：
#   - set -euo pipefail：任一步失败立即中止
#   - 所有 git 操作用 git mv / git rm 保留历史
#   - 删除空目录前用 rmdir（非空会自动拒绝，不会误删）
#   - 关键阶段有前置校验，不满足直接 exit 1
#
# 用法：
#   cd /mlx_devbox/users/mayi.summer/playground/SAPR-RAG
#   bash cleanup.sh
# ===========================================================================

set -euo pipefail

REPO_ROOT="/mlx_devbox/users/mayi.summer/playground/SAPR-RAG"
cd "$REPO_ROOT"

echo "================================================================"
echo "SAPR-RAG 清理脚本启动"
echo "工作目录: $(pwd)"
echo "================================================================"

# ---------------------------------------------------------------------------
# 前置校验：确保不在脏的 git 状态、确保前置 AI 工作已完成
# ---------------------------------------------------------------------------
echo ""
echo "[前置校验] 检查仓库状态..."

if [[ ! -d ".git" ]]; then
    echo "ERROR: 当前目录不是 git 仓库根目录"
    exit 1
fi

if [[ ! -f "gate0/GATE0_STATUS.md" ]]; then
    echo "ERROR: gate0/GATE0_STATUS.md 不存在，请先执行差异合并"
    exit 1
fi

# refine-logs 可能已经在上一次运行中被 P0 清掉；只在仍存在时强制校验
REFINE_LOGS_EXISTS=0
if [[ -d "refine-logs" ]]; then
    REFINE_LOGS_EXISTS=1

    if [[ ! -f "refine-logs/GATE0_STATUS.md" ]]; then
        echo "ERROR: refine-logs/ 仍存在但 GATE0_STATUS.md 缺失，状态不一致"
        exit 1
    fi

    # 确认 4 个待迁移的源文件存在
    for f in \
        "refine-logs/FINAL_PROPOSAL_v4.md" \
        "refine-logs/EXPERIMENT_PLAN_v2.md" \
        "refine-logs/EXPERIMENT_TRACKER_v2.md" \
        "refine-logs/PIPELINE_SUMMARY.md"
    do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: 待迁移文件不存在：$f"
            exit 1
        fi
    done
else
    echo "[前置校验] refine-logs/ 已不存在，跳过 P0（Step 1-5），直接执行 P4"
fi

# 确认 docs/history.md 已被 AI 写好
if [[ ! -f "docs/history.md" ]]; then
    echo "ERROR: docs/history.md 不存在；脚本要求 AI 先写好阶段性总结再执行清理"
    exit 1
fi

echo "[前置校验] 通过"

# ===========================================================================
# P0 部分（Step 1-5）：仅在 refine-logs/ 仍存在时执行，保持脚本幂等
# ===========================================================================
if [[ "$REFINE_LOGS_EXISTS" -eq 1 ]]; then

# ===========================================================================
# Step 1：创建 docs/ 目录（若不存在）
# ===========================================================================
echo ""
echo "[Step 1/8] 创建 docs/ 目录..."
mkdir -p docs

# ===========================================================================
# Step 2：迁移 4 个保留文件到 docs/（小写文件名，保留 git 历史）
# ===========================================================================
echo ""
echo "[Step 2/8] 迁移最新版文档到 docs/ ..."

git mv refine-logs/FINAL_PROPOSAL_v4.md      docs/proposal.md
git mv refine-logs/EXPERIMENT_PLAN_v2.md     docs/experiment_plan.md
git mv refine-logs/EXPERIMENT_TRACKER_v2.md  docs/experiment_tracker.md
git mv refine-logs/PIPELINE_SUMMARY.md       docs/pipeline.md

echo "    docs/proposal.md          (from FINAL_PROPOSAL_v4.md)"
echo "    docs/experiment_plan.md   (from EXPERIMENT_PLAN_v2.md)"
echo "    docs/experiment_tracker.md (from EXPERIMENT_TRACKER_v2.md)"
echo "    docs/pipeline.md          (from PIPELINE_SUMMARY.md)"

# ===========================================================================
# Step 3：删除 refine-logs/GATE0_STATUS.md
#         （内容已由 AI 合并进 gate0/GATE0_STATUS.md）
# ===========================================================================
echo ""
echo "[Step 3/8] 删除已合并的 refine-logs/GATE0_STATUS.md ..."
git rm refine-logs/GATE0_STATUS.md

# ===========================================================================
# Step 4：删除 19 个旧快照
# ===========================================================================
echo ""
echo "[Step 4/8] 删除 19 个旧快照文件..."

OLD_SNAPSHOTS=(
    # EXPERIMENT_PLAN 历史 (3)
    "refine-logs/EXPERIMENT_PLAN.md"
    "refine-logs/EXPERIMENT_PLAN_20260528_200150.md"
    "refine-logs/EXPERIMENT_PLAN_20260528_200527.md"
    # EXPERIMENT_TRACKER 历史 (3)
    "refine-logs/EXPERIMENT_TRACKER.md"
    "refine-logs/EXPERIMENT_TRACKER_20260528_200150.md"
    "refine-logs/EXPERIMENT_TRACKER_20260528_200527.md"
    # FINAL_PROPOSAL 历史 (5)
    "refine-logs/FINAL_PROPOSAL.md"
    "refine-logs/FINAL_PROPOSAL_20260528_200150.md"
    "refine-logs/FINAL_PROPOSAL_20260528_200527.md"
    "refine-logs/FINAL_PROPOSAL_v2.md"
    "refine-logs/FINAL_PROPOSAL_v3.md"
    # PIPELINE_SUMMARY 历史 (2)
    "refine-logs/PIPELINE_SUMMARY_20260528_200150.md"
    "refine-logs/PIPELINE_SUMMARY_20260528_200527.md"
    # REFINEMENT_REPORT (3，全删)
    "refine-logs/REFINEMENT_REPORT.md"
    "refine-logs/REFINEMENT_REPORT_20260528_200150.md"
    "refine-logs/REFINEMENT_REPORT_20260528_200527.md"
    # REVIEW_SUMMARY (3，全删)
    "refine-logs/REVIEW_SUMMARY.md"
    "refine-logs/REVIEW_SUMMARY_20260528_200150.md"
    "refine-logs/REVIEW_SUMMARY_20260528_200527.md"
)

for f in "${OLD_SNAPSHOTS[@]}"; do
    if [[ -f "$f" ]]; then
        git rm "$f"
    else
        echo "    WARN: $f 不存在，跳过"
    fi
done

# ===========================================================================
# Step 5：删除空 refine-logs/ 目录
# ===========================================================================
echo ""
echo "[Step 5/8] 删除空目录 refine-logs/ ..."
remaining=$(find refine-logs -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
if [[ "$remaining" -ne 0 ]]; then
    echo "ERROR: refine-logs/ 内仍有 $remaining 个残留项，无法 rmdir"
    find refine-logs -mindepth 1 -maxdepth 1
    exit 1
fi
# 上一次运行 git rm 完所有受跟踪文件后，git 会自动让目录变为不可见；
# 这里 rmdir 兜底处理本地残留的空目录
if [[ -d "refine-logs" ]]; then
    rmdir refine-logs
else
    echo "    refine-logs/ 已被 git 自动清理"
fi

fi  # end of REFINE_LOGS_EXISTS block

# ===========================================================================
# Step 6：删除 4 个 .gitkeep（仅删占位，保留目录，目录里有实际内容）
# ===========================================================================
echo ""
echo "[Step 6/8] 删除 4 个保留目录的 .gitkeep ..."

KEEP_DIRS_GITKEEPS=(
    "02_baseline_reasonrag/scripts/.gitkeep"
    "03_sapr_rag/scripts/.gitkeep"
    "04_experiments/metrics/.gitkeep"
    "06_notes/idea_notes/.gitkeep"
)

for f in "${KEEP_DIRS_GITKEEPS[@]}"; do
    if [[ -f "$f" ]]; then
        git rm "$f"
    else
        echo "    WARN: $f 不存在，跳过"
    fi
done

# ===========================================================================
# Step 7：删除 27 个空目录（含其中的 .gitkeep）
# ===========================================================================
echo ""
echo "[Step 7/8] 删除 27 个空目录及其 .gitkeep ..."

EMPTY_DIRS=(
    # 00_project_management (3)
    "00_project_management/meeting_notes"
    "00_project_management/progress_reports"
    "00_project_management/weekly_plans"
    # 01_literature (2)
    "01_literature/paper_tables"
    "01_literature/related_work_drafts"
    # 02_baseline_reasonrag (5)
    "02_baseline_reasonrag/analysis"
    "02_baseline_reasonrag/badcases"
    "02_baseline_reasonrag/configs"
    "02_baseline_reasonrag/results"
    "02_baseline_reasonrag/trajectories"
    # 03_sapr_rag (7)
    "03_sapr_rag/ablations"
    "03_sapr_rag/configs"
    "03_sapr_rag/evidence_reward"
    "03_sapr_rag/query_reward"
    "03_sapr_rag/results"
    "03_sapr_rag/reward_prompts"
    "03_sapr_rag/stop_reward"
    # 04_experiments (2)
    "04_experiments/figures"
    "04_experiments/tables"
    # 05_reports (3)
    "05_reports/aaai2027_paper"
    "05_reports/master_thesis"
    "05_reports/midterm_report"
    # 06_notes (2)
    "06_notes/debug_notes"
    "06_notes/writing_notes"
    # 07_assets (3)
    "07_assets/diagrams"
    "07_assets/figures"
    "07_assets/slides"
)

for d in "${EMPTY_DIRS[@]}"; do
    if [[ ! -d "$d" ]]; then
        echo "    WARN: $d 不存在，跳过"
        continue
    fi

    # 先删该目录下唯一的 .gitkeep（受 git 跟踪）
    gk="$d/.gitkeep"
    if [[ -f "$gk" ]]; then
        git rm "$gk"
    fi

    # 校验目录现在确实为空
    leftovers=$(find "$d" -mindepth 1 | wc -l)
    if [[ "$leftovers" -ne 0 ]]; then
        echo "    ERROR: $d 删 .gitkeep 后仍有内容，跳过 rmdir"
        find "$d" -mindepth 1
        continue
    fi

    rmdir "$d"
    echo "    rmdir $d"
done

# ===========================================================================
# Step 8：输出最终 git status，方便用户 review 后再 commit
# ===========================================================================
echo ""
echo "[Step 8/8] 清理完成，当前 git status:"
echo "----------------------------------------------------------------"
git status
echo "----------------------------------------------------------------"

echo ""
echo "================================================================"
echo "清理脚本执行完毕。"
echo ""
echo "下一步建议（手动执行，脚本不自动 commit）:"
echo "  1) 检查上方 git status 输出无误"
echo "  2) git diff --stat   # 看变更体量"
echo "  3) git add -A && git commit -m 'chore: clean up refine-logs and empty placeholder dirs'"
echo "================================================================"

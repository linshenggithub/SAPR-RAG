#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-grpo_reward_v3_marginal_w015_s500_$(date +%Y%m%d_%H%M%S)}"
export REWARD_FUNCS="${REWARD_FUNCS:-sapr_f1 sapr_marginal_relevance sapr_format sapr_turn_cost sapr_repeat_query sapr_max_turn}"
export REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0 0.15 0.05 0.02 0.15 0.50}"

bash "$SCRIPT_DIR/launch_grpo_reward_v2_lora.sh"

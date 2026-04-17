#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-coll_venv}"
DRY_RUN="${DRY_RUN:-0}"
if [ -d "$VENV_DIR/bin" ]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
fi

CONFIG_DIR="${CONFIG_DIR:-conf/config_PPO_exponential_const420}"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: config directory not found: $CONFIG_DIR"
    exit 1
fi

mapfile -t CONFIG_FILES < <(find "$CONFIG_DIR" -maxdepth 1 -name 'config_PPO_expo*_const420.yaml' | sort)

if [ "${#CONFIG_FILES[@]}" -eq 0 ]; then
    echo "Error: no config_PPO_expo*_const420.yaml files found in $CONFIG_DIR"
    exit 1
fi

echo "Running PPO exponential const420 configs serially"
echo "Config directory: $CONFIG_DIR"
echo "Number of configs: ${#CONFIG_FILES[@]}"
echo "Resume policy: skip configs whose exp/<group>/log_data.pkl already exists"
echo "Dry run: $DRY_RUN"
echo "================================================"

for config_path in "${CONFIG_FILES[@]}"; do
    config_name="$(basename "$config_path" .yaml)"
    IFS=$'\t' read -r group_name expected_num_iters expected_num_seeds < <(python3 - "$config_path" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
resolved = OmegaConf.to_container(cfg, resolve=True)
print(
    resolved["wandb"]["group"],
    resolved["num_iters"],
    resolved["num_seeds"],
    sep="\t",
)
PY
)
    output_dir="exp/${group_name}"
    done_marker="${output_dir}/log_data.pkl"
    args_path="${output_dir}/args.pkl"

    echo
    echo "[${config_name}]"
    echo "group: ${group_name}"
    echo "output: ${output_dir}"

    if [ -f "$done_marker" ]; then
        if [ ! -f "$args_path" ]; then
            echo "Existing log found, but ${args_path} is missing; rerunning."
        else
            match_status="$(python3 - "$args_path" "$expected_num_iters" "$expected_num_seeds" <<'PY'
import pickle
import sys

args_path = sys.argv[1]
expected_num_iters = int(sys.argv[2])
expected_num_seeds = int(sys.argv[3])

with open(args_path, "rb") as f:
    args = pickle.load(f)

if isinstance(args, dict):
    actual_num_iters = args.get("num_iters")
    actual_num_seeds = args.get("num_seeds")
else:
    actual_num_iters = getattr(args, "num_iters", None)
    actual_num_seeds = getattr(args, "num_seeds", None)

if actual_num_iters == expected_num_iters and actual_num_seeds == expected_num_seeds:
    print("match")
else:
    print(f"mismatch\t{actual_num_iters}\t{actual_num_seeds}")
PY
)"
            IFS=$'\t' read -r match_tag actual_num_iters actual_num_seeds <<< "$match_status"
            if [ "$match_tag" = "match" ]; then
                echo "Skipping: found completed run with num_iters=${expected_num_iters}, num_seeds=${expected_num_seeds}"
                continue
            fi
            echo "Existing log found, but args mismatch (num_iters=${actual_num_iters:-missing}, num_seeds=${actual_num_seeds:-missing}); rerunning."
        fi
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "Would run: python main.py --config-path conf/config_PPO_exponential_const420 --config-name ${config_name}"
        continue
    fi

    echo "Starting training..."
    python main.py --config-path conf/config_PPO_exponential_const420 --config-name "${config_name}"
    echo "Finished: ${config_name}"

    if [ ! -f "$done_marker" ]; then
        echo "Error: run finished but ${done_marker} was not created"
        exit 1
    fi
done

echo
echo "================================================"
echo "All PPO exponential const420 configs processed."

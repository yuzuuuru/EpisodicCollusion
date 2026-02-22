#!/bin/bash
# Usage: ./run_all_plots.sh [save_dir]
# Example: ./run_all_plots.sh exp/DQN

SAVE_DIR="${1:-exp/DQN}"

if [ ! -d "$SAVE_DIR" ]; then
    echo "Error: Directory $SAVE_DIR does not exist"
    exit 1
fi

echo "Running all plotting scripts for: $SAVE_DIR"
echo "================================================"

export EC_SAVE_DIR="$SAVE_DIR"

# Figure 2: Training results
echo "Running plotting_fig2_trainresult.py..."
python plotting_fig2_trainresult.py --save_dir "$SAVE_DIR"

# Figure 3a: Deviation analysis
echo "Running plotting_fig3a_deviation.py..."
python plotting_fig3a_deviation.py

# Figure 3b: Reaction surface
echo "Running plotting_fig3b_reaction_suface.py..."
python plotting_fig3b_reaction_suface.py

# Prices plot
echo "Running plotting_prices.py..."
python plotting_prices.py

echo "================================================"
echo "All plots completed! Results saved in: $SAVE_DIR/paper_plots/"

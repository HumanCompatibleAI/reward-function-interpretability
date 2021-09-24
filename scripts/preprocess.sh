#!/bin/bash

# quit on errors, forbid undefined variables
set -euo pipefail

DRLHP=0
AIRL=0
TRUTH=0
FAST=""

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --airl)
      AIRL=1
      shift
      ;;
    --ground-truth)
      TRUTH=1
      shift
      ;;
    --fast)
      FAST="fast"
      shift
      ;;
    --drlhp)
      DRLHP=1
      shift
      ;;
    --all)
      DRLHP=1
      AIRL=1
      TRUTH=1
      shift
      ;;
  esac
done


if [[ $DRLHP == 1 ]]; then
  mkdir -p processed/preference_comparisons
  for size in 4 10; do
    MODEL_PATHS=$(find results/preference_comparisons -type f -path "*/empty_maze_$size*/final_reward_net.pt" -printf "%P\n" | sed 's/\/final_reward_net\.pt$//')
    for path in $MODEL_PATHS; do
      echo "Processing results/preference_comparisons/$path"
      poetry run python -m reward_preprocessing.optimize_tabular with \
        env.empty_maze_$size $FAST \
        "model_path=results/preference_comparisons/$path/final_reward_net.pt" \
        "save_path=processed/preference_comparisons/$path"
    done
  done
fi

if [[ $AIRL == 1 ]]; then
  mkdir -p processed/adversarial
  for size in 4 10; do
    MODEL_PATHS=$(find results/adversarial -type f -path "*/empty_maze_$size*/checkpoints/final/reward_test.pt" -printf "%P\n" | sed 's/\/checkpoints\/final\/reward_test\.pt$//')
    for path in $MODEL_PATHS; do
      echo "Processing results/adversarial/$path"
      poetry run python -m reward_preprocessing.optimize_tabular with \
        env.empty_maze_$size $FAST \
        "model_path=results/adversarial/$path/checkpoints/final/reward_test.pt" \
        "save_path=processed/adversarial/$path"
    done
  done
fi

if [[ $TRUTH == 1 ]]; then
  mkdir -p processed/ground_truth
  for size in 4 10; do
    MODEL_PATHS=$(find results/ground_truth_models -type f -path "*/empty_maze_$size*.pt" -printf "%P\n" | sed 's/\.pt$//')
    for path in $MODEL_PATHS; do
      echo "Processing results/ground_truth_models/$path"
      poetry run python -m reward_preprocessing.optimize_tabular with \
        env.empty_maze_$size $FAST \
        "model_path=results/ground_truth_models/$path.pt" \
        "save_path=processed/ground_truth/$path"
    done
  done
fi
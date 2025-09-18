#!/bin/bash
set -e

seeds=(0 48 36 23 80)

echo "Running ECNews dataset with 5 seeds..."
for seed in "${seeds[@]}"; do
    echo "Running ECNews with seed=$seed, k_neighbors=3, weight_kl_loss=4, weight_cka_loss=450"
    python main.py --model GloCTM --seed $seed --dataset ECNews --k_neighbors 3 --weight_kl_loss 4 --weight_cka_loss 450
done

echo "Running Amazon_Review dataset with 5 seeds..."
for seed in "${seeds[@]}"; do
    echo "Running Amazon_Review with seed=$seed, k_neighbors=3, weight_kl_loss=3, weight_cka_loss=300"
    python main.py --model GloCTM --seed $seed --dataset Amazon_Review --k_neighbors 3 --weight_kl_loss 3 --weight_cka_loss 300
done

echo "Running Rakuten_Amazon dataset with 5 seeds..."
for seed in "${seeds[@]}"; do
    echo "Running Rakuten_Amazon with seed=$seed, k_neighbors=16, weight_kl_loss=7, weight_cka_loss=400"
    python main.py --model GloCTM --seed $seed --dataset Rakuten_Amazon --k_neighbors 12 --weight_kl_loss 2 --weight_cka_loss 450
done

echo "All experiments completed!"
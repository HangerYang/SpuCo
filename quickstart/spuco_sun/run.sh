#!/bin/bash

# Run commands with all combinations of parameters
for lr in 1 5 10 20; do
    for wd in 0.01 0.005 0.0001; do
        for iter in 1 5 10 20; do
            for difficulty in VARIANCE_HIGH VARIANCE_MEDIUM VARIANCE_LOW; do
                wandb_run_name="mnist_${lr}_${wd}_${iter}_${difficulty}"
                python spuco_mnist_ssa.py --gpu 4 --infer_lr $lr --infer_weight_decay $wd --infer_num_iters $iter --wandb --wandb_run_name "$wandb_run_name" --difficulty $difficulty
            done
        done
    done
done

# End of script

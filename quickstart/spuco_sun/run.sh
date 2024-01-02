<<<<<<< HEAD
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
=======
python spuco_sun_lrmix.py --root_dir /data/spucosun/6.0 --gpu 4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1e-4 --wandb_run_name sun_lrmix_6.0

python spuco_sun_lrmix.py --root_dir /data/spucosun/6.1 --gpu 4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1e-4 --wandb_run_name sun_lrmix_6.1

python spuco_sun_lrmix.py --root_dir /data/spucosun/6.2 --gpu 4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1e-4 --wandb_run_name sun_lrmix_6.2
>>>>>>> 7f31f9029ce3132559c148f7757bc8477080e000

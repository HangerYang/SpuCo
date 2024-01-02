# group_dro variance check

python spuco_sun_ssa.py --root_dir /data/spucosun/6.0 --gpu 2 --inf_lr 1e-5 --inf_num_iters 40 --inf_weight_decay 1e-4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1 --wandb_run_name sun_ssa_6.0

python spuco_sun_ssa.py --root_dir /data/spucosun/6.1 --gpu 2 --inf_lr 1e-5 --inf_num_iters 40 --inf_weight_decay 1e-4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1 --wandb_run_name sun_ssa_6.1

python spuco_sun_ssa.py --root_dir /data/spucosun/6.2 --gpu 2 --inf_lr 1e-5 --inf_num_iters 40 --inf_weight_decay 1e-4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1 --wandb_run_name sun_ssa_6.2

# group_dro variance check

python spuco_sun_cnc.py --root_dir /data/spucosun/6.0 --gpu 1 --inf_lr 1e-5 --inf_num_epochs 40 --inf_weight_decay 1e-4 --arch resnet50 --pretrained --wandb --train_lr 1e-4 --train_wd 1e-3 --train_num_epochs 5 --wandb_run_name sun_cnc_6.0

python spuco_sun_cnc.py --root_dir /data/spucosun/6.1 --gpu 1 --inf_lr 1e-5 --inf_num_epochs 40 --inf_weight_decay 1e-4 --arch resnet50 --pretrained --wandb --train_lr 1e-4 --train_wd 1e-3 --train_num_epochs 5 --wandb_run_name sun_cnc_6.1

python spuco_sun_cnc.py --root_dir /data/spucosun/6.2 --gpu 1 --inf_lr 1e-5 --inf_num_epochs 40 --inf_weight_decay 1e-4 --arch resnet50 --pretrained --wandb --train_lr 1e-4 --train_wd 1e-3 --train_num_epochs 5 --wandb_run_name sun_cnc_6.2

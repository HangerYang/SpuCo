# group_dro variance check
python spuco_sun_erm_gb.py --root_dir /data/spucosun/6.0 --gpu 0 --lr 1e-4 --weight_decay 1e-4 --arch resnet50 --pretrained --wandb --num_epochs 40 --wandb_run_name gb_6.0

python spuco_sun_erm_gb.py --root_dir /data/spucosun/6.1 --gpu 0 --lr 1e-4 --weight_decay 1e-4 --arch resnet50 --pretrained --wandb --num_epochs 40 --wandb_run_name gb_6.1

python spuco_sun_erm_gb.py --root_dir /data/spucosun/6.2 --gpu 0 --lr 1e-4 --weight_decay 1e-4 --arch resnet50 --pretrained --wandb --num_epochs 40 --wandb_run_name gb_6.2

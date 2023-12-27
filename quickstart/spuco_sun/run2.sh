python spuco_sun_ssa.py --root_dir /data/spucosun/6.0 --gpu 2 --inf_lr 1e-5 --inf_num_iters 40 --inf_weight_decay 1e-4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1 --wandb_run_name sun_ssa_6.0
python spuco_sun_ssa.py --root_dir /data/spucosun/6.1 --gpu 2 --inf_lr 1e-5 --inf_num_iters 40 --inf_weight_decay 1e-4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1 --wandb_run_name sun_ssa_6.1
python spuco_sun_ssa.py --root_dir /data/spucosun/6.2 --gpu 2 --inf_lr 1e-5 --inf_num_iters 40 --inf_weight_decay 1e-4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1 --wandb_run_name sun_ssa_6.2
python spuco_sun_cnc.py --root_dir /data/spucosun/6.0 --gpu 1 --inf_lr 1e-5 --inf_num_epochs 40 --inf_weight_decay 1e-4 --arch resnet50 --pretrained --wandb --train_lr 1e-4 --train_wd 1e-3 --train_num_epochs 5 --wandb_run_name sun_cnc_6.0
python spuco_sun_cnc.py --root_dir /data/spucosun/6.1 --gpu 1 --inf_lr 1e-5 --inf_num_epochs 40 --inf_weight_decay 1e-4 --arch resnet50 --pretrained --wandb --train_lr 1e-4 --train_wd 1e-3 --train_num_epochs 5 --wandb_run_name sun_cnc_6.1
python spuco_sun_cnc.py --root_dir /data/spucosun/6.2 --gpu 1 --inf_lr 1e-5 --inf_num_epochs 40 --inf_weight_decay 1e-4 --arch resnet50 --pretrained --wandb --train_lr 1e-4 --train_wd 1e-3 --train_num_epochs 5 --wandb_run_name sun_cnc_6.2
python spuco_sun_dfr.py --root_dir /data/spucosun/6.0 --gpu 3 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1e-4 --wandb_run_name sun_dfr_6.0
python spuco_sun_dfr.py --root_dir /data/spucosun/6.1 --gpu 3 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1e-4 --wandb_run_name sun_dfr_6.1
python spuco_sun_dfr.py --root_dir /data/spucosun/6.2 --gpu 3 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1e-4 --wandb_run_name sun_dfr_6.2
python spuco_sun_lrmix.py --root_dir /data/spucosun/6.0 --gpu 4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1e-4 --wandb_run_name sun_lrmix_6.0
python spuco_sun_lrmix.py --root_dir /data/spucosun/6.1 --gpu 4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1e-4 --wandb_run_name sun_lrmix_6.1
python spuco_sun_lrmix.py --root_dir /data/spucosun/6.2 --gpu 4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1e-4 --wandb_run_name sun_lrmix_6.2
python spuco_sun_lff.py --root_dir /data/spucosun/6.0 --gpu 4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1e-4 --wandb_run_name sun_lff_6.0
python spuco_sun_lff.py --root_dir /data/spucosun/6.1 --gpu 4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1e-4 --wandb_run_name sun_lff_6.1
python spuco_sun_lff.py --root_dir /data/spucosun/6.2 --gpu 4 --arch resnet50 --pretrained --wandb --num_epochs 40 --lr 1e-5 --weight_decay 1e-4 --wandb_run_name sun_lff_6.2

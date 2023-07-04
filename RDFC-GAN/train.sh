# Please modify the settings as you see fit
python train.py \
    --dataset nyuv2 \
    --data_root ./datasets/nyuv2 \
    --batch_size 4 \
    --model_cfg_path ./RDFC-GAN/config/rdf_cycle_patchgan_config.yaml \
    --work_dir ./RDFC-GAN/test_code_training \
    --gpus 0 \
    --num_classes 14 \
    --label_wall 12 \
    --label_floor 5 \
    --label_ceiling 3
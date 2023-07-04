# Please modify the settings as you see fit

python test.py \
    --dataset nyuv2 \
    --data_root ./datasets/nyuv2 \
    --model_cfg_path ./RDFC-GAN/config/rdf_cycle_patchgan_config.yaml \
    --work_dir ./RDFC-GAN/test_code \
    --load_from ./RDFC-GAN/checkpoints/best.pth \
    --gpus 0 \
    --batch_size 1 \
    --out_height 256 \
    --out_width 256

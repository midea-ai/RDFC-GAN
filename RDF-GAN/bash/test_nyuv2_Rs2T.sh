#!/usr/bin/bash
# random sampled raw depth ---> GTi

python lib/tools/main_vanilla.py \
	--dataset nyuv21400_s2d \
	--data_root data/nyuv2 \
	--work_dir work_dir/nyuv2_Rs2T \
	--height 228 \
	--width 304 \
	--num_classes 40 \
	--guidance_encoder resnet34 \
	--guidance_encoder_block BasicBlock \
	--guidance_channels_decoder 512 256 128 \
	--guidance_nr_decoder_blocks 3 3 3 \
	--guidance_encoder_decoder_fusion add \
	--guidance_context_module ppm \
	--guidance_weighting_in_encoder SE-add \
	--guidance_upsampling learned-3x3-zeropad \
	--reduction_glo_guid_module \
	--glo_guid_channels_out_0 40 \
	--glo_guid_channels_out_1 1 \
	--separate_global_guidance_module \
	--encoder_rgb resnet18 \
	--encoder_depth resnet18 \
	--encoder_block BasicBlock \
	--rgb_channels_decoder 512 256 128 64 64 \
	--depth_channels_decoder 512 256 128 64 64 \
	--nr_decoder_blocks 3 3 3 0 0 \
	--fuse_depth_in_rgb_encoder None \
	--fuse_depth_in_rgb_decoder AdaIN \
	--encoder_decoder_fusion add \
	--activation LeakyReLU \
	--norm_layer_type IN2d \
	--upsampling_mode bilinear \
	--adain_weighting \
	--inference \
	--batch_size 1 \
	${@:1}


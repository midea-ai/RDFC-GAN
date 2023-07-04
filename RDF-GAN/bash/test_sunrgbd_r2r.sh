#!/usr/bin/bash


python lib/tools/main.py \
	--dataset sunrgbd \
	--data_root data/sunrgbd \
	--height 480 \
	--width 640 \
	--guidance_encoder resnet50 \
	--guidance_encoder_block Bottleneck \
	--guidance_channels_decoder 512 256 128 \
	--guidance_nr_decoder_blocks 3 3 3 \
	--guidance_encoder_decoder_fusion add \
	--guidance_context_module ppm \
	--guidance_weighting_in_encoder SE-add \
	--guidance_upsampling learned-3x3-zeropad \
	--num_classes 37 \
	--semantic_channels_in 37 \
	--encoder_rgb resnet18 \
	--encoder_depth resnet18 \
	--rgb_channels_encoder 64 64 128 256 512 512 \
	--depth_channels_encoder 64 64 128 256 512 512 \
	--rgb_channels_decoder 256 128 64 64 \
	--depth_channels_decoder 256 128 64 64 \
	--fuse_depth_in_rgb_decoder AdaIN \
	--rgb_encoder_decoder_fusion concat \
	--depth_encoder_decoder_fusion concat \
	--activation LeakyReLU \
	--norm_layer_type IN2d \
	--adain_weighting \
	--use_nlspn_to_refine \
	--inference \
	--seed 4828 \
	--prop_time 12 \
	${@:1}


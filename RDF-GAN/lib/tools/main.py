import os
from config import args
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))

import torch
from torch import distributed as dist
from lib.utils.seed_all import set_random_seed
import torch.optim as optim
from lib.tools.helper import (Logger, MovingAverage, reduce_loss,
                              save_checkpoint, init_weights, resume_from, load_checkpoint)

from torch.autograd import grad
from lib.evaluator.rdf_gan_evaluator import (Eval, DistEval)
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def build_model(args):
    from lib.models.segmentator.esa_net.esa_net_one_modality import ESANetOneModality
    from lib.models.generator.rdf_gan_generator.rdf_gan_generator import DCVGANGenerator
    from lib.models.discriminator.patch_gan_discriminator import PatchGANDiscriminator

    # build the global_guidance_module, use U-net or ESANet-One-Modality
    encoder_decoder_module = ESANetOneModality(height=args.height,
                                               width=args.width,
                                               num_classes=args.num_classes,
                                               pretrained_on_imagenet=args.pretrained_on_imagenet,
                                               pretrained_dir=args.guidance_pretrained_dir,
                                               encoder=args.guidance_encoder,
                                               encoder_block=args.guidance_encoder_block,
                                               channels_decoder=args.guidance_channels_decoder,
                                               nr_decoder_blocks=args.guidance_nr_decoder_blocks,
                                               encoder_decoder_fusion=args.guidance_encoder_decoder_fusion,
                                               context_module=args.guidance_context_module,
                                               weighting_in_encoder=args.guidance_weighting_in_encoder,
                                               upsampling=args.guidance_upsampling,
                                               pyramid_supervision=False)
    # if args.use_pretrained_encoder_decoder:
    #     from lib.tools.train_segmentator import load_checkpoint as load_segmentator_checkpoint
    #     checkpoint = load_segmentator_checkpoint(encoder_decoder_module,
    #                                              args.load_encoder_decoder_path,
    #                                              map_location=torch.device('cpu'))


    generator = DCVGANGenerator(global_guidance_module=encoder_decoder_module,
                                encoder_rgb=args.encoder_rgb,
                                encoder_depth=args.encoder_depth,
                                semantic_channels_in=args.semantic_channels_in,
                                rgb_channels_encoder=args.rgb_channels_encoder,
                                depth_channels_encoder=args.rgb_channels_encoder,
                                rgb_channels_decoder=args.rgb_channels_decoder,
                                depth_channels_decoder=args.depth_channels_decoder,
                                pretrained_on_imagenet=True if not args.inference else False,
                                fuse_depth_in_rgb_decoder=None if args.fuse_depth_in_rgb_decoder == 'None' else args.fuse_depth_in_rgb_decoder,
                                adain_weighting=args.adain_weighting,
                                rgb_skip_connection_type=args.rgb_encoder_decoder_fusion,
                                depth_skip_connection_type=args.depth_encoder_decoder_fusion,
                                use_nlpsn_refine=args.use_nlspn_to_refine,
                                nlspn_configs=dict(prop_kernel=args.prop_kernel,
                                                   prop_time=args.prop_time,
                                                   preserve_input=args.preserve_input,
                                                   affinity=args.affinity,
                                                   affinity_gamma=args.affinity_gamma,
                                                   conf_prop=args.conf_prop,
                                                   ))

    discriminator = PatchGANDiscriminator(in_channels=1,
                                          norm_cfg=dict(type='BN') if args.disc_norm_type.lower() == 'bn' else dict(type='IN'),
                                          activation='LeakyReLU' if args.disc_act_type.lower() == 'leakyrelu' else 'ReLU')

    # generator's parameters are initialized inside the class

    # initializes the discriminator parameters explicitly
    if args.init_disc:
        init_weights(discriminator)

    return generator, discriminator


def get_dataloader(args):
    from lib.dataset.build_dataloader import build_dataloader

    if args.dataset == 'sunrgbd':
        from lib.dataset.sunrgbd.sunrgbd_dataset import SUNRGBDPseudoDataset
        DATASET = SUNRGBDPseudoDataset

        dataset_kwargs = dict(max_depth=10.0,
                              rgb_mean=[0.485, 0.456, 0.406],
                              rgb_std=[0.229, 0.224, 0.225],
                              depth_mean=[5.0],
                              depth_std=[5.0],
                              )

    elif args.dataset == 'nyudepthv2_s2d':
        from lib.dataset.nyuv2.nyuv2_sparse_to_dense_dataset import NYUV2S2DDataset
        DATASET = NYUV2S2DDataset

        dataset_kwargs = dict(num_sample=500,
                              max_depth=10.0,
                              depth_mean=[5.0],
                              depth_std=[5.0]
                              )

    elif args.dataset == 'nyudepthv2_r2r':
        from lib.dataset.nyuv2.nyuv2_raw_to_reconstructed_dataset import NYUV2R2RDataset

        DATASET = NYUV2R2RDataset

        dataset_kwargs = dict(num_sample=500,
                              max_depth=10.0,
                              depth_mean=[5.0],
                              depth_std=[5.0]
                              )
    elif args.dataset == 'nyuv21400_s2d':
        from lib.dataset.nyuv2.nyuv2_1400_sparse_to_dense_dataset import NYUV21400S2DDataset

        DATASET = NYUV21400S2DDataset
        dataset_kwargs = dict(num_sample=500,
                              max_depth=10.0,
                              depth_mean=[5.0],
                              depth_std=[5.0]
                              )

    elif args.dataset == 'ddrnet_human':
        from lib.dataset.ddrnet_human.ddrnet_human_dataset import DDRNetHumanDataset

        DATASET = DDRNetHumanDataset
        dataset_kwargs = dict(max_depth=3.0,
                              depth_mean=[1.5],
                              depth_std=[1.5],
                              times=20)

    else:
        raise NotImplementedError(f'Only SUN RGBD, NYUDepthv2, are supported so far, '
                                  f'but got {args.dataset}')


    train_dataset = DATASET(data_root=args.data_root,
                            mode='train',
                            **dataset_kwargs)

    val_dataset = DATASET(data_root=args.data_root,
                          mode='test' if args.dataset in ['nyudepthv2_s2d', 'ddrnet_human'] else 'val',
                          **dataset_kwargs)

    train_dataloader = build_dataloader(train_dataset,
                                        samples_per_gpu=args.batch_size,
                                        workers_per_gpu=args.num_workers,
                                        num_gpus=len(args.gpus),
                                        dist=len(args.gpus) > 1,
                                        seed=args.seed,
                                        pin_memory=False,
                                        drop_last=True)
    val_dataloader = build_dataloader(val_dataset,
                                      samples_per_gpu=args.batch_size,
                                      workers_per_gpu=0,
                                      num_gpus=len(args.gpus),
                                      dist=len(args.gpus) > 1,
                                      pin_memory=False,
                                      shuffle=False)

    return train_dataloader, val_dataloader


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def main():
    EPSION = 1e-6

    args.gpus = [int(i) for i in args.gpus.split(',')]
    distributed = args.num_gpus > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir, exist_ok=True)

    logger = Logger(name='RDF-GAN', local_rank=args.local_rank, save_dir=args.work_dir)

    logger.log(str(args))

    logger.log(f'Set random seed to {args.seed}')

    if args.local_rank == 0:
        args.sample_dir = os.path.join(args.work_dir, 'samples')
        if not os.path.exists(args.sample_dir):
            os.mkdir(args.sample_dir)

    set_random_seed(args.seed)

    # build dataloader
    train_dataloader, val_dataloader = get_dataloader(args)
    fixed_samples = next(iter(val_dataloader))

    # build model, including generator and discriminator
    generator, discriminator = build_model(args)

    # put model to corresponding device
    device = torch.device("cuda", args.local_rank)
    if not distributed:
        generator = generator.to(device)
        discriminator = discriminator.to(device)
    else:
        generator = torch.nn.parallel.DistributedDataParallel(generator.to(device),
                                                              device_ids=[args.local_rank],
                                                              # find_unused_parameters=True
                                                              )
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator.to(device),
                                                                  device_ids=[args.local_rank],
                                                                  # find_unused_parameters=True
                                                                  )

    evaluator = Eval(val_dataloader, device=device) if not distributed else DistEval(val_dataloader, device=device)

    # optimizer hyper parameters setting
    learning_rate = args.learning_rate
    beta1, beta2 = args.beta1, args.beta2

    # filter parameters that do not want to be updated here
    if args.use_pretrained_encoder_decoder:
        generator_params = [{"params": [p for n, p in generator.named_parameters() if 'global_guidance_module' not in n and p.requires_grad]},
                            {"params": [p for n, p in generator.named_parameters() if 'global_guidance_module' in n and p.requires_grad],
                             "lr": learning_rate * 0.5}]
    else:
        generator_params = generator.parameters()

    if args.optimizer.lower() == 'adam':
        optimizer_generator = optim.Adam(generator_params, lr=learning_rate, betas=(beta1, beta2))
        optimizer_disc = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    elif args.optimizer.lower() == 'sgd':
        optimizer_generator = optim.SGD(generator_params, lr=learning_rate)
        optimizer_disc = optim.SGD(discriminator.parameters(), lr=learning_rate)

    elif args.optimizer.lower() == 'rmsprop':
        optimizer_generator = optim.RMSprop(generator_params, lr=learning_rate)
        optimizer_disc = optim.RMSprop(discriminator.parameters(), lr=learning_rate)

    else:
        raise NotImplementedError(f'Only Adam, SGD, RMSprop optimizers are supported, but got {args.optimizer}')

    if args.lr_scheduler == 'step':
        lr_scheduler_generator = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_generator,
                                                                milestones=args.lr_decay_epochs,
                                                                gamma=args.lr_decay_rate)
        lr_scheduler_disc = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_disc,
                                                           milestones=args.lr_decay_epochs,
                                                           gamma=args.lr_decay_rate)

    elif args.lr_scheduler == 'onecycle':
        lr_scheduler_generator = optim.lr_scheduler.OneCycleLR(optimizer=optimizer_generator,
                                                               max_lr=[i['lr'] for i in optimizer_generator.param_groups],
                                                               total_steps=args.max_epoch,
                                                               div_factor=args.div_factor,
                                                               pct_start=args.pct_start,
                                                               anneal_strategy='cos',
                                                               final_div_factor=1e4)
        lr_scheduler_disc = optim.lr_scheduler.OneCycleLR(optimizer=optimizer_disc,
                                                              max_lr=[i['lr'] for i in optimizer_disc.param_groups],
                                                              total_steps=args.max_epoch,
                                                              div_factor=args.div_factor,
                                                              pct_start=args.pct_start,
                                                              anneal_strategy='cos',
                                                              final_div_factor=1e4)
    elif args.lr_scheduler == 'cosine':
        # FIXME: The CosineAnnealingLR used before has problems, the final learning rate will not drop to zero, it only take half a cycle.

        lr_scheduler_generator = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_generator,
                                                                      # T_max=args.max_epoch + 10
                                                                      T_max=args.t_max
                                                                      )
        lr_scheduler_disc = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_disc,
                                                                     # T_max=args.max_epoch + 10,
                                                                     T_max=args.t_max
                                                                     )
    elif args.lr_scheduler == "lambda":
        from lib.tools.helper import LRFactor
        lr_calculator = LRFactor(args.lambda_lr_decay, args.lambda_lr_gamma)
        lr_scheduler_generator = optim.lr_scheduler.LambdaLR(optimizer=optimizer_generator,
                                                             lr_lambda=lr_calculator.get_factor)
        lr_scheduler_disc = optim.lr_scheduler.LambdaLR(optimizer=optimizer_disc,
                                                        lr_lambda=lr_calculator.get_factor)
    else:
        raise NotImplementedError(f'Only multi step, cosine annealing and onecycle lr scheduler are supported'
                                  f'but got {args.lr_scheduler}')

    start_epoch = 0
    num_iters_per_epoch = len(train_dataloader)

    if args.resume_from is not None:
        # warp the parts to dict type
        start_epoch = resume_from(models=dict(generator=generator,
                                              discriminator=discriminator),
                                  optimizers=dict(generator=optimizer_generator,
                                                  discrimiator=optimizer_disc),
                                  lr_schedulers=dict(generator=lr_scheduler_generator,
                                                     discriminator=lr_scheduler_disc),
                                  filename=args.resume_from,
                                  logger=logger)

    if args.load_from is not None:
        logger.log(f'load checkpoint from {args.load_from}')
        checkpoint = load_checkpoint(model=dict(generator=generator,
                                                discriminator=discriminator),
                                     filename=args.load_from,
                                     map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()),
                                     logger=logger)

    if args.cal_fps:
        num_warmup = 5
        pure_inf_time = 0
        log_interval = 50
        samples = 300

        generator.eval()

        for i, data in enumerate(val_dataloader):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                generator(data['rgb'].to(device),
                          data['raw_depth'].to(device))

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(f'Done image [{i + 1:<3}/ {samples}], '
                          f'fps: {fps:.1f} img / s')

            if (i + 1) == samples:
                pure_inf_time += elapsed
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Overall fps: {fps:.1f} img / s')
                break
        generator.train()
        exit(0)

    def colored_depth_map(depth, d_min=None, d_max=None, cmap=plt.cm.viridis):
        if d_min is None:
            d_min = np.min(depth)
        if d_max is None:
            d_max = np.max(depth)

        depth_relative = (depth - d_min) / (d_max - d_min)
        return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C

    def save_img(arr, save_path):
        img = Image.fromarray(arr)
        img.save(save_path)


    if args.inference:
        # from lib.dcvgan_tools.train_dcvgan_helper import statistic_params
        # statistic_params(generator)

        # show evaluate metrics
        evaluator.evaluate(generator, logger)

        """
        generator.eval()
        for i, data in enumerate(val_dataloader):
            with torch.no_grad():
                depth1, conf1, depth2, conf2, pred_depth_map = generator(data['rgb'].to(device),
                                                                         data['raw_depth'].to(device))

            # ----------------- save info for generating point clouds ----------------------------------
            # pred_depth_map = pred_depth_map * 5.0 + 5.0
            #
            # h, w = int(data['origin_h'][0]), int(data['origin_w'][0])
            # resized_depth = F.interpolate(pred_depth_map,
            #                               (h, w),
            #                               mode='bilinear',
            #                               align_corners=False)
            # resized_depth_np = resized_depth.detach().cpu().numpy().squeeze()
            # np.save(os.path.join(out_dir, f"{int(data['real_idx'][0]):06d}.npy"), resized_depth_np)
            # ----------------- save info for generating point clouds ----------------------------------

            # ------------------------ save images for difference branch -----------------------
            # out_dir = './2022-01-28'
            # os.makedirs(out_dir, exist_ok=True)
            # pred depth value & colored depth
            # path_save_pred = 'vis_tmp/part/pred/{:04d}.png'.format(i)
            # path_save_pred_colored = os.path.join(out_dir, f"pred_depth_colored/{i:04d}.png")
            # pred = pred_depth_map.cpu().squeeze().detach().numpy()
            # pred = pred * 5.0 + 5.0
            # pred_colored = colored_depth_map(pred)
            # pred = (pred * 25.5).astype(np.uint8)
            # save_img(pred, path_save_pred)
            # save_img(pred_colored.astype(np.uint8), path_save_pred_colored)
            #
            # path_save_rgb_branch = 'vis_tmp/part/rgb_branch/{:04d}.png'.format(i)
            # path_save_rgb_branch_colored = 'vis_tmp/part/rgb_branch_colored/{:04d}.png'.format(i)
            # rgb = depth1.cpu().squeeze().detach().numpy()
            # rgb = rgb * 5.0 + 5.0
            # rgb_colored = colored_depth_map(rgb)
            # rgb = (rgb * 25.5).astype(np.uint8)
            # save_img(rgb, path_save_rgb_branch)
            # save_img(rgb_colored.astype(np.uint8), path_save_rgb_branch_colored)
            #
            # path_save_depth_branch = 'vis_tmp/part/depth_branch/{:04d}.png'.format(i)
            # path_save_depth_branch_colored = 'vis_tmp/part/depth_branch_colored/{:04d}.png'.format(i)
            # depth = depth2.cpu().squeeze().detach().numpy()
            # depth = depth * 5.0 + 5.0
            # depth_colored = colored_depth_map(depth)
            # depth = (depth * 25.5).astype(np.uint8)
            # save_img(depth, path_save_depth_branch)
            # save_img(depth_colored.astype(np.uint8), path_save_depth_branch_colored)
            # ------------------------ save images for difference branch -----------------------

            # if i >= 50:
            #     break

            # sample = dict(rgb=data['rgb'],
            #               raw_depth=data['raw_depth'],
            #               pred_depth=pred_depth_map.cpu(),
            #               gt_depth=data['gt_depth'],
            #               confidence_map_rgb=conf1.cpu(),
            #               confidence_map_depth=conf2.cpu(),
            #               depth_map_rgb=depth1.cpu(),
            #               depth_map_depth=depth2.cpu())
            # val_dataloader.dataset.show(sample, i, args.sample_dir, save_array=True)

            print(f'\r{i * val_dataloader.batch_size}/{len(val_dataloader)}', end='')
        generator.train()
        """
        exit(0)


    # build loss
    from lib.losses.rdf_gan_loss import GANLoss, L1_loss, L2_loss
    generator_loss_coef = 1.0

    l1_loss = L1_loss
    l2_loss = L2_loss
    gan_loss = GANLoss(gan_mode=args.gan_loss_type).to(device)

    _iters = start_epoch * num_iters_per_epoch

    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = args.warm_up_steps * len(train_dataloader)

    for epoch in range(start_epoch, args.max_epoch):
        if distributed:
            train_dataloader.sampler.set_epoch(epoch)

        step_losses = dict()

        for i, data in enumerate(train_dataloader):
            generator.train()
            discriminator.train()

            if epoch < args.warm_up_steps and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer_generator.param_groups:
                    lr_warm_up = args.warm_up_lr + (param_group['initial_lr'] - args.warm_up_lr) * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

                for param_group in optimizer_disc.param_groups:
                    lr_warm_up = args.warm_up_lr + (param_group['initial_lr'] - args.warm_up_lr) * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            # fetch a batch of data
            rgb = data['rgb'].to(device)
            depth = data['raw_depth'].to(device)
            depth_gt = data['gt_depth'].to(device)
            if 'depth_masks' in data:
                # whether ignore the invalid pixel or not
                mask = data['depth_masks'].to(device)
            else:
                mask = torch.ones_like(depth).to(device)

            l1_loss_weight = mask / (mask.sum() + EPSION)

            loss_stats = dict()

            #---------------------------------- train discriminator ---------------------------------- #
            requires_grad(generator, False)
            requires_grad(discriminator, True)

            depth_map_rgb, \
            confidence_map_rgb, \
            depth_map_depth, \
            confidence_map_depth, \
            final_depth_map = generator(rgb, depth)

            real_img, fake_img = depth_gt, depth_map_rgb
            discriminator.zero_grad()
            # real
            pred_real = discriminator(real_img)
            weight = torch.ones_like(pred_real).to(device)
            disc_rgb_real_loss = gan_loss(pred_real, True, weight=weight / (weight.sum() + EPSION))
            disc_rgb_real_loss.backward()

            # fake
            # use .detach() to cut off the gradient propagation to generator,[unnecessary, already setting gardient to False]
            pred_fake = discriminator(fake_img)
            weight = torch.ones_like(pred_fake).to(device)
            disc_rgb_fake_loss = gan_loss(pred_fake, False, weight=weight / (weight.sum() + EPSION))
            disc_rgb_fake_loss.backward()

            if args.gan_loss_type == 'wgangp':
                # I didn't implement gardient-penalty in GANLoss helper class
                # DistributedDataParallel not work with torch.autograd.grad()
                eps = torch.rand(real_img.shape[0], 1, 1, 1).to(device)
                x_hat = eps * real_img.data + (1 - eps) * fake_img.data
                x_hat.requires_grad = True
                hat_predict = discriminator(x_hat)
                grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
                grad_penalty = (
                        (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
                ).mean()
                grad_penalty = 10 * grad_penalty  # the lambda coefficient equals to 10 in wgan-gp
                grad_penalty.backward()
                loss_stats.update({'grad_penalty_rgb': grad_penalty})

            loss_stats.update(dict(disc_fake_loss=disc_rgb_fake_loss,
                                   disc_real_loss=disc_rgb_real_loss,
                                   disc_loss=disc_rgb_fake_loss + disc_rgb_real_loss))

            optimizer_disc.step()

            if args.gan_loss_type == 'wgan':
                for param in discriminator.parameters():
                    param.data.clamp_(-args.wgan_clip_value, args.wgan_clip_value)


            #---------------------------------- train generator ---------------------------------- #
            if (_iters + 1) % args.n_critic == 0:
                generator.zero_grad()
                requires_grad(generator, True)
                requires_grad(discriminator, False)

                depth_map_rgb, \
                confidence_map_rgb, \
                depth_map_depth, \
                confidence_map_depth, \
                final_depth_map = generator(rgb, depth)

                pred_fake = discriminator(depth_map_rgb)
                weight = torch.ones_like(pred_fake).to(device)
                gen_fake_loss = gan_loss(pred_fake, True, weight=weight / (weight.sum() + EPSION))
                gen_gan_loss = generator_loss_coef * gen_fake_loss

                # l1 loss only
                rgb_branch_l1_loss = args.rgb_branch_l1_loss_coef * l1_loss(depth_map_rgb, depth_gt, weight=l1_loss_weight)
                depth_branch_l1_loss = args.depth_branch_l1_loss_coef * l1_loss(depth_map_depth, depth_gt, weight=l1_loss_weight)
                final_l1_loss = args.final_l1_loss_coef * l1_loss(final_depth_map, depth_gt, weight=l1_loss_weight)

                gen_total_loss = gen_gan_loss + rgb_branch_l1_loss + depth_branch_l1_loss + final_l1_loss
                # gen_total_loss = gen_gan_loss + final_l1_loss
                loss_stats.update(dict(gen_fake_loss=gen_fake_loss,
                                       rgb_branch_l1_loss=rgb_branch_l1_loss,
                                       depth_branch_l1_loss=depth_branch_l1_loss,
                                       final_l1_loss=final_l1_loss
                                       ))


                gen_total_loss.backward()
                optimizer_generator.step()

                requires_grad(generator, False)
                requires_grad(discriminator, True)

            reduced_loss = reduce_loss(loss_stats)
            for k in reduced_loss:
                if k not in step_losses:
                    step_losses[k] = MovingAverage(reduced_loss[k],
                                                   window_size=args.log_interval if "gen" not in k else args.log_interval // args.n_critic)
                else:
                    step_losses[k].push(reduced_loss[k])


            if (i + 1) % args.log_interval == 0:
                log_msg = "Train - Epoch [{}][{}/{}] ".format(epoch + 1,
                                                              i + 1,
                                                              num_iters_per_epoch)

                for name in step_losses:
                    val = step_losses[name].avg()
                    log_msg += "{}: {:.4f}| ".format(name, val)
                    logger.scalar_summary(name, val, _iters + 1)

                lr_generator = [group['lr'] for group in optimizer_generator.param_groups]
                lr_disc_rgb = [group['lr'] for group in optimizer_disc.param_groups]
                logger.scalar_summary(f'lr_generator', lr_generator[0], _iters + 1)
                logger.scalar_summary(f'lr_disc', lr_disc_rgb[0], _iters + 1)

                logger.log(log_msg)

            if (_iters + 1) % args.sample_interval == 0:
                if args.local_rank == 0:
                    generator.eval()
                    with torch.no_grad():
                        depth_map_rgb, \
                        confidence_map_rgb, \
                        depth_map_depth, \
                        confidence_map_depth, \
                        final_depth_map = generator(fixed_samples['rgb'].to(device), fixed_samples['raw_depth'].to(device))

                    sample = dict(rgb=fixed_samples['rgb'],
                                  raw_depth=fixed_samples['raw_depth'],
                                  pred_depth=final_depth_map.cpu(),
                                  gt_depth=fixed_samples['gt_depth'],
                                  confidence_map_rgb=confidence_map_rgb.cpu(),
                                  confidence_map_depth=confidence_map_depth.cpu(),
                                  depth_map_rgb=depth_map_rgb.cpu(),
                                  depth_map_depth=depth_map_depth.cpu())
                    val_dataloader.dataset.show(sample, _iters + 1, args.sample_dir, save_array=True)
                    generator.train()

            _iters += 1

        del step_losses

        if not (epoch < args.warm_up_steps and args.warm_up):
            lr_scheduler_generator.step()
            lr_scheduler_disc.step()

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.max_epoch:
            # Epoch based
            if args.local_rank == 0:
                save_checkpoint(module=dict(generator=generator,
                                            discriminator=discriminator),
                                filename=os.path.join(args.work_dir, f'epoch_{epoch + 1}.pth'),
                                optimizer=dict(generator=optimizer_generator,
                                               discriminator=optimizer_disc),
                                lr_scheduler=dict(generator=lr_scheduler_generator,
                                                  discriminator=lr_scheduler_disc),
                                meta=dict(epoch=epoch + 1))


        if (epoch + 1) >= args.start_eval_epoch and (epoch + 1) % args.val_interval == 0:
            evaluator.evaluate(generator, logger)


if __name__ == '__main__':
    main()

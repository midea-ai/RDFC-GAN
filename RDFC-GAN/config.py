
import argparse
import os

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--dataset', type=str, default='nyuv2',
                    choices=['nyuv2', 'cleargrasp', 'thuman','sunrgbd'],
                    help='dataset name')
parser.add_argument('--data_root', type=str,
                    default=None,
                    required=True,
                    help='path to dataset')
                    
parser.add_argument('--real_or_syn', type=str, default='synthetic', help='data type for cleargrasp-test-val')
parser.add_argument('--obj_type', type=str, default='known', choices=['known', 'novel'])

parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU during training')
parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

parser.add_argument('--num_classes', type=int, default=14,help='Semantic number classes ')

parser.add_argument('--resize_height', type=int, default=240,help='resize height in augmentation')
parser.add_argument('--resize_width', type=int, default=320,help='resize width in augmentation')

parser.add_argument('--out_height', type=int, default=224,help='pred depth height')
parser.add_argument('--out_width', type=int, default=304,help='output pred depth height')

# Semantic label setting
# The default values correspond to 13 types of semantic labels for the NYUV2 dataset
parser.add_argument('--label_wall', type=int, default=12)
parser.add_argument('--label_floor', type=int, default=5)
parser.add_argument('--label_ceiling', type=int, default=3)

#default='./RDFC-GAN/config/rdf_cycle_patchgan_config.yaml'
parser.add_argument('--model_cfg_path', type=str, required=True)




# training
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.002,
                    help='Initial learning rate')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 coefficient used for computing running averages of gradient and its square in adam')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 coefficient used for computing running averages of gradient and its square in adam')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs of training')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')
parser.add_argument('--scheduler', type=str, default='linear', help='lr scheduler to use')
# lambdaLR
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
# multiStepLR
parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340], nargs='+',
                    help='for step scheduler. where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                    help='for step scheduler. decay rate for learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')



parser.add_argument('--warm_up_lr', type=float, default=0.000001,
                    help='determine min lr used to warm up')
parser.add_argument('--warm_up',
                    action='store_true',
                    default=False,
                    help='whether use lr warm up')
parser.add_argument('--warm_up_steps',
                    type=int,
                    default=1,
                    help='number of epoch to do lr warm up')
parser.add_argument('--gan_loss_type', type=str, default='lsgan',
                    choices=['wgan', 'wgangp', 'lsgan', 'vanilla'],
                    help='class of gan loss')

parser.add_argument('--pool_size', type=int, default=50)
parser.add_argument('--clip_grad', default=False, action='store_true')
parser.add_argument('--max_norm', default=10, type=float)
parser.add_argument('--norm_type', default=2, type=int)

# loss weights
parser.add_argument('--lambda_A', type=float, default=100., help='loss weight of forward cycle')
parser.add_argument('--lambda_B', type=float, default=100., help='loss weight of backward cycle')
parser.add_argument('--lambda_L1', type=float, default=100.)

parser.add_argument('--lambda_l1_rgb_branch', type=float, default=100.0, help='l1 loss weight of RDF rgb branch')
parser.add_argument('--lambda_l1_depth_branch', type=float, default=100.0, help='l1 loss weight of RDF depth branch')
parser.add_argument('--lambda_l1_fusion', type=float, default=100.0, help='l1 loss weight of RDF fuison')



# io
parser.add_argument('--work_dir', default=None,required=True, help='the dir to save logs and models')
parser.add_argument('--resume_from', default=None, help='ckpt file path to resume from')

#load_from pth file
parser.add_argument('--load_from', default=None, help='ckpt file path to load from') 
parser.add_argument('--log_interval', type=int, default=30, help='log msg frequency')
parser.add_argument('--save_interval', type=int, default=10, help='ckpt saving frequency')
parser.add_argument('--sample_interval', type=int, default=1000, help='val frequency, visiualization')
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--criterion_to_get_best_ckpt', type=str, default='RMSE')
parser.add_argument('--start_eval_epoch', type=int, default=1)
parser.add_argument('--sample_dir', type=str, default='./')

# seed, rank and others
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpus', type=str, default="0", help="gpus to use")
parser.add_argument("--local_rank", type=int, default=0, help='local rank for distributed training')
parser.add_argument('--test_only', default=False, action='store_true')

parser.add_argument('--init_disc', action='store_true')
# parser.add_argument('--cal_fps', default=False, action='store_true')


args = parser.parse_args()
args.num_gpus = len(args.gpus.split(','))

if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)

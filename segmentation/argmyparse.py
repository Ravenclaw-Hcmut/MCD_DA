import argparse
# import os

from datasets_segment import get_n_class, DATASET_LIST
from enum import Enum

SHAPE_WBC_1 = [120, 120]
SHAPE_WBC_2 = [300, 300]

class DatasetSplit(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    

def fix_img_shape_args(args):
    if "src_dataset" in args.__dict__.keys() and args.src_dataset == "2d3d":
        args.train_img_shape = [1080, 1080]
        print ("args.train_img_shape was changed to %s" % args.train_img_shape)

    if "tgt_dataset" in args.__dict__.keys() and args.tgt_dataset == "test":
        args.test_img_shape = [1280, 720]
        print ("args.test_img_shape was changed to %s" % args.test_img_shape)
    
    if "src_dataset" in args.__dict__.keys() and "tgt_dataset" in args.__dict__.keys() and args.src_dataset == "wbc_1" and args.tgt_dataset == "wbc_2":
        args.train_img_shape = SHAPE_WBC_1
        print ("args.train_img_shape was changed to %s" % args.train_img_shape)
    
    if "train_img_shape" in args.__dict__.keys():
        print (f"args.train_img_shape is {args.train_img_shape}")
        
    return args

def add_additional_params_to_args(args):
    dataset = args.src_dataset if "src_dataset" in args.__dict__.keys() else args.tgt_dataset
    
    # print(dataset)
    
    args.n_class, args.label_background = get_n_class(dataset)
    # print(args.n_class)
    # args.machine = os.uname()[1]

    return args


def get_common_training_parser(parser:argparse.ArgumentParser=None):
    # ---------- How to Save ---------- #
    parser.add_argument('--savename', type=str, default="normal", help="save name(Do NOT use '-')")
    parser.add_argument('--base_outdir', type=str, default='train_output',
                        help="base output dir")
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train (default: 2)')
    parser.add_argument("--max_iter", type=int, default=5000)  # Iter per epoch

    # ---------- Define Network ---------- #
    parser.add_argument('--net', type=str, default="drn_d_38", help="network structure",
                        choices=['fcn', 'psp', 'segnet', 'fcnvgg',
                                 "drn_c_26", "drn_c_42", "drn_c_58", "drn_d_22",
                                 "drn_d_38", "drn_d_54", "drn_d_105"])
    parser.add_argument('--res', type=str, default='50', metavar="ResnetLayerNum",
                        choices=["18", "34", "50", "101", "152"], help='which resnet 18,50,101,152')
    parser.add_argument("--is_data_parallel", action="store_true",
                        help='whether you use torch.nn.DataParallel')

    # ---------- Hyperparameters ---------- #
    parser.add_argument('--opt', type=str, default="sgd", choices=['sgd', 'adam'],
                        help="network optimizer")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--adjust_lr", action="store_true",
                        help='whether you change lr')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum sgd (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=2e-5,
                        help='weight_decay (default: 2e-5)')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help="batch_size")
    parser.add_argument("--normalize_way", type=str, default="imagenet", choices=["imagenet", "None"],
                        help="normalize way")
    
    # ---------- Optional Hyperparameters ---------- #
    # parser.add_argument('--augment', action="store_true",
    #                     help='whether you use data-augmentation or not')
    parser.add_argument('--crop_size', type=int, default=-1,
                        help='crop size (default: -1): 512 is good for cityscapes')
    # 512 is derived from https://github.com/mitmul/chainer-pspnet/blob/05545d5ed254ec557697442cf98b1a2c5135216a/datasets/cityscapes/cityscapes_transformed.py#L13
    parser.add_argument('--rotate_angle', type=int, default=0,
                        help='crop size (default: -1): lower than 10 is good(?)')
    
    parser.add_argument('--loss_weights_file', type=str, default=None,
                        help='Use this when you control the loss per class')
    parser.add_argument("--add_bg_loss", action="store_true",
                        help='whether you add background loss or not')
    parser.add_argument("--fix_bn", action="store_true", type=bool, default=True,
                        help='whether you fix the paramters of batch normalization layer')
    parser.add_argument("--no_dropout", action="store_true",
                        help='whether you use dropout')
    
    # ---------- Input Information Setting ---------- #
    parser.add_argument("--input_ch", type=int, default=3,
                        choices=[1, 3, 4])
    parser.add_argument('--train_img_shape', default=(120, 120), nargs=2, metavar=("W", "H"),
                        help="W H")
    parser.add_argument("--background_id", type=int, default=255,
                        help="background id")
    
    # ---------- Whether to Resume ---------- #
    parser.add_argument("--resume", type=str, default=None, metavar="PTH.TAR",
                        help="model(pth) path")
    return parser

def get_src_only_training_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='PyTorch Segmentation Adaptation')

    parser.add_argument('src_dataset', type=str, choices=DATASET_LIST)
    parser.add_argument('--split', type=str, default='train',
                        help="which split('train' or 'trainval' or 'val' or something else) is used ")
    return get_common_training_parser(parser)


def get_da_base_training_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='PyTorch Segmentation Adaptation')

    parser.add_argument('src_dataset', type=str, choices=DATASET_LIST)
    parser.add_argument('tgt_dataset', type=str, choices=DATASET_LIST)
    parser.add_argument('--src_split', type=str, default='train',
                        help="which split('train' or 'trainval' or 'val' or something else) is used ")
    parser.add_argument('--tgt_split', type=str, default='train',
                        help="which split('train' or 'trainval' or 'val' or something else) is used ")
    parser.add_argument('--id_crossval', type=int, default=0,
                        help="cross validation split id")
    parser.add_argument('--patience', type=int, default=30,
                        help="early stopping patience")
    # alpha_iou
    parser.add_argument('--alpha_iou', type=float, default=0.5,
                        help='alpha value for iou when calculating general metric')
    return get_common_training_parser(parser)


def get_da_mcd_training_parser():
    parser = get_da_base_training_parser()
    parser.add_argument('--method', type=str, default="MCD", help="Method Name")
    parser.add_argument('--num_k', type=int, default=4,
                        help='how many steps to repeat the generator update')
    parser.add_argument("--num_multiply_d_loss", type=int, default=1)
    parser.add_argument('--d_loss', type=str, default="diff",
                        choices=['mysymkl', 'symkl', 'diff'],
                        help="choose from ['mysymkl', 'symkl', 'diff']")
    parser.add_argument('--uses_one_classifier', action="store_true",
                        help="adversarial dropout regularization")

    return parser

from __future__ import division

import os
import logging

import torch
import tqdm
from PIL import Image
from tensorboard_logger import configure, log_value
# from torch.autograd import Variable
# from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
from argmyparse import add_additional_params_to_args, fix_img_shape_args, get_da_mcd_training_parser, DatasetSplit
from datasets_segment import ConcatDataset, get_dataset, check_src_tgt_ok
from loss import CrossEntropyLoss2d, get_prob_distance_criterion
from models.model_util import get_models, get_optimizer
from transform import ReLabel, ToLabel, Scale, RandomSizedCrop, RandomHorizontalFlip, RandomRotation
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done, save_checkpoint, adjust_learning_rate, \
    get_class_weight_from_file, setup_logging, Log_CSV
from eval_segmentation import infer_image, get_metric, get_general_metric

import numpy as np

parser = get_da_mcd_training_parser()
args = parser.parse_args()
args = add_additional_params_to_args(args)

setup_logging(args)

args = fix_img_shape_args(args)

# check_src_tgt_ok(args.src_dataset, args.tgt_dataset)

weight_loss = torch.ones(args.n_class)
if not args.add_bg_loss:
    weight_loss[args.label_background] = 0  # Ignore background loss

args.start_epoch = 0
resume_flg = True if args.resume else False
start_epoch = 0
if args.resume:
    # print("=> loading checkpoint '{}'".format(args.resume))
    logging.info("=> loading checkpoint '{}'".format(args.resume))
    if not os.path.exists(args.resume):
        raise OSError("%s does not exist!" % args.resume)

    indir, infn = os.path.split(args.resume)

    old_savename = args.savename
    args.savename = infn.split("-")[0]
    # print ("savename is %s (original savename %s was overwritten)" % (args.savename, old_savename))
    logging.info("savename is %s (original savename %s was overwritten)" % (args.savename, old_savename))

    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    # ---------- Replace Args!!! ----------- #
    args = checkpoint['args']
    # -------------------------------------- #
    model_g, model_f1, model_f2 = get_models(net_name=args.net, res=args.res, input_ch=args.input_ch,
                                             n_class=args.n_class, method=args.method,
                                             is_data_parallel=args.is_data_parallel)
    optimizer_g = get_optimizer(model_g.parameters(), lr=args.lr, momentum=args.momentum, opt=args.opt,
                                weight_decay=args.weight_decay)
    optimizer_f = get_optimizer(list(model_f1.parameters()) + list(model_f2.parameters()), lr=args.lr, opt=args.opt,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_f1.load_state_dict(checkpoint['f1_state_dict'])
    if not args.uses_one_classifier:
        model_f2.load_state_dict(checkpoint['f2_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_f.load_state_dict(checkpoint['optimizer_f'])
    # print("=> loaded checkpoint '{}'".format(args.resume))
    logging.info("=> loaded checkpoint '{}'".format(args.resume))

else:
    model_g, model_f1, model_f2 = get_models(net_name=args.net, res=args.res, input_ch=args.input_ch,
                                             n_class=args.n_class,
                                             method=args.method, uses_one_classifier=args.uses_one_classifier,
                                             is_data_parallel=args.is_data_parallel)
    optimizer_g = get_optimizer(model_g.parameters(), lr=args.lr, momentum=args.momentum, opt=args.opt,
                                weight_decay=args.weight_decay)
    optimizer_f = get_optimizer(list(model_f1.parameters()) + list(model_f2.parameters()), opt=args.opt,
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
if args.uses_one_classifier:
    # print ("uses_one_classifier, f1 and f2 are same!")
    logging.info("uses_one_classifier, f1 and f2 are same!")
    model_f2 = model_f1

# mode = "%s-%s2%s-%s_%sch" % (args.src_dataset, args.src_split, args.tgt_dataset, args.tgt_split, args.input_ch)
mode = f"train-{args.src_dataset}_to_{args.tgt_dataset}-fold{args.id_crossval}-{args.input_ch}ch"
log_csv = Log_CSV(mode=mode)
logging.info("mode: %s" % mode)

if args.net in ["fcn", "psp"]:
    model_name = "%s-%s-%s-res%s" % (args.method, args.savename, args.net, args.res)
else:
    model_name = "%s-%s-%s" % (args.method, args.savename, args.net)

outdir = os.path.join(args.base_outdir, mode)

# Create Model Dir
pth_dir = os.path.join(outdir, "pth")
mkdir_if_not_exist(pth_dir)

# Create Model Dir and  Set TF-Logger
tflog_dir = os.path.join(outdir, "tflog", model_name)
mkdir_if_not_exist(tflog_dir)
configure(tflog_dir, flush_secs=5)

# Save param dic
if resume_flg:
    json_fn = os.path.join(args.outdir, "param-%s_resume.json" % model_name)
else:
    json_fn = os.path.join(outdir, "param-%s.json" % model_name)
check_if_done(json_fn)
save_dic_to_json(args.__dict__, json_fn)

train_img_shape = tuple([int(x) for x in args.train_img_shape])
img_transform_list = [
    Scale(train_img_shape, Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225])
]

try:
    if args.augment:
        aug_list = [
            RandomRotation(),
            # RandomVerticalFlip(), # non-realistic
            RandomHorizontalFlip(),
            RandomSizedCrop()
        ]
        img_transform_list = aug_list + img_transform_list
except AttributeError:
    # print("augment is not defined. Do nothing.")
    logging.info("augment is not defined. Do nothing.")

img_transform = Compose(img_transform_list)

label_transform = Compose([
    Scale(train_img_shape, Image.NEAREST),
    ToLabel(),
    ReLabel(0, args.label_background),  # Last Class is "Void" or "Background" class
])

src_dataset = get_dataset(dataset_name=args.src_dataset, split=DatasetSplit.TRAIN.value, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args.input_ch, id_crossval=args.id_crossval)

tgt_dataset = get_dataset(dataset_name=args.tgt_dataset, split=DatasetSplit.TRAIN.value, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args.input_ch, id_crossval=args.id_crossval)

src_val_dataset = get_dataset(dataset_name=args.src_dataset, split=DatasetSplit.VAL.value, img_transform=img_transform,
                              label_transform=label_transform, test=True, input_ch=args.input_ch, id_crossval=args.id_crossval)

tgt_val_dataset = get_dataset(dataset_name=args.tgt_dataset, split=DatasetSplit.VAL.value, img_transform=img_transform,
                               label_transform=label_transform, test=True, input_ch=args.input_ch, id_crossval=args.id_crossval)

logging.info(src_dataset.files)
logging.info(tgt_dataset.files)
logging.info(tgt_val_dataset.files)

train_loader = torch.utils.data.DataLoader(
    ConcatDataset(
        src_dataset,
        tgt_dataset
    ),
    batch_size=args.batch_size, shuffle=True,
    pin_memory=True)

src_val_loader = torch.utils.data.DataLoader(
    src_val_dataset,
    batch_size=args.batch_size, shuffle=False,
    pin_memory=True)

tgt_val_loader = torch.utils.data.DataLoader(
    tgt_val_dataset,
    batch_size=args.batch_size, shuffle=False,
    pin_memory=True)

def get_unique_labels(dataset):
    """Extract unique labels from dataset"""
    all_labels = []
    logging.info(f"len(dataset): {len(dataset)}")
    for source, target in dataset:
        unique_values = torch.unique(source[1])
        all_labels.append(unique_values.tolist())
    return all_labels
    # return sorted(list(set(all_labels)))

train_labels = get_unique_labels(train_loader.dataset)
logging.info(f"Unique labels in dataset: {train_labels}")

weight_loss = get_class_weight_from_file(n_class=args.n_class, weight_filename=args.loss_weights_file,
                                    add_bg_loss=args.add_bg_loss)

if torch.cuda.is_available():
    model_g.cuda()
    model_f1.cuda()
    model_f2.cuda()
    weight_loss = weight_loss.cuda()

criterion = CrossEntropyLoss2d(weight_loss)
criterion_d = get_prob_distance_criterion(args.d_loss)

patience = args.patience
best_metric = 0
best_metric_epoch = 0
alpha_iou = args.alpha_iou

model_g.train()
model_f1.train()
model_f2.train()

# print('Epoch\tLoss_C\tLoss_D\tVal_IoU\tVal_Dice\tVal_IoU_Dice\tBest_Val_IoU_Dice\tBest_Epoch\tLR')

for epoch in range(start_epoch, args.epochs):
    d_loss_per_epoch = 0
    c_loss_per_epoch = 0

    for ind, (source, target) in tqdm.tqdm(enumerate(train_loader)):        
        src_imgs, src_lbls = source[0], source[1]
        tgt_imgs = target[0]
                
        if torch.cuda.is_available():
            src_imgs, src_lbls, tgt_imgs = src_imgs.cuda(), src_lbls.cuda(), tgt_imgs.cuda()

        # update generator and classifiers by source samples
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        loss = 0
        loss_weight = [1.0, 1.0]
        outputs = model_g(src_imgs)

        outputs1 = model_f1(outputs)
        outputs2 = model_f2(outputs)
                
        loss += criterion(outputs1, src_lbls)
        loss += criterion(outputs2, src_lbls)
        loss.backward()
        c_loss = loss.data
        c_loss_per_epoch += c_loss

        optimizer_g.step()
        optimizer_f.step()
        # update for classifiers
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        outputs = model_g(src_imgs)
        outputs1 = model_f1(outputs)
        outputs2 = model_f2(outputs)
        loss = 0
        loss += criterion(outputs1, src_lbls)
        loss += criterion(outputs2, src_lbls)
        outputs = model_g(tgt_imgs)
        outputs1 = model_f1(outputs)
        outputs2 = model_f2(outputs)
        loss -= criterion_d(outputs1, outputs2)
        loss.backward()
        optimizer_f.step()
        
        d_loss = 0.0
        # update generator by discrepancy
        for i in range(args.num_k):
            optimizer_g.zero_grad()
            loss = 0
            outputs = model_g(tgt_imgs)
            outputs1 = model_f1(outputs)
            outputs2 = model_f2(outputs)
            loss += criterion_d(outputs1, outputs2) * args.num_multiply_d_loss
            loss.backward()
            optimizer_g.step()

        d_loss += loss.data / args.num_k
        d_loss_per_epoch += d_loss
        if ind % 100 == 0:
            print("iter [%d] DLoss: %.6f CLoss: %.4f" % (ind, d_loss, c_loss))
        # if ind > args.max_iter:
        #     break

    with torch.no_grad():
        _preds_val_src, _lbls_val_src = infer_image(model_g, model_f1, model_f2, src_val_loader)
        _preds_val_tgt, _lbls_val_tgt = infer_image(model_g, model_f1, model_f2, tgt_val_loader)
        ious_val_src, dices_val_src = get_metric(_preds_val_src, _lbls_val_src, args.n_class)
        ious_val_tgt, dices_val_tgt = get_metric(_preds_val_tgt, _lbls_val_tgt, args.n_class)
        
        mean_iou_src, mean_dice_src = np.mean(ious_val_src), np.mean(dices_val_src)
        mean_iou_tgt, mean_dice_tgt = np.mean(ious_val_tgt), np.mean(dices_val_tgt)
        
        metric_general_tgt = get_general_metric(ious_val_tgt, dices_val_tgt, alpha_iou)
        metric_general_src = get_general_metric(ious_val_src, dices_val_src, alpha_iou)
        
        if metric_general_tgt >= best_metric:
            best_metric, best_metric_epoch = metric_general_tgt, epoch
        
        if epoch - best_metric_epoch > patience:
            # print(f"Early Stopping at Epoch {epoch}, Best Metric: {best_metric}, Best Metric Epoch: {best_metric_epoch}")
            logging.info(f"Early Stopping at Epoch {epoch}, Best Metric: {best_metric}, Best Metric Epoch: {best_metric_epoch}")
            break
        
    log_csv.update(epoch, c_loss_per_epoch, d_loss_per_epoch, ious_val_src, dices_val_src, ious_val_tgt, dices_val_tgt, 
                   mean_iou_src, mean_dice_src, metric_general_src, mean_iou_tgt, mean_dice_tgt, metric_general_tgt, best_metric, best_metric_epoch)
    
    # print(f'{epoch}\t{c_loss_per_epoch}\t{d_loss_per_epoch}\t{ious_val_tgt}\t{dices_val_tgt}\t\t{metric_general_tgt}\t\t{best_metric}\t\t\t{best_metric_epoch}\t{args.lr}')
    logging.info(f"Epoch {epoch} Loss_C: {c_loss_per_epoch} Loss_D: {d_loss_per_epoch} Val_IoU_src: {ious_val_src} Val_Dice_src: {dices_val_src} Val_IoU_Dice_src {metric_general_src} Val_IoU_tgt: {ious_val_tgt} Val_Dice_tgt: {dices_val_tgt} Val_IoU_Dice_tgt: {metric_general_tgt} Best_Val_IoU_Dice: {best_metric} Best_Epoch: {best_metric_epoch} Lr: {args.lr}")
    # print("Epoch [%d] DLoss: %.4f CLoss: %.4f" % (epoch, d_loss_per_epoch, c_loss_per_epoch))

    log_value('c_loss', c_loss_per_epoch, epoch)
    log_value('d_loss', d_loss_per_epoch, epoch)
    log_value('lr', args.lr, epoch)

    if 'adjust_lr' in args and args.adjust_lr:
        args.lr = adjust_learning_rate(optimizer_g, args.lr, args.weight_decay, epoch, args.epochs)
        args.lr = adjust_learning_rate(optimizer_f, args.lr, args.weight_decay, epoch, args.epochs)

    checkpoint_fn = os.path.join(pth_dir, "%s-%s.pth.tar" % (model_name, epoch + 1))
    args.start_epoch = epoch + 1
    save_dic = {
        'epoch': epoch + 1,
        'args': args,
        'g_state_dict': model_g.state_dict(),
        'f1_state_dict': model_f1.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_f': optimizer_f.state_dict(),
    }
    if not args.uses_one_classifier:
        save_dic['f2_state_dict'] = model_f2.state_dict()

    save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)

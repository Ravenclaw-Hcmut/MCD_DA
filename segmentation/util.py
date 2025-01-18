import json
import os
import shutil

import torch
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

LOG_DIR = 'logs'

def set_debugger_org():
    if not sys.excepthook == sys.__excepthook__:
        from IPython.core import ultratb
        sys.excepthook = ultratb.FormattedTB(call_pdb=True)


def set_debugger_org_frc():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)


def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def mkdir_if_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def yes_no_input():
    while True:
        choice = raw_input("Please respond with 'yes' or 'no' [y/N]: ").lower()
        if choice in ['y', 'ye', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False


def check_if_done(filename):
    if os.path.exists(filename):
        print ("%s already exists. Is it O.K. to overwrite it and start this program?" % filename)
        # if not yes_no_input():
        #     raise Exception("Please restart training after you set args.savename differently!")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_dic_to_json(dic, fn, verbose=True):
    with open(fn, "w") as f:
        json_str = json.dumps(dic, sort_keys=True, indent=4)
        if verbose:
            print (json_str)
        f.write(json_str)
    print ("param file '%s' was saved!" % fn)


def emphasize_str(string):
    print ('#' * 100)
    print (string)
    print ('#' * 100)


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs, decay_epoch=15):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init
    if epoch == decay_epoch:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_class_weight_from_file(n_class, weight_filename=None, add_bg_loss=False):
    weight = torch.ones(n_class)
    if weight_filename:
        import pandas as pd

        loss_df = pd.read_csv(weight_filename)
        loss_df.sort_values("class_id", inplace=True)
        weight *= torch.FloatTensor(loss_df.weight.values)

    if not add_bg_loss:
        weight[n_class - 1] = 0  # Ignore background loss
    return weight


def setup_logging(args):
    """Configure logging to both file and console."""
    # Create logs directory if it doesn't exist
    log_dir = LOG_DIR
    mkdir_if_not_exist(log_dir)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Setup file handler
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info('Started training with configuration:')
    logging.info(str(args))
    
    
class Log_CSV:
    def __init__(self, mode = 'train_unknow', filename=None):
        self.filename = filename if filename else f'log_{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self.filename = os.path.join(LOG_DIR, self.filename)
        self.df = pd.DataFrame()
        self.set_class = ['background', 'boundary', 'wbc']
        name_IoU_src = [f'IoU_{c}_src' for c in self.set_class]
        name_Dice_src = [f'Dice_{c}_src' for c in self.set_class]
        name_IoU_tgt = [f'IoU_{c}_tgt' for c in self.set_class]
        name_Dice_tgt = [f'Dice_{c}_tgt' for c in self.set_class]
        self.columns = ['epoch', 'GPU_mem', 'c_loss', 'd_loss'] + name_IoU_src + name_Dice_src + name_IoU_tgt + name_Dice_tgt + ['IoU_src', 'Dice_src', 'val_Iou_Dice_src', 'IoU_tgt', 'Dice_tgt', 'val_Iou_Dice_tgt', 'best_val_Iou_Dice_tgt', 'best_epoch']
        self.df = pd.DataFrame(columns=self.columns)

    def append(self, epoch, c_loss, d_loss, arr_IoU_src, arr_Dice_src, arr_IoU_tgt, arr_Dice_tgt, 
               IoU_src, Dice_src, val_Iou_Dice_src, IoU_tgt, Dice_tgt, val_Iou_Dice_tgt, best_val_Iou_Dice, best_epoch):
        mem_GPU = 0
        if torch.cuda.is_available():
            device = torch.device("cuda")
            mem_GPU = torch.cuda.memory_allocated(device) / 1024 ** 2
        new_row = col_loss | col_IoU_src | col_Dice_src | col_IoU_tgt | col_Dice_tgt | col_IoU_Dice_src | col_IoU_Dice_tgt
        
        arr_IoU_src = arr_IoU_src.tolist()
        arr_Dice_src = arr_Dice_src.tolist()
        arr_IoU_tgt = arr_IoU_tgt.tolist()
        arr_Dice_tgt = arr_Dice_tgt.tolist()
        col_loss = {'epoch': epoch, 'GPU_mem': mem_GPU, 'c_loss': c_loss, 'd_loss': d_loss}
        col_IoU_src = {f'IoU_{c}_src': iou for c, iou in zip(self.set_class, arr_IoU_src)}
        col_Dice_src = {f'Dice_{c}_src': dice for c, dice in zip(self.set_class, arr_Dice_src)}
        col_IoU_tgt = {f'IoU_{c}_tgt': iou for c, iou in zip(self.set_class, arr_IoU_tgt)}
        col_Dice_tgt = {f'Dice_{c}_tgt': dice for c, dice in zip(self.set_class, arr_Dice_tgt)}
        col_IoU_Dice_src = {'IoU_src': IoU_src, 'Dice_src': Dice_src, 'val_Iou_Dice_src': val_Iou_Dice_src}
        col_IoU_Dice_tgt = {'IoU_tgt': IoU_tgt, 'Dice_tgt': Dice_tgt, 'val_Iou_Dice_tgt': val_Iou_Dice_tgt, 'best_val_Iou_Dice_tgt': best_val_Iou_Dice, 'best_epoch': best_epoch}

        new_row_df = pd.DataFrame([new_row])  # Convert the new row to a DataFrame
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)  # Add the new row to the DataFrame and reset the index
        
    def save(self):
        self.df.to_csv(self.filename, index=False)
    
    def update (self, epoch, c_loss, d_loss, arr_IoU_src, arr_Dice_src, arr_IoU_tgt, arr_Dice_tgt, 
               IoU_src, Dice_src, val_Iou_Dice_src, IoU_tgt, Dice_tgt, val_Iou_Dice_tgt, best_val_Iou_Dice, best_epoch):
        self.append(epoch, c_loss, d_loss, arr_IoU_src, arr_Dice_src, arr_IoU_tgt, arr_Dice_tgt, 
                    IoU_src, Dice_src, val_Iou_Dice_src, IoU_tgt, Dice_tgt, val_Iou_Dice_tgt, best_val_Iou_Dice, best_epoch)
        self.save()
    
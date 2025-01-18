import numpy as np
import torch
from tqdm import tqdm


def infer_image(model_g, model_f1, model_f2, data_loader):
    """
    Perform inference on a set of images using the provided models.
    Args:
        model_g (torch.nn.Module): The feature extraction model.
        model_f1 (torch.nn.Module): The first classification model.
        model_f2 (torch.nn.Module): The second classification model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the target dataset.
    Returns:
        list: A list of predicted labels for each image in the data_loader.
    """
    model_g.eval()
    model_f1.eval()
    model_f2.eval()

    results = []
    lbls = []
    for batch in enumerate(data_loader):
        index, (imgs, lbl_batch, paths) = batch
        if torch.cuda.is_available():
            imgs = imgs.cuda()

        with torch.no_grad():
            features = model_g(imgs)
            outputs1 = model_f1(features)
            outputs2 = model_f2(features)

        # outputs = (outputs1 + outputs2) / 2.0
        outputs = torch.max(outputs1, outputs2)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        lbl_batch = lbl_batch.numpy()

        lbls.append(lbl_batch)
        for pred in preds:
            results.append(pred)

    return results, lbls


def get_metric(preds:list[np.ndarray], targets:list[np.ndarray], n_class:int):
    """
    Compute Intersection over Union (IoU) and Dice coefficient for segmentation.
    Args:
        preds (list): List of predicted image.
        targets (list): List of ground truth image.
        n_class (int): Number of classes.
    
    """
    
    iou_list = []
    dice_list = []

    for pred, target in zip(preds, targets):
        iou_per_class = []
        dice_per_class = []
        for cls in range(n_class):
            pred_inds = pred == cls
            target_inds = target == cls

            intersection = np.logical_and(pred_inds, target_inds).sum()
            union = np.logical_or(pred_inds, target_inds).sum()
            dice = (2 * intersection) / (pred_inds.sum() + target_inds.sum())

            if union == 0:
                iou = float('nan')
            else:
                iou = intersection / union

            iou_per_class.append(iou)
            dice_per_class.append(dice)

        iou_list.append(iou_per_class)
        dice_list.append(dice_per_class)

    mean_iou = np.nanmean(iou_list, axis=0)
    mean_dice = np.nanmean(dice_list, axis=0)

    return mean_iou, mean_dice

def get_general_metric(mean_iou_perclass, mean_dice_perclass, alpha_iou):
    """
    Compute the general metric using the mean IoU and Dice coefficient.
    Args:
        mean_iou_perclass (np.ndarray): Mean IoU for each class.
        mean_dice_perclass (np.ndarray): Mean Dice coefficient for each class.
        alpha_iou (float): Alpha value for IoU.
    Returns:
        float: The general metric.
    """
    
    general_metric = (1 - alpha_iou) * np.mean(mean_dice_perclass) + alpha_iou * np.mean(mean_iou_perclass)
    return general_metric
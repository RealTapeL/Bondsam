import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def calculate_metric(results, obj):
    """
    Calculate evaluation metrics for a specific object class
    """
    gt_list = results['imgs_gts']
    pr_list = results['anomaly_scores']
    gt_mask_list = results['imgs_masks']
    pr_mask_list = results['anomaly_maps']
    
    obj_gt_list = []
    obj_pr_list = []
    obj_gt_mask_list = []
    obj_pr_mask_list = []
    
    for i, cls_name in enumerate(results['cls_names']):
        if cls_name == obj:
            obj_gt_list.append(gt_list[i])
            obj_pr_list.append(pr_list[i])
            obj_gt_mask_list.append(gt_mask_list[i])
            obj_pr_mask_list.append(pr_mask_list[i])
    
    # Calculate image-level metrics
    obj_gt_list = np.array(obj_gt_list)
    obj_pr_list = np.array(obj_pr_list)
    
    if len(obj_gt_list) > 0 and len(np.unique(obj_gt_list)) > 1:
        auroc_img = roc_auc_score(obj_gt_list, obj_pr_list)
        ap_img = average_precision_score(obj_gt_list, obj_pr_list)
        f1_img = f1_score(obj_gt_list, obj_pr_list > 0.5)
    else:
        auroc_img = 0.0
        ap_img = 0.0
        f1_img = 0.0
    
    # Calculate pixel-level metrics
    obj_gt_mask_list = np.array(obj_gt_mask_list)
    obj_pr_mask_list = np.array(obj_pr_mask_list)
    
    if len(obj_gt_mask_list) > 0:
        obj_gt_mask_list = obj_gt_mask_list.flatten()
        obj_pr_mask_list = obj_pr_mask_list.flatten()
        
        if len(np.unique(obj_gt_mask_list)) > 1:
            auroc_pixel = roc_auc_score(obj_gt_mask_list, obj_pr_mask_list)
            ap_pixel = average_precision_score(obj_gt_mask_list, obj_pr_mask_list)
            f1_pixel = f1_score(obj_gt_mask_list, obj_pr_mask_list > 0.5)
        else:
            auroc_pixel = 0.0
            ap_pixel = 0.0
            f1_pixel = 0.0
    else:
        auroc_pixel = 0.0
        ap_pixel = 0.0
        f1_pixel = 0.0
    
    return {
        'auroc_im': auroc_img * 100,
        'f1_im': f1_img * 100,
        'ap_im': ap_img * 100,
        'auroc_px': auroc_pixel * 100,
        'f1_px': f1_pixel * 100,
        'ap_px': ap_pixel * 100
    }


def calculate_average_metric(metric_dict):
    """
    Calculate average metrics across all object classes
    """
    avg_metric = {
        'auroc_im': 0.0,
        'f1_im': 0.0,
        'ap_im': 0.0,
        'auroc_px': 0.0,
        'f1_px': 0.0,
        'ap_px': 0.0
    }
    
    count = 0
    for key, metrics in metric_dict.items():
        if key != 'Average':
            count += 1
            for metric_key in avg_metric:
                avg_metric[metric_key] += metrics[metric_key]
    
    if count > 0:
        for metric_key in avg_metric:
            avg_metric[metric_key] /= count
    
    return avg_metric
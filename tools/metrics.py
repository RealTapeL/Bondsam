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
    
    if len(obj_gt_list) == 0:
        return {
            'auroc_im': 0.0,
            'f1_im': 0.0,
            'ap_im': 0.0,
            'auroc_px': 0.0,
            'f1_px': 0.0,
            'ap_px': 0.0
        }
    
    # 检查标签是否包含至少两个类别
    unique_labels = np.unique(obj_gt_list)
    if len(obj_gt_list) > 0 and len(unique_labels) > 1:
        try:
            auroc_img = roc_auc_score(obj_gt_list, obj_pr_list)
            ap_img = average_precision_score(obj_gt_list, obj_pr_list)
            # 使用最优阈值计算F1分数
            thresholds = np.linspace(0, 1, 100)
            f1_scores = [f1_score(obj_gt_list, obj_pr_list > t) for t in thresholds]
            f1_img = max(f1_scores) if f1_scores else 0.0
        except Exception as e:
            print(f"Error calculating image-level metrics for {obj}: {e}")
            auroc_img = 0.0
            ap_img = 0.0
            f1_img = 0.0
    else:
        # 当只有一个类别时，使用简单的准确率作为替代
        if len(obj_gt_list) > 0:
            accuracy = np.mean((obj_pr_list > 0.5) == obj_gt_list)
            auroc_img = accuracy
            ap_img = accuracy
            f1_img = accuracy
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
        
        unique_mask_labels = np.unique(obj_gt_mask_list)
        if len(unique_mask_labels) > 1:
            try:
                auroc_pixel = roc_auc_score(obj_gt_mask_list, obj_pr_mask_list)
                ap_pixel = average_precision_score(obj_gt_mask_list, obj_pr_mask_list)
                # 使用最优阈值计算F1分数
                thresholds = np.linspace(obj_pr_mask_list.min(), obj_pr_mask_list.max(), 100)
                f1_scores = [f1_score(obj_gt_mask_list, obj_pr_mask_list > t) for t in thresholds]
                f1_pixel = max(f1_scores) if f1_scores else 0.0
            except Exception as e:
                print(f"Error calculating pixel-level metrics for {obj}: {e}")
                auroc_pixel = 0.0
                ap_pixel = 0.0
                f1_pixel = 0.0
        else:
            # 当只有一个类别时，使用简单的准确率作为替代
            accuracy = np.mean((obj_pr_mask_list > 0.5) == obj_gt_mask_list)
            auroc_pixel = accuracy
            ap_pixel = accuracy
            f1_pixel = accuracy
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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import os
import torch
from scipy.ndimage import gaussian_filter
import cv2
import pandas as pd
import numpy as np
from PIL import Image

from tools import write2csv, setup_seed, Logger
from tools.memorybank import MemoryBank
from dataset import get_data, dataset_dict
from method.trainer import BondSAM_Trainer

setup_seed(111)

def normalize(pred, max_value=None, min_value=None):
    """
    Normalize the anomaly map to [0, 1] range
    """
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    else:
        return (pred - min_value) / (max_value - min_value + 1e-8)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    """
    Apply anomaly map as overlay on the original image
    """
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def create_binary_mask(scoremap, threshold=0.5):
    """
    Create binary mask from anomaly scoremap
    """
    binary_mask = (scoremap > threshold).astype(np.uint8) * 255
    return binary_mask

def overlay_mask_on_image(image, mask, alpha=0.6):
    """
    Overlay binary mask on image with deep blue color
    """
    # Create a deep blue mask
    blue_mask = np.zeros_like(image)
    blue_mask[:, :, 0] = 139  # B
    blue_mask[:, :, 1] = 0    # G
    blue_mask[:, :, 2] = 0    # R
    
    # Apply mask only where anomaly is detected
    mask_3d = np.stack([mask/255.0]*3, axis=-1)
    overlay = image * (1 - alpha * mask_3d) + blue_mask * (alpha * mask_3d)
    
    return overlay.astype(np.uint8)

def add_anomaly_info(image, anomaly_score):
    """
    Add anomaly score text to the image with a nice color
    """
    vis_with_text = image.copy()
    
    # Use a nice gold color for the text
    text_color = (0, 255, 0)  # BGR format (Gold)
    
    # Add score text
    score_text = f"Anomaly Score: {anomaly_score:.3f}"
    
    # Add text with a background for better visibility
    cv2.putText(vis_with_text, score_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
    
    return vis_with_text

def draw_contours_on_image(image, mask, contour_color=(0, 255, 255)):
    """
    Draw contours of anomaly regions on the image
    """
    vis_with_contours = image.copy()
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    cv2.drawContours(vis_with_contours, contours, -1, contour_color, 2)
    
    return vis_with_contours

def test_bondsam(args):
    assert os.path.isfile(args.ckt_path), f"Please check the path of pre-trained model, {args.ckt_path} is not valid."
    
    batch_size = args.batch_size
    image_size = args.image_size
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    save_fig = args.save_fig

    logger = Logger('bondsam_test_log.txt')
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')
    config_path = os.path.join('./model_configs', f'{args.model}.json')

    with open(config_path, 'r') as f:
        model_configs = json.load(f)

    if 'vision_cfg' in model_configs:
        model_configs['vision_cfg']['image_size'] = image_size

    n_layers = model_configs['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]

    model = BondSAM_Trainer(
        backbone=args.model,
        feat_list=features_list,
        input_dim=model_configs['vision_cfg']['width'],
        output_dim=model_configs['embed_dim'],
        learning_rate=0.,
        device=device,
        image_size=image_size,
        prompting_depth=args.prompting_depth,
        prompting_length=args.prompting_length,
        prompting_branch=args.prompting_branch,
        prompting_type=args.prompting_type,
        use_hsf=args.use_hsf,
        k_clusters=args.k_clusters,
        use_fastsam=True,
        use_memory_bank=args.use_memory_bank,
        k_shot=args.k_shot,
        memory_cluster_method=args.memory_cluster_method,
        memory_cluster_size=args.memory_cluster_size
    ).to(device)

    # 加载模型权重
    model.load(args.ckt_path)

    # 如果使用Memory Bank且是少样本模式，仅针对指定类别构建Memory Bank
    if args.use_memory_bank and args.mode == 'few_shot' and args.testing_model == 'image':
        assert args.testing_data in dataset_dict.keys(), f"You entered {args.testing_data}, but we only support " \
                                                         f"{dataset_dict.keys()}"

        # 获取训练数据用于构建Memory Bank，仅针对指定类别
        train_data_cls_names, train_data, train_data_root = get_data(
            dataset_type_list=args.testing_data,
            transform=model.preprocess,
            target_transform=model.transform,
            training=True,
            use_fastsam=True
        )

        # 只选择与测试图像类别相同的训练数据
        if args.class_name in train_data_cls_names:
            # 过滤数据集，只保留指定类别的数据
            filtered_indices = [i for i, item in enumerate(train_data) if item['cls_name'] == args.class_name]
            if filtered_indices:
                # 从训练数据中随机选择k_shot个样本
                import random
                if len(filtered_indices) > args.k_shot:
                    filtered_indices = random.sample(filtered_indices, args.k_shot)
                
                # 创建只包含指定类别的子数据集
                class_specific_data = torch.utils.data.Subset(train_data, filtered_indices)
                train_dataloader = torch.utils.data.DataLoader(class_specific_data, batch_size=1, shuffle=False)
                
                # 构建Memory Bank
                if hasattr(model, 'memory_bank') and model.memory_bank:
                    logger.info(f"Building Memory Bank for few-shot anomaly detection for class: {args.class_name}")
                    # 创建只包含当前类别的列表
                    single_class_list = [args.class_name]
                    memory_features = model.memory_bank.build_memory_bank(
                        model, train_dataloader, single_class_list, k_shot=min(args.k_shot, len(filtered_indices))
                    )
                    logger.info(f"Memory Bank built with {len(memory_features)} classes")
            else:
                logger.info(f"No training data found for class: {args.class_name}")
        else:
            logger.info(f"Class {args.class_name} not found in training data")

    if args.testing_model == 'dataset':
        assert args.testing_data in dataset_dict.keys(), f"You entered {args.testing_data}, but we only support " \
                                                         f"{dataset_dict.keys()}"

        save_root = args.save_path
        csv_root = os.path.join(save_root, 'csvs')
        image_root = os.path.join(save_root, 'images')
        csv_path = os.path.join(csv_root, f'{args.testing_data}_bondsam.csv')
        image_dir = os.path.join(image_root, f'{args.testing_data}_bondsam')
        os.makedirs(image_dir, exist_ok=True)

        test_data_cls_names, test_data, test_data_root = get_data(
            dataset_type_list=args.testing_data,
            transform=model.preprocess,
            target_transform=model.transform,
            training=False,
            use_fastsam=True
        )

        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        save_fig_flag = save_fig

        metric_dict = model.evaluation(
            test_dataloader,
            test_data_cls_names,
            save_fig_flag,
            image_dir,
        )

        for tag, data in metric_dict.items():
            logger.info(
                '{:>15} \t\tI-Auroc:{:.2f} \tI-F1:{:.2f} \tI-AP:{:.2f} \tP-Auroc:{:.2f} \tP-F1:{:.2f} \tP-AP:{:.2f}'.
                    format(tag,
                           data['auroc_im'],
                           data['f1_im'],
                           data['ap_im'],
                           data['auroc_px'],
                           data['f1_px'],
                           data['ap_px'])
            )

        for k in metric_dict.keys():
            write2csv(metric_dict[k], test_data_cls_names, k, csv_path)

    elif args.testing_model == 'image':
        assert os.path.isfile(args.image_path), f"Please verify the input image path: {args.image_path}"
        ori_image = cv2.resize(cv2.imread(args.image_path), (args.image_size, args.image_size))
        pil_img = Image.open(args.image_path).convert('RGB')

        img_input = model.preprocess(pil_img)
        if not isinstance(img_input, torch.Tensor):
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            img_input = to_tensor(img_input)
        
        img_input = img_input.unsqueeze(0)
        img_input = img_input.to(model.device)

        # 处理FastSAM掩码
        if hasattr(model, 'fastsam_processor') and model.fastsam_processor is not None:
            fastsam_mask = model.fastsam_processor.process_image(args.image_path)
            fastsam_mask = torch.from_numpy(fastsam_mask).float().unsqueeze(0)
            fastsam_mask = fastsam_mask.to(model.device)
            logger.info(f"FastSAM mask stats - Min: {fastsam_mask.min()}, Max: {fastsam_mask.max()}, Mean: {fastsam_mask.mean()}")
        else:
            fastsam_mask = torch.ones((1, args.image_size, args.image_size), dtype=torch.float32).to(model.device) * 0.5
            logger.info("Using default FastSAM mask (all 0.5)")

        with torch.no_grad():
            anomaly_map, anomaly_score = model.clip_model(img_input, [args.class_name], aggregation=True, fastsam_mask=fastsam_mask)
            
            # 记录异常图的一些统计信息
            logger.info(f"Anomaly map stats - Min: {anomaly_map.min()}, Max: {anomaly_map.max()}, Mean: {anomaly_map.mean()}")

        anomaly_map = anomaly_map[0, :, :]
        anomaly_score = anomaly_score[0]
        anomaly_map = anomaly_map.cpu().numpy()
        anomaly_score = anomaly_score.cpu().numpy()

        # Apply Gaussian filter to smooth the anomaly map
        anomaly_map_smoothed = gaussian_filter(anomaly_map, sigma=4)
        
        # Normalize the anomaly map
        anomaly_map_normalized = normalize(anomaly_map_smoothed)
        
        # 记录归一化后的异常图统计信息
        logger.info(f"Normalized anomaly map stats - Min: {anomaly_map_normalized.min()}, Max: {anomaly_map_normalized.max()}, Mean: {anomaly_map_normalized.mean()}")
        
        # Create binary mask for highlighting anomalies using configurable threshold
        binary_mask = create_binary_mask(anomaly_map_normalized, threshold=args.anomaly_threshold)
        
        # 记录二值化掩码的统计信息
        logger.info(f"Binary mask stats - Unique values: {np.unique(binary_mask)}, Non-zero ratio: {np.count_nonzero(binary_mask) / binary_mask.size}")
        
        # Overlay deep blue mask on the original image
        masked_image = overlay_mask_on_image(ori_image, binary_mask, alpha=0.6)
        
        # Add only anomaly score to the image (without threshold)
        final_image = add_anomaly_info(masked_image, anomaly_score)
        
        # Draw contours around anomaly regions
        final_image_with_contours = draw_contours_on_image(final_image, binary_mask)
        
        # Save the final visualization
                # Save the final visualization with high quality PNG
        save_path = os.path.join(args.save_path, args.save_name)
        if not args.save_name.endswith('.png'):
            save_path = save_path.replace('.jpg', '.png').replace('.jpeg', '.png')
        cv2.imwrite(save_path, final_image_with_contours)
        
        # Save the binary mask with high quality PNG
        mask_save_path = os.path.join(args.save_path, f"mask_{args.save_name}")
        if not args.save_name.endswith('.png'):
            mask_save_path = mask_save_path.replace('.jpg', '.png').replace('.jpeg', '.png')
        cv2.imwrite(mask_save_path, binary_mask)
        
        # Save the heatmap with high quality PNG
        heatmap = (anomaly_map_normalized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_save_path = os.path.join(args.save_path, f"heatmap_{args.save_name}")
        if not args.save_name.endswith('.png'):
            heatmap_save_path = heatmap_save_path.replace('.jpg', '.png').replace('.jpeg', '.png')
        cv2.imwrite(heatmap_save_path, heatmap)
        
        print(f"Anomaly score: {anomaly_score:.3f}")
        print(f"Anomaly threshold: {args.anomaly_threshold:.3f}")
        print(f"Visualization saved to: {save_path}")

def write2csv(results:dict,total_classes,cur_class,csv_path):
    keys=list(results.keys())

    if not os.path.exists(csv_path):
        df_all=None
        for class_name in total_classes:
            r=dict()
            for k in keys:
                r[k]=0.00
            df_temp=pd.DataFrame(r,index=[f'{class_name}'])

            if df_all is None:
                def_all=df_temp
            else:
                df_all=pd.concat([df_all,df_temp],axis=0)

        df_all.to_csv(csv_path,header=True,float_format='%.2f')
    df=pd.read_csv(csv_path,index_col=0)

    for k in keys:
        df.loc[f'{cur_class}',k]=results[k]

    df.to_csv(csv_path,header=True,float_format='%.2f')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing BondSAM')
    parser.add_argument("--ckt_path", type=str, required=True, help="path to the pre-trained model")
    parser.add_argument("--testing_model", type=str, default="dataset", choices=["dataset", "image"])
    parser.add_argument("--testing_data", type=str, default="mvtec")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--save_fig", action="store_true")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--model", type=str, default="ViT-B-16")
    parser.add_argument("--prompting_depth", type=int, default=3)
    parser.add_argument("--prompting_length", type=int, default=2)
    parser.add_argument("--prompting_branch", type=str, default="VL")
    parser.add_argument("--prompting_type", type=str, default="SD")
    parser.add_argument("--use_hsf", action="store_true")
    parser.add_argument("--k_clusters", type=int, default=20)
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--class_name", type=str, default="")
    parser.add_argument("--save_name", type=str, default="result.jpg")
    # 添加新参数
    parser.add_argument("--use_memory_bank", action="store_true", help="use memory bank for few-shot anomaly detection")
    parser.add_argument("--mode", type=str, default="zero_shot", choices=["zero_shot", "few_shot"], help="inference mode")
    parser.add_argument("--k_shot", type=int, default=10, help="number of shots for few-shot learning")
    # 添加异常阈值参数
    parser.add_argument("--anomaly_threshold", type=float, default=0.5, help="threshold for anomaly detection")
    # 添加Memory Bank聚类相关参数
    parser.add_argument("--memory_cluster_method", type=str, default="kmeans", choices=["kmeans", "adaptive_pooling", "random_sampling"], help="clustering method for memory bank")
    parser.add_argument("--memory_cluster_size", type=int, default=32, help="number of clusters for memory bank")
    
    args = parser.parse_args()
    test_bondsam(args)
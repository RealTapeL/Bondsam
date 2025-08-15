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
import os

from tools import write2csv, setup_seed, Logger
from dataset import get_data, dataset_dict
from method.trainer import BondSAM_Trainer

from PIL import Image
import numpy as np

setup_seed(111)

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
        use_fastsam=True
    ).to(device)

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

        if hasattr(model, 'fastsam_processor') and model.fastsam_processor is not None:
            fastsam_mask = model.fastsam_processor.process_image(args.image_path)
            fastsam_mask = torch.from_numpy(fastsam_mask).float().unsqueeze(0)
            fastsam_mask = fastsam_mask.to(model.device)
        else:
            fastsam_mask = torch.ones((1, args.image_size, args.image_size), dtype=torch.float32).to(model.device) * 0.5

        with torch.no_grad():
            anomaly_map, anomaly_score = model.clip_model(img_input, [args.class_name], aggregation=True, fastsam_mask=fastsam_mask)

        anomaly_map = anomaly_map[0, :, :]
        anomaly_score = anomaly_score[0]
        anomaly_map = anomaly_map.cpu().numpy()
        anomaly_score = anomaly_score.cpu().numpy()

        anomaly_threshold = 0.5

        if anomaly_score > anomaly_threshold:
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            heatmap_threshold = 0.7
            
            mask = (anomaly_map > heatmap_threshold).astype(np.uint8) * 255
            
            anomaly_map_normalized = ((anomaly_map - anomaly_map.min()) / 
                                    (anomaly_map.max() - anomaly_map.min()) * 255).astype(np.uint8)
            
            heat_map = cv2.applyColorMap(anomaly_map_normalized, cv2.COLORMAP_JET)
            
            heat_map_masked = cv2.bitwise_and(heat_map, heat_map, mask=mask)
            
            vis_map = ori_image.copy()
            vis_map = cv2.addWeighted(vis_map, 1.0, heat_map_masked, 0.7, 0)
            
            # 在图像上添加异常分数文本
            score_text = f"Anomaly Score: {anomaly_score:.3f}"
            cv2.putText(vis_map, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            
            save_path = os.path.join(args.save_path, args.save_name)
            print(f"Anomaly detected! Results saved in {save_path}, anomaly score: {anomaly_score:.3f}")
            cv2.imwrite(save_path, vis_map)
            
            mask_path = os.path.join(args.save_path, "mask_" + args.save_name)
            cv2.imwrite(mask_path, mask)
        else:
            print(f"No anomaly detected. Anomaly score: {anomaly_score:.3f} (threshold: {anomaly_threshold})")
            
            # 即使没有检测到异常，也在图像上显示分数
            vis_map = ori_image.copy()
            score_text = f"Anomaly Score: {anomaly_score:.3f}"
            cv2.putText(vis_map, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            save_path = os.path.join(args.save_path, args.save_name)
            cv2.imwrite(save_path, vis_map)

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

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("BondSAM Wire Bonding Detection", add_help=True)

    parser.add_argument("--ckt_path", type=str, default='weights/pretrained_mvtec.pth',
                        help="Path to the pre-trained model (default: weights/pretrained_mvtec.pth)")

    parser.add_argument("--testing_model", type=str, default="image", choices=["dataset", "image"],
                        help="Model for testing (default: 'dataset')")

    parser.add_argument("--testing_data", type=str, default="mvtec", help="Dataset for testing (default: 'mvtec')")

    parser.add_argument("--image_path", type=str, default="asset/img.png",
                        help="Model for testing (default: 'asset/img.png')")
    parser.add_argument("--class_name", type=str, default="wire_bonding",
                        help="The class name of the testing image (default: 'wire_bonding')")
    parser.add_argument("--save_name", type=str, default="bondsam_test.png",
                        help="Model for testing (default: 'bondsam_test.png')")

    parser.add_argument("--save_path", type=str, default='./workspaces/bondsam_results',
                        help="Directory to save results (default: './workspaces/bondsam_results')")

    parser.add_argument("--model", type=str, default="ViT-L-14",
                        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"],
                        help="The CLIP model to be used (default: 'ViT-L-14')")

    parser.add_argument("--save_fig", type=str2bool, default=True,
                        help="Save figures for visualizations (default: True)")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--image_size", type=int, default=518, help="Size of the input images (default: 518)")

    parser.add_argument("--prompting_depth", type=int, default=4, help="Depth of prompting (default: 4)")
    parser.add_argument("--prompting_length", type=int, default=5, help="Length of prompting (default: 5)")
    parser.add_argument("--prompting_type", type=str, default='SD', choices=['', 'S', 'D', 'SD'],
                        help="Type of prompting. 'S' for Static, 'D' for Dynamic, 'SD' for both (default: 'SD')")
    parser.add_argument("--prompting_branch", type=str, default='VL', choices=['', 'V', 'L', 'VL'],
                        help="Branch of prompting. 'V' for Visual, 'L' for Language, 'VL' for both (default: 'VL')")

    parser.add_argument("--use_hsf", type=str2bool, default=True,
                        help="Use HSF for aggregation. If False, original class embedding is used (default: True)")
    parser.add_argument("--k_clusters", type=int, default=20, help="Number of clusters (default: 20)")
    
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for testing (default: 'cpu')")

    args = parser.parse_args()

    if args.batch_size != 1:
        raise NotImplementedError(
            "Currently, only batch size of 1 is supported due to unresolved bugs. Please set --batch_size to 1.")

    test_bondsam(args)
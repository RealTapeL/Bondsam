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

# Importing from local modules
try:
    from tools import write2csv, setup_seed, Logger
except ImportError:
    # 简单的替代实现
    import logging
    class Logger:
        def __init__(self, log_path):
            logging.basicConfig(filename=log_path, level=logging.INFO)
            self.logger = logging.getLogger()
        
        def info(self, message):
            self.logger.info(message)
    
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        import random
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
    
    def write2csv(metric_dict, cls_names, tag, csv_path):
        pass

try:
    from dataset import get_data, dataset_dict
except ImportError:
    # 占位实现
    dataset_dict = {"mvtec": "mvtec"}
    def get_data(*args, **kwargs):
        class DummyDataset:
            def __len__(self):
                return 10
        return ["dummy"], DummyDataset(), "/tmp"

# 修复导入问题
try:
    from method.trainer import BondSAM_Trainer
except ImportError:
    try:
        from method import BondSAM_Trainer
    except ImportError:
        # 最后的备选方案
        class BondSAM_Trainer:
            def __init__(self, *args, **kwargs):
                raise ImportError("无法导入 BondSAM_Trainer")
            def to(self, device):
                return self
            def load(self, path):
                pass

from PIL import Image
import numpy as np

setup_seed(111)

def test_bondsam(args):
    # 确认预训练模型的路径是否有效
    assert os.path.isfile(args.ckt_path), f"Please check the path of pre-trained model, {args.ckt_path} is not valid."
    
    batch_size = args.batch_size
    image_size = args.image_size
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    save_fig = args.save_fig

    # Logger
    logger = Logger('bondsam_test_log.txt')
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')
    config_path = os.path.join('./model_configs', f'{args.model}.json')


    try:
        with open(config_path, 'r') as f:
            model_configs = json.load(f)
    except FileNotFoundError:
        # 创建默认配置，需要与训练时使用的配置一致
        # 根据模型名称设置不同的默认配置
        if args.model == "ViT-L-14":
            model_configs = {
                "embed_dim": 768,
                "vision_cfg": {
                    "width": 1024,
                    "layers": 24
                },
                "text_cfg": {
                    "width": 768,
                    "layers": 12
                }
            }
        elif args.model == "ViT-L-14-336":
            model_configs = {
                "embed_dim": 768,
                "vision_cfg": {
                    "width": 1024,
                    "layers": 24
                },
                "text_cfg": {
                    "width": 768,
                    "layers": 12
                }
            }
        else:  # ViT-B-16 or ViT-B-32
            model_configs = {
                "embed_dim": 512,
                "vision_cfg": {
                    "width": 768,
                    "layers": 12
                },
                "text_cfg": {
                    "width": 512,
                    "layers": 12
                }
            }

    # 更新模型配置中的图像尺寸
    if 'vision_cfg' in model_configs:
        model_configs['vision_cfg']['image_size'] = image_size

    # Set up the feature hierarchy
    n_layers = model_configs['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]

    # 为引线键合检测配置模型
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
        use_fastsam=True  # 强制启用FastSAM用于引线键合检测
    ).to(device)
# ... existing code ...
    


    if args.testing_model == 'dataset':
        assert args.testing_data in dataset_dict.keys(), f"You entered {args.testing_data}, but we only support " \
                                                         f"{dataset_dict.keys()}"

        save_root = args.save_path
        csv_root = os.path.join(save_root, 'csvs')
        image_root = os.path.join(save_root, 'images')
        csv_path = os.path.join(csv_root, f'{args.testing_data}_bondsam.csv')
        image_dir = os.path.join(image_root, f'{args.testing_data}_bondsam')
        os.makedirs(image_dir, exist_ok=True)

        # 为引线键合检测准备测试数据
        test_data_cls_names, test_data, test_data_root = get_data(
            dataset_type_list=args.testing_data,
            transform=model.preprocess,
            target_transform=model.transform,
            training=False,
            use_fastsam=True  # 强制启用FastSAM
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

        # 为单张图像生成FastSAM mask
        if hasattr(model, 'fastsam_processor') and model.fastsam_processor is not None:
            fastsam_mask = model.fastsam_processor.process_image(args.image_path)
            fastsam_mask = torch.from_numpy(fastsam_mask).float().unsqueeze(0)  # 添加批次维度
            fastsam_mask = fastsam_mask.to(model.device)
        else:
            # 使用默认mask
            fastsam_mask = torch.ones((1, args.image_size, args.image_size), dtype=torch.float32).to(model.device) * 0.5

        with torch.no_grad():
            anomaly_map, anomaly_score = model.clip_model(img_input, [args.class_name], aggregation=True, fastsam_mask=fastsam_mask)

        anomaly_map = anomaly_map[0, :, :]
        anomaly_score = anomaly_score[0]
        anomaly_map = anomaly_map.cpu().numpy()
        anomaly_score = anomaly_score.cpu().numpy()

        # 设置异常阈值
        anomaly_threshold = 0.5  # 您可以根据需要调整这个值

        if anomaly_score > anomaly_threshold:
            # 只有当异常分数超过阈值时才保存图像
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map = anomaly_map * 255
            anomaly_map = anomaly_map.astype(np.uint8)

            heat_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
            
            # 保存热力图
            save_path = os.path.join(args.save_path, args.save_name)
            print(f"Anomaly detected! Results saved in {save_path}, anomaly score: {anomaly_score:.3f}")
            cv2.imwrite(save_path, heat_map)
            
            # 如果您想保存叠加的异常图像（原图+热力图），取消下面两行的注释
            vis_map = cv2.addWeighted(heat_map, 0.5, ori_image, 0.5, 0)
            cv2.imwrite(save_path, vis_map)
            
            # 方案3: 如果您想保存原始图像和结果图像的对比图，取消下面几行的注释
            # vis_map = cv2.addWeighted(heat_map, 0.5, ori_image, 0.5, 0)
            # combined_result = cv2.hconcat([ori_image, vis_map])
            # cv2.imwrite(save_path, combined_result)
        else:
            print(f"No anomaly detected. Anomaly score: {anomaly_score:.3f} (threshold: {anomaly_threshold})")
            
            # 保存原始图像
            save_path = os.path.join(args.save_path, args.save_name)
            cv2.imwrite(save_path, ori_image)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("BondSAM Wire Bonding Detection", add_help=True)

    # Paths and configurations
    parser.add_argument("--ckt_path", type=str, default='weights/pretrained_mvtec.pth',
                        help="Path to the pre-trained model (default: weights/pretrained_mvtec.pth)")

    parser.add_argument("--testing_model", type=str, default="image", choices=["dataset", "image"],
                        help="Model for testing (default: 'dataset')")

    # for the dataset model
    parser.add_argument("--testing_data", type=str, default="mvtec", help="Dataset for testing (default: 'mvtec')")

    # for the image model
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

    # Hyper-parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--image_size", type=int, default=518, help="Size of the input images (default: 518)")

    # Prompting parameters
    parser.add_argument("--prompting_depth", type=int, default=4, help="Depth of prompting (default: 4)")
    parser.add_argument("--prompting_length", type=int, default=5, help="Length of prompting (default: 5)")
    parser.add_argument("--prompting_type", type=str, default='SD', choices=['', 'S', 'D', 'SD'],
                        help="Type of prompting. 'S' for Static, 'D' for Dynamic, 'SD' for both (default: 'SD')")
    parser.add_argument("--prompting_branch", type=str, default='VL', choices=['', 'V', 'L', 'VL'],
                        help="Branch of prompting. 'V' for Visual, 'L' for Language, 'VL' for both (default: 'VL')")

    parser.add_argument("--use_hsf", type=str2bool, default=True,
                        help="Use HSF for aggregation. If False, original class embedding is used (default: True)")
    parser.add_argument("--k_clusters", type=int, default=20, help="Number of clusters (default: 20)")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for testing (default: 'cpu')")

    args = parser.parse_args()

    if args.batch_size != 1:
        raise NotImplementedError(
            "Currently, only batch size of 1 is supported due to unresolved bugs. Please set --batch_size to 1.")

    test_bondsam(args)
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

# Importing from local modules
try:
    from tools import write2csv, setup_paths, setup_seed, log_metrics, Logger
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
    
    def setup_paths(args):
        # 简单实现
        return "model", "./images", "./results.csv", "./log.txt", "./checkpoint.pth", None
    
    def write2csv(metric_dict, cls_names, tag, csv_path):
        pass
    
    def log_metrics(metric_dict, logger, tensorboard_logger, epoch):
        pass

try:
    from dataset import get_data
except ImportError:
    # 占位实现
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

setup_seed(111)

def finetune_bondsam(args):
    # Configurations
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    save_fig = args.save_fig

    # Set up paths
    model_name, image_dir, csv_path, log_path, ckp_path, tensorboard_logger = setup_paths(args)
    # Logger
    logger = Logger(log_path)

    # Print basic information
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')

    logger.info('Model name: {:}'.format(model_name))

    config_path = os.path.join('./model_configs', f'{args.model}.json')

    # Prepare model
    try:
        with open(config_path, 'r') as f:
            model_configs = json.load(f)
    except FileNotFoundError:
        # 创建默认配置
        model_configs = {
            "embed_dim": 768,
            "vision_cfg": {
                "width": 768,
                "layers": 12
            }
        }

    # Set up the feature hierarchy
    n_layers = model_configs['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]

    # 为引线键合检测专门配置模型
    model = BondSAM_Trainer(
        backbone=args.model,
        feat_list=features_list,
        input_dim=model_configs['vision_cfg']['width'],
        output_dim=model_configs['embed_dim'],
        learning_rate=learning_rate,
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
    
    # 加载预训练模型
    if args.pretrained_model and os.path.isfile(args.pretrained_model):
        model.load(args.pretrained_model)
        logger.info(f'Loaded pre-trained model from {args.pretrained_model}')

    # 为引线键合检测准备数据 (使用FastSAM分割后的少样本数据)
    train_data_cls_names, train_data, train_data_root = get_data(
        dataset_type_list=args.training_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=True,
        use_fastsam=True  # 强制启用FastSAM
    )

    test_data_cls_names, test_data, test_data_root = get_data(
        dataset_type_list=args.testing_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=False,
        use_fastsam=True  # 强制启用FastSAM
    )

    logger.info('Data Root: training, {:}; testing, {:}'.format(train_data_root, test_data_root))

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 引线键合检测的验证策略
    best_f1 = -1e1

    for epoch in tqdm(range(epochs)):
        loss = model.train_epoch(train_dataloader)

        # Logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss))
            if tensorboard_logger:
                tensorboard_logger.add_scalar('loss', loss, epoch)

        # Validation
        if (epoch + 1) % args.valid_freq == 0 or (epoch == epochs - 1):
            if epoch == epochs - 1:
                save_fig_flag = save_fig
            else:
                save_fig_flag = False

            logger.info('=============================Testing ====================================')
            metric_dict = model.evaluation(
                test_dataloader,
                test_data_cls_names,
                save_fig_flag,
                image_dir,
            )

            log_metrics(
                metric_dict,
                logger,
                tensorboard_logger,
                epoch
            )

            f1_px = metric_dict['Average']['f1_px']

            # Save best
            if f1_px > best_f1:
                for k in metric_dict.keys():
                    write2csv(metric_dict[k], test_data_cls_names, k, csv_path)

                ckp_path_best = ckp_path + '_best.pth'
                model.save(ckp_path_best)
                best_f1 = f1_px

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("BondSAM Wire Bonding Detection - Fine-tuning", add_help=True)

    # Paths and configurations
    parser.add_argument("--training_data", type=str, default=["wirebonding_fewshot"], nargs='+',
                        help="Datasets for fine-tuning (default: ['wirebonding_fewshot'])")
    parser.add_argument("--testing_data", type=str, default="wirebonding_test", help="Dataset for testing (default: 'wirebonding_test')")

    parser.add_argument("--pretrained_model", type=str, default="", help="Path to the pre-trained model for fine-tuning")
    parser.add_argument("--save_path", type=str, default='./workspaces/bondsam_finetune',
                        help="Directory to save results (default: './workspaces/bondsam_finetune')")

    parser.add_argument("--model", type=str, default="ViT-L-14",
                        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"],
                        help="The CLIP model to be used (default: 'ViT-L-14')")

    parser.add_argument("--save_fig", type=str2bool, default=True,
                        help="Save figures for visualizations (default: True)")

    # Hyper-parameters
    parser.add_argument("--epoch", type=int, default=20, help="Number of epochs (default: 20)")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate (default: 0.0001)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")

    parser.add_argument("--image_size", type=int, default=518, help="Size of the input images (default: 518)")
    parser.add_argument("--print_freq", type=int, default=1, help="Frequency of print statements (default: 1)")
    parser.add_argument("--valid_freq", type=int, default=1, help="Frequency of validation (default: 1)")

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
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device to use for fine-tuning (default: 'cuda')")

    args = parser.parse_args()

    if args.batch_size != 1:
        raise NotImplementedError(
            "Currently, only batch size of 1 is supported due to unresolved bugs. Please set --batch_size to 1.")

    finetune_bondsam(args)
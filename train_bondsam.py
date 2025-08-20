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
import torchvision.transforms as transforms
from datetime import datetime

# Importing from local modules
try:
    from tools import write2csv, setup_paths, setup_seed, log_metrics, Logger
except ImportError:
    # Simple fallback implementation
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
        # 创建符合要求的路径结构
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = args.save_path
        
        # 创建logs目录和以时间戳命名的子目录
        logs_subdir = os.path.join(save_dir, "logs", timestamp)
        images_dir = os.path.join(save_dir, "images")
        
        os.makedirs(logs_subdir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # 日志文件路径，加上时间戳
        log_path = os.path.join(logs_subdir, f"train_{timestamp}.log")
        
        # 模型文件路径也在同一个目录下
        ckp_path = os.path.join(logs_subdir, "checkpoint")
        csv_path = os.path.join(save_dir, "results.csv")
        
        return "model", images_dir, csv_path, log_path, ckp_path, None
    
    def write2csv(metric_dict, cls_names, tag, csv_path):
        pass
    
    def log_metrics(metric_dict, logger, tensorboard_logger, epoch):
        pass

try:
    from dataset import get_data
except ImportError:
    # Placeholder implementation
    def get_data(*args, **kwargs):
        class DummyDataset:
            def __len__(self):
                return 10
        return ["dummy"], DummyDataset(), "/tmp"

# Fix import issue
try:
    from method.trainer import BondSAM_Trainer
except ImportError:
    try:
        from method import BondSAM_Trainer
    except ImportError:
        # Final fallback
        class BondSAM_Trainer:
            def __init__(self, *args, **kwargs):
                raise ImportError("Cannot import BondSAM_Trainer")

setup_seed(111)

def train_bondsam(args):
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
        learning_rate=learning_rate,
        device=device,
        image_size=image_size,
        prompting_depth=args.prompting_depth,
        prompting_length=args.prompting_length,
        prompting_branch=args.prompting_branch,
        prompting_type=args.prompting_type,
        use_hsf=args.use_hsf,
        k_clusters=args.k_clusters,
        use_fastsam=args.use_fastsam,
        use_memory_bank=args.use_memory_bank,  # 添加Memory Bank选项
        k_shot=args.k_shot  # 添加少样本数量参数
    ).to(device)

    # 获取训练数据
    logger.info("Loading training data...")
    train_data_cls_names, train_data, train_data_root = get_data(
        dataset_type_list=args.training_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=True,
        use_fastsam=args.use_fastsam
    )
    logger.info(f"Training data loaded. Classes: {train_data_cls_names}, Total samples: {len(train_data)}")

    # 如果使用Memory Bank且是少样本模式，构建Memory Bank
    if args.use_memory_bank and args.mode == 'few_shot':
        logger.info("Starting to build Memory Bank for few-shot anomaly detection...")
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
        
        # 构建Memory Bank
        if hasattr(model, 'memory_bank') and model.memory_bank:
            logger.info(f"Building Memory Bank with k_shot={args.k_shot}")
            memory_features = model.memory_bank.build_memory_bank(
                model, train_dataloader, train_data_cls_names, k_shot=args.k_shot
            )
            logger.info(f"Memory Bank built with {len(memory_features)} classes")
        else:
            logger.info("Memory Bank not available, skipping...")

    # 获取测试数据
    logger.info("Loading test data...")
    test_data_cls_names, test_data, test_data_root = get_data(
        dataset_type_list=args.training_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=False,
        use_fastsam=args.use_fastsam
    )
    logger.info(f"Test data loaded. Classes: {test_data_cls_names}, Total samples: {len(test_data)}")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Training
    best_loss = float('inf')
    logger.info("Starting training loop...")
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        # Training epoch with progress bar
        train_loader_with_progress = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        train_loss = model.train_epoch_with_progress(train_loader_with_progress)
        
        # Evaluation
        if epoch % args.eval_interval == 0:
            logger.info(f"Evaluating at epoch {epoch}")
            metric_dict = model.evaluation(test_dataloader, test_data_cls_names, save_fig=False)
            logger.info(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.4f}')
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
            
            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                model.save(f'{ckp_path}_best.pth')
                logger.info(f'Best model saved at epoch {epoch} with loss {best_loss:.4f}')
        
        # Save checkpoint
        if epoch % args.save_interval == 0 and epoch > 0:
            model.save(f'{ckp_path}_epoch{epoch}.pth')
            logger.info(f'Model checkpoint saved at epoch {epoch}')

    logger.info('Training completed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training BondSAM')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--save_fig", action="store_true")
    parser.add_argument("--save_path", type=str, default="./workspaces/bondsam_exp")
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--model", type=str, default="ViT-B-16")
    parser.add_argument("--prompting_depth", type=int, default=3)
    parser.add_argument("--prompting_length", type=int, default=2)
    parser.add_argument("--prompting_branch", type=str, default="VL")
    parser.add_argument("--prompting_type", type=str, default="SD")
    parser.add_argument("--use_hsf", action="store_true")
    parser.add_argument("--k_clusters", type=int, default=20)
    parser.add_argument("--training_data", type=str, default="mvtec")
    parser.add_argument("--use_fastsam", action="store_true")
    # 添加新参数
    parser.add_argument("--use_memory_bank", action="store_true", help="use memory bank for few-shot anomaly detection")
    parser.add_argument("--mode", type=str, default="zero_shot", choices=["zero_shot", "few_shot"], help="training mode")
    parser.add_argument("--k_shot", type=int, default=10, help="number of shots for few-shot learning")
    
    args = parser.parse_args()
    train_bondsam(args)
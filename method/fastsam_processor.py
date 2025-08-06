import torch
import numpy as np
import sys
import os

class FastSAMProcessor:
    def __init__(self, model_path=None, device='cpu'):
        """
        初始化FastSAM处理器
        
        Args:
            model_path: FastSAM模型路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        # 设置FastSAM目录
        fastsam_dir = '/home/ps/few-shot-research/Bondsam/FastSAM'
        
        # 添加FastSAM到Python路径
        sys.path.append(fastsam_dir)
        
        if model_path is None:
            # 默认使用FastSAM.pt
            model_path = os.path.join(fastsam_dir, 'weights', 'FastSAM.pt')
            
        try:
            if os.path.exists(model_path):
                # 尝试导入FastSAM
                try:
                    from fastsam import FastSAM
                    self.model = FastSAM(model_path)
                    self.device = device
                    print(f"FastSAM processor initialized with model: {model_path}")
                except ImportError as e:
                    print(f"Failed to import FastSAM: {e}")
                    print("Trying alternative import method...")
                    # 尝试直接导入
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("fastsam", os.path.join(fastsam_dir, "fastsam.py"))
                    fastsam_module = importlib.util.module_from_spec(spec)
                    sys.modules["fastsam"] = fastsam_module
                    spec.loader.exec_module(fastsam_module)
                    self.model = fastsam_module.FastSAM(model_path)
                    self.device = device
                    print(f"FastSAM processor initialized with model: {model_path}")
            else:
                print(f"FastSAM model not found at: {model_path}")
                print("Available files in weights directory:")
                weights_dir = os.path.join(fastsam_dir, 'weights')
                if os.path.exists(weights_dir):
                    for f in os.listdir(weights_dir):
                        print(f"  {f}")
                self.model = None
        except Exception as e:
            print(f"Failed to initialize FastSAM: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
        
    def process_image(self, image_path, confidence=0.4, iou=0.9, imgsz=1024):
        """
        处理单张图像，获取分割mask
        
        Args:
            image_path: 图像路径
            confidence: 置信度阈值
            iou: IOU阈值
            imgsz: 图像尺寸
            
        Returns:
            segmentation_mask: 分割mask
        """
        if self.model is None:
            # 返回默认mask
            import cv2
            image = cv2.imread(image_path)
            if image is not None:
                return np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * 0.5
            else:
                return np.zeros((224, 224), dtype=np.float32)
        
        try:
            from PIL import Image
            from fastsam import FastSAMPrompt
            
            # 加载图像
            input_image = Image.open(image_path).convert("RGB")
            
            # 运行FastSAM
            everything_results = self.model(
                input_image,
                device=self.device,
                retina_masks=True,
                imgsz=imgsz,
                conf=confidence,
                iou=iou    
            )
            
            # 使用everything prompt获取所有分割结果
            prompt_process = FastSAMPrompt(input_image, everything_results, device=self.device)
            ann = prompt_process.everything_prompt()
            
            # 如果有分割结果，合并所有mask
            if len(ann) > 0 and hasattr(ann[0], 'masks') and ann[0].masks is not None:
                masks_data = ann[0].masks.data
                if len(masks_data) > 0:
                    # 合并所有mask
                    combined_mask = torch.zeros_like(masks_data[0], dtype=torch.float32)
                    for mask in masks_data:
                        combined_mask = torch.logical_or(combined_mask, mask)
                    return combined_mask.cpu().numpy()
                    
        except Exception as e:
            print(f"Error processing FastSAM for {image_path}: {e}")
            import traceback
            traceback.print_exc()
        
        # 如果没有检测到对象或出错，返回空mask
        import cv2
        image = cv2.imread(image_path)
        if image is not None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            return np.zeros((224, 224), dtype=np.float32)
    
    def process_tensor(self, image_tensor, confidence=0.4, iou=0.9, imgsz=1024):
        """
        处理张量图像，获取分割mask
        
        Args:
            image_tensor: 图像张量 (B, C, H, W)
            confidence: 置信度阈值
            iou: IOU阈值
            imgsz: 图像尺寸
            
        Returns:
            segmentation_masks: 分割masks
        """
        if self.model is None:
            # 返回默认mask
            if image_tensor.dim() == 4:
                # 批处理
                batch_size = image_tensor.shape[0]
                masks = []
                for i in range(batch_size):
                    h, w = image_tensor.shape[2], image_tensor.shape[3]
                    mask = np.ones((h, w), dtype=np.float32) * 0.5
                    masks.append(mask)
                return np.stack(masks, axis=0)
            else:
                # 单张图像
                h, w = image_tensor.shape[1], image_tensor.shape[2]
                return np.ones((h, w), dtype=np.float32) * 0.5
        
        try:
            from PIL import Image
            from fastsam import FastSAMPrompt
            import torchvision.transforms as transforms
            
            # 将张量转换为PIL图像
            if image_tensor.dim() == 4:
                # 批处理
                batch_size = image_tensor.shape[0]
                masks = []
                for i in range(batch_size):
                    # 转换张量到PIL图像
                    img = transforms.ToPILImage()(image_tensor[i].cpu())
                    mask = self._process_pil_image(img, confidence, iou, imgsz)
                    masks.append(mask)
                return np.stack(masks, axis=0)
            else:
                # 单张图像
                img = transforms.ToPILImage()(image_tensor.cpu())
                return self._process_pil_image(img, confidence, iou, imgsz)
        except Exception as e:
            print(f"Error processing tensor with FastSAM: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认mask
            if image_tensor.dim() == 4:
                batch_size = image_tensor.shape[0]
                h, w = image_tensor.shape[2], image_tensor.shape[3]
                return np.ones((batch_size, h, w), dtype=np.float32) * 0.5
            else:
                h, w = image_tensor.shape[1], image_tensor.shape[2]
                return np.ones((h, w), dtype=np.float32) * 0.5
    
    def _process_pil_image(self, pil_image, confidence, iou, imgsz):
        """
        处理PIL图像
        """
        try:
            # 运行FastSAM
            everything_results = self.model(
                pil_image,
                device=self.device,
                retina_masks=True,
                imgsz=imgsz,
                conf=confidence,
                iou=iou    
            )
            
            # 使用everything prompt获取所有分割结果
            from fastsam import FastSAMPrompt
            prompt_process = FastSAMPrompt(pil_image, everything_results, device=self.device)
            ann = prompt_process.everything_prompt()
            
            # 如果有分割结果，合并所有mask
            if len(ann) > 0 and hasattr(ann[0], 'masks') and ann[0].masks is not None:
                masks_data = ann[0].masks.data
                if len(masks_data) > 0:
                    # 合并所有mask
                    combined_mask = torch.zeros_like(masks_data[0], dtype=torch.float32)
                    for mask in masks_data:
                        combined_mask = torch.logical_or(combined_mask, mask)
                    return combined_mask.cpu().numpy()
        except Exception as e:
            print(f"Error processing PIL image with FastSAM: {e}")
            import traceback
            traceback.print_exc()
        
        # 返回默认mask
        # 将PIL图像转换为numpy获取尺寸
        img_array = np.array(pil_image)
        return np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.float32) * 0.5

def create_fastsam_processor(model_path=None, device='cpu'):
    """
    创建FastSAM处理器实例
    """
    return FastSAMProcessor(model_path, device)
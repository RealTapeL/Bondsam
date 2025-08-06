from .trainer import BondSAM_Trainer
from .bondsam_model import BondSAM
from .fastsam_processor import FastSAMProcessor, create_fastsam_processor

try:
    from .clip_model import create_model_and_transforms
except ImportError:
    def create_model_and_transforms(*args, **kwargs):
        import torchvision.transforms as transforms
        import torch.nn as nn
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = nn.Module()
                self.transformer = nn.Module()
                self.token_embedding = nn.Module()
                self.positional_embedding = torch.tensor([])
                self.ln_final = nn.Module()
                self.text_projection = torch.tensor([])
                self.attn_mask = None
            
            def to(self, device):
                return self
                
            def eval(self):
                return self
        
        return DummyModel(), None, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

__all__ = [
    'BondSAM_Trainer',
    'BondSAM',
    'FastSAMProcessor',
    'create_fastsam_processor',
    'create_model_and_transforms'
]
import os
from .base_dataset import BaseDataset
from config import MVTec_ROOT

'''MVTec数据集: https://www.mvtec.com/company/research/datasets/mvtec-ad'''

MVTec_CLS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]

class MVTecDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=MVTec_CLS_NAMES, 
                 aug_rate=0.2, root=MVTec_ROOT, training=True):
        super(MVTecDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
from .mvtec import MVTec_CLS_NAMES, MVTecDataset, MVTec_ROOT
from .base_dataset import BaseDataset
from torch.utils.data import ConcatDataset

dataset_dict = {
    'mvtec': (MVTec_CLS_NAMES, MVTecDataset, MVTec_ROOT),
}

def get_data(dataset_type_list, transform, target_transform, training, use_fastsam=False):
    if not isinstance(dataset_type_list, list):
        dataset_type_list = [dataset_type_list]

    dataset_cls_names_list = []
    dataset_instance_list = []
    dataset_root_list = []
    for dataset_type in dataset_type_list:
        if dataset_dict.get(dataset_type, ''):
            dataset_cls_names, dataset_instance, dataset_root = dataset_dict[dataset_type]
            dataset_instance = dataset_instance(
                clsnames=dataset_cls_names,
                transform=transform,
                target_transform=target_transform,
                training=training
            )

            dataset_cls_names_list.append(dataset_cls_names)
            dataset_instance_list.append(dataset_instance)
            dataset_root_list.append(dataset_root)

        else:
            print(f'Only support {list(dataset_dict.keys())}, but entered {dataset_type}...')
            raise NotImplementedError

    if len(dataset_type_list) > 1:
        dataset_instance = ConcatDataset(dataset_instance_list)
        dataset_cls_names = dataset_cls_names_list
        dataset_root = dataset_root_list
    else:
        dataset_instance = dataset_instance_list[0]
        dataset_cls_names = dataset_cls_names_list[0]
        dataset_root = dataset_root_list[0]

    return dataset_cls_names, dataset_instance, dataset_root
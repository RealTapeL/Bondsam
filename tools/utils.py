import pandas as pd
import os
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import random
import torch
import numpy as np
import logging
from datetime import datetime


def write2csv(results:dict, total_classes, cur_class,  csv_path):
    keys = list(results.keys())

    if not os.path.exists(csv_path):
        df_all = None
        for class_name in total_classes:
            r = dict()
            for k in keys:
                r[k] = 0.00
            df_temp = pd.DataFrame(r, index=[f'{class_name}'])

            if df_all is None:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all, df_temp], axis=0)

        df_all.to_csv(csv_path, header=True, float_format='%.2f')

    df = pd.read_csv(csv_path, index_col=0)

    for k in keys:
        df.loc[f'{cur_class}', k] = results[k]

    df.to_csv(csv_path, header=True, float_format='%.2f')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class Logger(object):
    def __init__(self, txt_path):
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.WARNING)
        self.txt_path = txt_path
        self.logger = logging.getLogger('train')
        self.formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        self.logger.setLevel(logging.INFO)

    def __console(self, level, message):
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        file_handler = logging.FileHandler(self.txt_path, mode='a')
        console_handler = logging.StreamHandler()

        file_handler.setFormatter(self.formatter)
        console_handler.setFormatter(self.formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)

        self.logger.removeHandler(file_handler)
        self.logger.removeHandler(console_handler)

        file_handler.close()

    def debug(self, message):
        self.__console('debug', message)

    def info(self, message):
        self.__console('info', message)

    def warning(self, message):
        self.__console('warning', message)

    def error(self, message):
        self.__console('error', message)
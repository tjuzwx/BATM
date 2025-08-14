"""
Data processing module
It includes data loading, preprocessing and management functions
"""

from .data_loader import DataManager, get_data_manager


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.data_utils import CustomDataset, get_datasets, DATASETS
except ImportError:
    # 如果导入失败，提供占位符
    CustomDataset = None
    get_datasets = None
    DATASETS = {}

__all__ = ['DataManager', 'get_data_manager', 'CustomDataset', 'get_datasets', 'DATASETS']

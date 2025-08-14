"""
Data processing module
It includes data loading, preprocessing and management functions
"""

from .data_loader import Regression, Classfication_corrupted, Classfication_imbalance, Classfication_multi


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.data_utils import CustomDataset, get_datasets, DATASETS
except ImportError:

    CustomDataset = None
    get_datasets = None
    DATASETS = {}

__all__ = ['Regression', 'Classfication_corrupted', 'Classfication_imbalance', 'Classfication_multi', 'CustomDataset', 'get_datasets', 'DATASETS']

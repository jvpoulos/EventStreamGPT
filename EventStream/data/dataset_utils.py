# dataset_utils.py

def get_dataset_class():
    from .dataset_polars import Dataset
    return Dataset
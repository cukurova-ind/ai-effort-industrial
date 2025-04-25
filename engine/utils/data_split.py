import pandas as pd
from sklearn.model_selection import train_test_split
import random
import os


def save_splits(conf, split_ratio=0.1, rs=None):
    main_path = conf["dataset"]["main_path"]
    version_path = conf["dataset"]["version_path"]
    input_file_name = conf["dataset"]["input_file_name"]
    train_path = conf["dataset"]["train_path"]
    val_path = conf["dataset"]["val_path"]
    features = os.path.join(main_path, version_path, input_file_name)
    df_feat = pd.read_csv(features)
    train_ds, val_ds = train_test_split(df_feat, test_size=split_ratio, random_state=rs)
    train_ds.to_csv(os.path.join(main_path, version_path, train_path, input_file_name), index=None)
    val_ds.to_csv(os.path.join(main_path, version_path, val_path, input_file_name), index=None)

def save_splits_by_type(conf, num_val_types=2, rs=None):

    if rs is not None:
        random.seed(rs)
    
    main_path = conf["dataset"]["main_path"]
    version_path = conf["dataset"]["version_path"]
    input_file_name = conf["dataset"]["input_file_name"]
    train_path = conf["dataset"]["train_path"]
    val_path = conf["dataset"]["val_path"]
    features = os.path.join(main_path, version_path, input_file_name)
    df_feat = pd.read_csv(features)

    unique_types = df_feat['type'].unique()
    val_types = random.sample(list(unique_types), num_val_types)

    val_ds = df_feat[df_feat['type'].isin(val_types)]
    train_ds = df_feat[~df_feat['type'].isin(val_types)]

    train_ds.to_csv(os.path.join(main_path, version_path, train_path, input_file_name), index=False)
    val_ds.to_csv(os.path.join(main_path, version_path, val_path, input_file_name), index=False)

def random_split(df, split_ratio=0.1, rs=None):
    train_ds, test_ds = train_test_split(df, test_size=split_ratio, random_state=rs)
    return train_ds, test_ds
    
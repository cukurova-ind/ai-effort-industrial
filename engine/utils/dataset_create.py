import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from .load_config import load_config
from .data_split import random_split


class CustomDataset(Dataset):
    def __init__(self, conf, df, conditional=True, image=True):
        self.conf = conf
        self.img_width = int(conf["image"]["img_width"])
        self.img_height = int(conf["image"]["img_height"])
        self.channels = int(conf["image"]["n_channels"])
        
        self.data_csv = os.path.join("media", self.conf["dataset"]["main_path"], self.conf["dataset"]["csv_folder"])
        self.input_features = conf["dataset"]["input_features"].split(",")
        self.df = df
        self.input_maxs = [float(mx) for mx in conf["dataset"]["input_maxs"].split(",")]
        self.input_mins = [float(mn) for mn in conf["dataset"]["input_mins"].split(",")]
        self.conditional = conditional

        self.image = image
        self._set_paths()

        self.input_data, self.image_paths, self.target_image_paths, self.recipes = self._load_data()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels)
        ])

    def _load_data(self):
        
        features_df = self.df
        input_data = features_df[self.input_features].astype(np.float32)
        
        if self.conf["feature_normalization"]:
            for i, c in enumerate(input_data.columns):
                input_data.loc[:, c] = 2 * (input_data.loc[:, c] - self.input_mins[i])/(self.input_maxs[i] - self.input_mins[i]) - 1

        input_data = input_data[self.input_features].values.astype(np.float32)
        
        image_paths = features_df['input_image'].values if self.image else None
        target_image_paths = features_df['output_image'].values if self.image else None

        recipes = features_df['recipe_id'].values.astype(np.int32)

        return input_data, image_paths, target_image_paths, recipes
    
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        
        input_feat = torch.tensor(self.input_data[idx])

        if self.image and self.image_paths is not None:
            
            input_image_path = os.path.join(self.data_version, self.conf["dataset"]["raw_image_folder"], self.image_paths[idx])
            input_image = Image.open(input_image_path).convert("RGB")
            input_image = self.transform(input_image)

            target_image_path = os.path.join(self.data_version, self.conf["dataset"]["image_folder"], self.target_image_paths[idx])
            target_image = Image.open(target_image_path).convert("RGB")
            target_image = self.transform(target_image)

            if self.to=="test":
                return input_image, input_feat, target_image, target_image
            else:
                current_recipe = self.recipes[idx]
                diff_recipe_indices = np.where(self.recipes != current_recipe)[0]

                if len(diff_recipe_indices) == 0:
                    wrong_idx = (idx + np.random.randint(1, len(self.target_image_paths))) % len(self.target_image_paths)
                else:
                    wrong_idx = np.random.choice(diff_recipe_indices)

                wrong_image_path = os.path.join(self.data_version, self.conf["dataset"]["image_folder"], self.target_image_paths[wrong_idx])
                wrong_image = Image.open(wrong_image_path).convert("RGB")
                wrong_image = self.transform(wrong_image)

            return input_image, input_feat, target_image, wrong_image
        else:
            return None
        
class RegressionDataset(Dataset):
    def __init__(self, conf, df):
        self.conf = conf

        self.df = df
        self.input_features = conf["input_features"].split(",")
        self.target_features = conf["target_features"].split(",")
        self.feature_types = conf["input_feature_types"].split(",")
        self.input_maxs = [float(mx) for mx in conf["input_maxs"].split(",")]
        self.input_mins = [float(mn) for mn in conf["input_mins"].split(",")]

        self.input_data, self.target_data = self._load_data()

    def _load_data(self):
        
        features_df = self.df
        input_data = features_df[self.input_features]
        
        for i, c in enumerate(input_data.columns):
            if self.feature_types[i]=="disc":
                input_data = pd.concat([input_data, pd.get_dummies(input_data.pop(c))], axis=1)
            else:
                if self.conf["input_scaling"]:
                    input_data.loc[:, c] = 2 * (input_data.loc[:, c] - self.input_mins[i])/(self.input_maxs[i] - self.input_mins[i]) - 1

        input_data = input_data.values.astype(np.float32)
        target_data = features_df[self.target_features].values.astype(np.float32)

        return input_data, target_data
    
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        
        input_feat = torch.tensor(self.input_data[idx])
        target_feat = torch.tensor(self.target_data[idx])

        return input_feat, target_feat

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

def create_custom_dataset(df, path):
    conf = load_config(path)   
    train_df = df
    test_dl = None
    val_dl = None
    if float(conf["test_size"]) > 0:
        random_state = int(conf["random_state"]) if int(conf["random_state"])>-1 else None
        train_df, test_df = random_split(df, float(conf["test_size"]), random_state)
        if conf["single_shot_validation"]=="on" and float(conf["val_size"]) > 0:
            train_df, val_df = random_split(train_df, float(conf["val_size"]), random_state)
            val_ds = RegressionDataset(conf, val_df)
            val_dl = DataLoader(val_ds, batch_size=int(conf["batch_size"]), num_workers=4, pin_memory=True, shuffle=True)
        test_ds = RegressionDataset(conf, test_df)
        test_dl = DataLoader(test_ds, batch_size=int(conf["batch_size"]), num_workers=4, pin_memory=True, shuffle=True)
    train_ds = RegressionDataset(conf, train_df)
    train_dl = DataLoader(train_ds, batch_size=int(conf["batch_size"]), num_workers=4, pin_memory=True, shuffle=True)

    channel_layer = get_channel_layer()
    group_name = f"train_{conf["username"]}"

    async_to_sync(channel_layer.group_send)(
        group_name,
        {
            "type": "operation_message",
            "message": "data loader created."
        },
    )

    return train_dl, test_dl, val_dl
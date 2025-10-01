import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from .load_config import load_config
from .data_split import random_split


class ImageDataset(Dataset):
    def __init__(self, conf, df, to="train", conditional=True):
        self.conf = conf
        self.df = df
        self.img_width = int(conf["img_width"])
        self.img_height = int(conf["img_height"])
        self.channels = int(conf["n_channels"])
        
        self.column_list = conf["column_list"]
        self.input_features = conf["input_features"]
        self.target_features = conf["target_features"]
        self.feature_types = conf["input_feature_types"]
        self.input_maxs = [float(mx) for mx in conf["input_maxs"]]
        self.input_mins = [float(mn) for mn in conf["input_mins"]]
        self.input_categories = conf["input_categories"]
        self.conditional = conditional
        self.to = to

        self.input_data, self.input_image_paths, self.target_image_paths, self.labels = self._load_data()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels)
        ])

    def _load_data(self):
        
        features_df = self.df
        input_data = features_df[self.input_features].reset_index().drop(columns="index")

        new_df = pd.DataFrame() 
        for i, c in enumerate(input_data.columns):
            if self.feature_types[i]=="disc":
                encode_dict = self.input_categories.get(c)
                input_data[c] = input_data[c].map(encode_dict)
                one_hot = np.eye(len(encode_dict))[input_data[c].values]
                sorted_keys = sorted(encode_dict, key=encode_dict.get)
                cat_df = pd.DataFrame(one_hot, columns=sorted_keys)
                new_df = pd.concat([new_df, cat_df], axis=1)
            else:
                if self.conf["input_scaling"]=="norm2":
                    input_data.loc[:, c] = input_data.loc[:, c].astype(np.float32)
                    new_df.loc[:, c] = 2 * (input_data.loc[:, c] - self.input_mins[i])/(self.input_maxs[i] - self.input_mins[i]) - 1
                elif self.conf["input_scaling"]=="norm1":
                    input_data.loc[:, c] = input_data.loc[:, c].astype(np.float32)
                    new_df.loc[:, c] = (input_data.loc[:, c] - self.input_mins[i])/(self.input_maxs[i] - self.input_mins[i])
                else:
                    input_data.loc[:, c] = input_data.loc[:, c].astype(np.float32)
                    new_df.loc[:, c] = input_data.loc[:, c]

        new_input_data = new_df.values.astype(np.float32)
        
        input_image_paths = features_df['input_image'].values
        target_image_paths = features_df['output_image'].values

        labels = features_df['recipe_id'].values.astype(np.int32)

        return new_input_data, input_image_paths, target_image_paths, labels
    
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        
        input_feat = torch.tensor(self.input_data[idx])

        if self.input_image_paths is not None:
            
            input_image_path = os.path.join(self.conf["raw_image_folder"], self.input_image_paths[idx])
            input_image = Image.open(input_image_path).convert("RGB")
            input_image = self.transform(input_image)

            target_image_path = os.path.join(self.conf["output_image_folder"], self.target_image_paths[idx])
            target_image = Image.open(target_image_path).convert("RGB")
            target_image = self.transform(target_image)

            if self.to=="test":
                return input_feat, input_image, target_image, target_image
            else:
                current_label = self.labels[idx]
                diff_label_indices = np.where(self.labels != current_label)[0]

                if len(diff_label_indices) == 0:
                    wrong_idx = (idx + np.random.randint(1, len(self.target_image_paths))) % len(self.target_image_paths)
                else:
                    wrong_idx = np.random.choice(diff_label_indices)

                wrong_image_path = os.path.join(self.conf["output_image_folder"], self.target_image_paths[wrong_idx])
                wrong_image = Image.open(wrong_image_path).convert("RGB")
                wrong_image = self.transform(wrong_image)

            return input_feat, input_image, target_image, wrong_image
        else:
            return None
        
        
class RegressionDataset(Dataset):
    def __init__(self, conf, df, to="train", input_image=False):
        self.conf = conf
        self.to = to
        self.df = df

        self.input_image = input_image
        self.img_width = int(conf["img_width"])
        self.img_height = int(conf["img_height"])
        self.channels = int(conf["n_channels"])

        self.column_list = conf["column_list"]
        self.input_features = conf["input_features"]
        self.target_features = conf["target_features"]
        self.feature_types = conf["input_feature_types"]
        self.input_maxs = [float(mx) for mx in conf["input_maxs"]]
        self.input_mins = [float(mn) for mn in conf["input_mins"]]
        self.input_categories = conf["input_categories"]

        self.image_paths = None
        if self.input_image:
            self.input_data, self.image_paths, self.target_data = self._load_data()
        else:
            self.input_data, self.target_data = self._load_data()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels)
        ])

    def _load_data(self):
        
        features_df = self.df
        input_data = features_df[self.input_features].reset_index().drop(columns="index")

        if self.to=="inference":
            new_df = pd.DataFrame() 
            for i, c in enumerate(input_data.columns):
                if self.feature_types[i]=="disc":
                    encode_dict = self.input_categories.get(c)
                    input_data[c] = input_data[c].map(encode_dict)
                    one_hot = np.eye(len(encode_dict))[input_data[c].values]
                    sorted_keys = sorted(encode_dict, key=encode_dict.get)
                    cat_df = pd.DataFrame(one_hot, columns=sorted_keys)
                    new_df = pd.concat([new_df, cat_df], axis=1)
                else:
                    if self.conf["input_scaling"]=="norm2":
                        input_data.loc[:, c] = input_data.loc[:, c].astype(np.float32)
                        new_df.loc[:, c] = 2 * (input_data.loc[:, c] - self.input_mins[i])/(self.input_maxs[i] - self.input_mins[i]) - 1
                    elif self.conf["input_scaling"]=="norm1":
                        input_data.loc[:, c] = input_data.loc[:, c].astype(np.float32)
                        new_df.loc[:, c] = (input_data.loc[:, c] - self.input_mins[i])/(self.input_maxs[i] - self.input_mins[i])
                    else:
                        input_data.loc[:, c] = input_data.loc[:, c].astype(np.float32)
                        new_df.loc[:, c] = input_data.loc[:, c]

            new_df = new_df[sorted(new_df.columns)]
            new_input_data = new_df.values.astype(np.float32)

            if self.input_image:
                image_paths = features_df['input_image'].values
                return new_input_data, image_paths, None
        
            return new_input_data, None
        
        new_df = pd.DataFrame() 
        for i, c in enumerate(input_data.columns):
            if self.feature_types[i]=="disc":
                encode_dict = self.input_categories.get(c)
                input_data[c] = input_data[c].map(encode_dict)
                one_hot = np.eye(len(encode_dict))[input_data[c].values]
                sorted_keys = sorted(encode_dict, key=encode_dict.get)
                cat_df = pd.DataFrame(one_hot, columns=sorted_keys)
                new_df = pd.concat([new_df, cat_df], axis=1)
            else:
                if self.conf["input_scaling"]=="norm2":
                    input_data.loc[:, c] = input_data.loc[:, c].astype(np.float32)
                    new_df.loc[:, c] = 2 * (input_data.loc[:, c] - self.input_mins[i])/(self.input_maxs[i] - self.input_mins[i]) - 1
                elif self.conf["input_scaling"]=="norm1":
                    input_data.loc[:, c] = input_data.loc[:, c].astype(np.float32)
                    new_df.loc[:, c] = (input_data.loc[:, c] - self.input_mins[i])/(self.input_maxs[i] - self.input_mins[i])
                else:
                    input_data.loc[:, c] = input_data.loc[:, c].astype(np.float32)
                    new_df.loc[:, c] = input_data.loc[:, c]

        new_input_data = new_df.values.astype(np.float32)
        target_data = features_df[self.target_features].values.astype(np.float32)
        
        if self.input_image:
            image_paths = features_df['input_image'].values
            return new_input_data, image_paths, target_data
   
        return new_input_data, target_data
    
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):

        input_feat = torch.tensor(self.input_data[idx])

        if self.image_paths is not None:
            
            raw_image_path = os.path.join(self.conf["raw_image_folder"], self.image_paths[idx])
            if self.to=="inference":
                raw_image_path = os.path.join(self.conf["upload_image_folder"], self.image_paths[idx])
            raw_image = Image.open(raw_image_path).convert("RGB")
            raw_image = self.transform(raw_image)

            if self.target_data is not None:
                target_feat = torch.tensor(self.target_data[idx])
                return input_feat, raw_image, target_feat
            else:
                return input_feat, raw_image
            
        if self.target_data is not None:
            target_feat = torch.tensor(self.target_data[idx])
            return input_feat, target_feat
        else:
            return input_feat


def create_custom_dataset(df, path, to="train", input_image=False):
    conf = load_config(path)   
    val_dl = None
    if to=="inference":
        infer_ds = RegressionDataset(conf, df, to, input_image=input_image)
        infer_dl = DataLoader(infer_ds, batch_size=int(conf["batch_size"]), num_workers=4, pin_memory=True)
        return infer_dl
    elif to=="train":
        if conf["single_shot_validation"]=="on" and float(conf["val_size"]) > 0:
            train_df, val_df = random_split(df, float(conf["val_size"]))
            if conf["model_type"]=="simple_gan":
                val_ds = ImageDataset(conf, val_df, to=to)
                val_dl = DataLoader(val_ds, batch_size=int(conf["batch_size"]), num_workers=4, pin_memory=True, shuffle=True)
                train_ds = ImageDataset(conf, train_df, to=to)
                train_dl = DataLoader(train_ds, batch_size=int(conf["batch_size"]), num_workers=4, pin_memory=True, shuffle=True)
            else:
                val_ds = RegressionDataset(conf, val_df, input_image=input_image)
                val_dl = DataLoader(val_ds, batch_size=int(conf["batch_size"]), num_workers=4, pin_memory=True, shuffle=True)
                train_ds = RegressionDataset(conf, train_df, input_image=input_image)
                train_dl = DataLoader(train_ds, batch_size=int(conf["batch_size"]), num_workers=4, pin_memory=True, shuffle=True)
        else:
            if conf["model_type"]=="simple_gan":
                train_ds = ImageDataset(conf, train_df, to=to)
                train_dl = DataLoader(train_ds, batch_size=int(conf["batch_size"]), num_workers=4, pin_memory=True, shuffle=True)
            else:
                train_ds = RegressionDataset(conf, df, input_image=input_image)
                train_dl = DataLoader(train_ds, batch_size=int(conf["batch_size"]), num_workers=4, pin_memory=True, shuffle=True)
        return train_dl, val_dl
    else:
        if conf["model_type"]=="simple_gan":
            test_ds = ImageDataset(conf, df, to=to)
            test_dl = DataLoader(test_ds, batch_size=int(conf["batch_size"]), num_workers=4, pin_memory=True, shuffle=True)
        else:
            test_ds = RegressionDataset(conf, df, input_image=input_image)
            test_dl = DataLoader(test_ds, batch_size=int(conf["batch_size"]), num_workers=4, pin_memory=True, shuffle=True)
        return test_dl
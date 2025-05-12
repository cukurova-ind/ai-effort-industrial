import os
import shutil
import pandas as pd
import numpy as np
from openpyxl import Workbook
from sklearn.model_selection import train_test_split
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.sessions.models import Session
from django.contrib import messages
from django.utils.http import url_has_allowed_host_and_scheme
from django.views import View
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.db.models import Q
from dataops.models import Experiment, Input, Fabric, Recipe
from django.core.cache import cache
from engine.models import LoggedInUser
from .image_processor import Preprocessor
from .utils import util, discretes, label, objects
from engine.utils import load_config, data_split
from numba import cuda

raw_image_path = os.path.join(settings.MEDIA_ROOT, "data", "raw")
hypo_image_path = os.path.join(settings.MEDIA_ROOT, "data", "hypo")
output_image_path = os.path.join(settings.MEDIA_ROOT, "data", "output")

image_cache_path = os.path.join(settings.MEDIA_ROOT, "modeling", "image", "cache")
csv_all_path = os.path.join(settings.MEDIA_ROOT, "modeling", "csv", "all")
csv_training_path = os.path.join(settings.MEDIA_ROOT, "modeling", "csv", "training")
csv_test_path = os.path.join(settings.MEDIA_ROOT, "modeling", "csv", "test")


def main_page(req):
    if req.user.is_authenticated:
        if cuda.is_available():
            print("GPU is available")
            for i, gpu in enumerate(cuda.gpus):
                with gpu:
                    print(f"GPU {i}: {cuda.get_current_device().name}")
        else:
            print("GPU is not available")
        return render(req, "modeling_main_page.html")
    else:
        return HttpResponseRedirect("/login/?next=/modeling/")

    
def data_settings(req, profile="unknownprofile"):

    if req.user.is_authenticated:
        safe_folder_name = req.user.email.replace("@", "_at_").replace(".", "_dot_")
        safe_profiles = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "profiles")
        files = sorted([f.split(".")[0] for f in os.listdir(safe_profiles) if os.path.isfile(os.path.join(safe_profiles, f))])
        if "unknownprofile" in files:
            os.unlink(os.path.join(safe_profiles, "unknownprofile.yaml"))
        conf = {}
        if profile and profile in files:
            saved_profile_path = os.path.join(safe_profiles, profile+".yaml")
            if os.path.exists(saved_profile_path):
                conf = load_config.load_config(saved_profile_path)
                conf = conf or {}
        else:
            cache.clear()              

        if req.method == "POST":
            postdata = req.POST
            df_input = data_framer(req.user.username)
            n_features = 0
            column_list = []
            input_list, target_list, input_types, input_maxs, input_mins = [], [], [], [], []
            filter_mins, filter_maxs, filter_values = [], [], []
            category_dict = {}
            for p in postdata:
                if p.split("_")[0]=="inp":
                    column_list.append(postdata[p])
                    if postdata[p] in objects:
                        feature_name = "_".join(p.split("_")[1:])
                    elif postdata[p] in discretes:
                        input_list.append(postdata[p])
                        feature_name = "_".join(p.split("_")[1:])
                        categories = df_input[postdata[p]].value_counts().keys()
                        encoder_dict = {category: i for i, category in enumerate(categories)}
                        category_dict[feature_name] = encoder_dict
                        input_types.append("disc")
                        input_maxs.append("0")
                        input_mins.append("0")
                        n_features += len(categories)
                    else:
                        input_list.append(postdata[p])
                        input_types.append("cont")
                        input_maxs.append(str(df_input[postdata[p]].max()))
                        input_mins.append(str(df_input[postdata[p]].min()))
                        n_features += 1
                elif p.split("_")[0]=="out":
                    target_list.append(postdata[p])
                elif p.split("_")[0]=="filter":
                    if p.split("_")[1]=="min":
                        if postdata[p] != "":
                            filter_mins.append({"_".join(p.split("_")[2:]): postdata[p]})
                    if p.split("_")[1]=="max":
                        if postdata[p] != "":
                            filter_maxs.append({"_".join(p.split("_")[2:]): postdata[p]})
                    if p.split("_")[1]=="values":
                        if postdata[p] != "*":
                            filter_values.append({"_".join(p.split("_")[2:]): postdata[p]})

            if len(input_list)==0 or len(target_list)==0:
                return JsonResponse({"err": "en az 1 girdi ve çıktı belirlenmelidir."})
                
            conf["img_width"] = 256
            conf["img_height"] = 256
            conf["n_channels"] = 3
            conf["noise_dim"] = 128
            conf["embed_dim"] = 128
            conf["embed_out_dim"] = 256

            conf["n_features"] = n_features
            conf["column_list"] = column_list
            conf["input_features"] = input_list
            conf["input_feature_types"] = input_types
            conf["input_maxs"] = input_maxs
            conf["input_mins"] = input_mins
            conf["filter_maxs"] = filter_maxs
            conf["filter_mins"] = filter_mins
            conf["filter_values"] = filter_values
            conf["target_features"] = target_list
            conf["input_categories"] = category_dict
            conf["test_size"] = float(postdata.get("test_size", 0.0))
            conf["random_state"] = int(postdata.get("random_state", 0))
            conf["input_scaling"] = postdata.get("input_scaling", "off")
            conf["model_type"] = postdata.get("model", "")
            conf["retrain"] = postdata.get("retrain", "off")
            conf["retraining_model_name"] = postdata.get("saved_model")
            conf["batch_size"] = int(postdata.get("batch_size", 0))
            conf["n_epoch"] = int(postdata.get("n_epoch", 0))
            conf["loss_function"] = postdata.get("loss_function", "mse")
            conf["learning_rate"] = float(postdata.get("learning_rate", 0))
            conf["single_shot_validation"] = postdata.get("single_val", "off")
            conf["val_size"] = float(postdata.get("val_size", 0.0))
            conf["username"] = req.user.username
            conf["checkpoint_path"] = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "checkpoints")
            conf["raw_image_folder"] = os.path.join(settings.MEDIA_ROOT, "data", "raw")
            conf["hypo_image_folder"] = os.path.join(settings.MEDIA_ROOT, "data", "hypo")
            conf["output_image_folder"] = os.path.join(settings.MEDIA_ROOT, "data", "output")
            conf["device"] = "cuda:0"

            conf["image_input_column"] = "input_image" if "input_image" in column_list else None
            conf["image_target_column"] = "output_image" if "output_image" in column_list else None

            df_all = data_framer(req.user.username)
            df_filtered = data_filter(df_all, filter_mins, filter_maxs, filter_values)

            column_list = conf["column_list"]
            input_list = conf["input_features"]
            target_list = conf["target_features"]

            df_features = df_filtered[column_list]
            #df_input = df_filtered[input_list]
            df_target = df_filtered[target_list]
            
            df_dataset = pd.concat([df_features, df_target], axis=1)

            test_df = None
            if float(postdata.get("test_size", 0.0)) > 0:
                train_df, test_df = data_split.random_split(df_dataset, split_ratio=float(postdata.get("test_size", 0.0)), 
                                            rs=int(postdata.get("random_state", 0)))
            else:
                train_df = df_dataset
            cache.set(f"cached_trainset_{req.user.username}", train_df)
            if test_df is not None:
                cache.set(f"cached_testset_{req.user.username}", test_df)

            if int(postdata.get("save")) == 1:
                profile_name = postdata.get("currentProfileName")
                profile_name = "_".join(profile_name.lower().split(" "))
                saved_profile_path = os.path.join(safe_profiles, profile_name + ".yaml")
                load_config.save_config(conf, saved_profile_path)
                return JsonResponse({"status": "stay", "profile": profile_name})
            
            if int(postdata.get("save")) == 0:
                profile_name = postdata.get("currentProfileName")
                if profile_name:
                    profile = "_".join(profile_name.lower().split(" "))
                else:
                    profile = "unknownprofile.yaml"
                    saved_profile_path = os.path.join(safe_profiles, profile)
                    load_config.save_config(conf, saved_profile_path)
                return JsonResponse({"status": "skip", "profile": profile})

        if req.method == "GET":
            
            df_all = data_framer(req.user.username)
            attr_type, d_type = None, None
            columns, column_features, input_features, target_features = [], [], [], []
            filter_mins, filter_maxs, filter_values = [], [], []
            if conf:
                column_features = conf["column_list"]
                input_features = conf["input_features"]
                target_features = conf["target_features"]
                filter_maxs = conf["filter_maxs"]
                filter_mins = conf["filter_mins"]
                filter_values = conf["filter_values"]
            
            df_filtered = data_filter(df_all, filter_mins, filter_maxs, filter_values)
            
            for c in df_filtered.columns:

                d_type = df_filtered[c].dtype
                attr_type = "categorical" if d_type=="object" else "continuous"
                options = df_filtered[c].value_counts().keys() if d_type=="object" else []
                input_checked = True if c in column_features else False
                target_checked = True if c in target_features else False
                min_filter, max_filter, value_filter = "", "", "*"
                for f in filter_mins:
                    if f.get(c):
                        min_filter = f.get(c)
                        break
                for f in filter_maxs:
                    if f.get(c):
                        max_filter = f.get(c)
                        break
                for f in filter_values:
                    if f.get(c):
                        value_filter = f.get(c)
                        break

                columns.append({"column": c,
                                "label": label.get(c, c),
                                "attr_type": attr_type,
                                "options": options,
                                "d_type": d_type,
                                "input_checked": input_checked,
                                "min_filter": min_filter,
                                "max_filter": max_filter,
                                "value_filter": value_filter,
                                "target_checked": target_checked})

            df_filtered = df_filtered.head(10)

            if profile in files:
                current_profile = profile
            else:
                current_profile = None

            context = {
                "columns": columns,
                "rows": df_filtered.values,
                "profiles": files,
                "profile_name": current_profile,
                "conf": conf
            }

            return render(req, "modeling_configuration.html", context)
        else:
            return HttpResponseRedirect("/login/?next=/modeling/dataset/settings/")


def profile_delete(req, profile="unknownprofile"):

    if req.user.is_authenticated:
        if req.method == "GET":
            safe_folder_name = req.user.email.replace("@", "_at_").replace(".", "_dot_")
            safe_profiles = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "profiles")
            files = sorted([f.split(".")[0] for f in os.listdir(safe_profiles) if os.path.isfile(os.path.join(safe_profiles, f))])
            if "unknownprofile" in files:
                os.unlink(os.path.join(safe_profiles, "unknownprofile.yaml"))
            if profile and profile in files:
                saved_profile_path = os.path.join(safe_profiles, profile+".yaml")
                if os.path.exists(saved_profile_path):
                    os.unlink(saved_profile_path)            
            return HttpResponseRedirect("/modeling/dataset/settings/")
    else:
        return HttpResponseRedirect("/login/?next=/modeling/dataset/settings/")
        


def download_df(req, what="train"):

    cache_key = f"cached_{what}set_{req.user.username}"
    df = cache.get(cache_key)

    response = HttpResponse(content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response['Content-Disposition'] = f'attachment; filename="{what}.xlsx"'

    wb = Workbook()
    ws = wb.active
    ws.title = "sheet1"

    ws.append(df.columns.tolist())

    for row in df.itertuples(index=False):
        ws.append(row)

    wb.save(response)
    return response


def data_framer(username):
    cache_key = f"cached_main_{username}"
    df = cache.get(cache_key)
    if df is not None:
        df_input = df
    else:

        qe = Experiment.objects.values()
        df_exp = pd.DataFrame.from_records(qe)
        qi = Input.objects.values()
        df_inp = pd.DataFrame.from_records(qi)
        qf = Fabric.objects.values()
        df_fab = pd.DataFrame.from_records(qf)
        qr = Recipe.objects.values()
        df_rec = pd.DataFrame.from_records(qr)
        df_inp = df_fab.merge(df_inp, right_on="type_id", left_on="id", how="left")
        df_input = df_inp.merge(df_exp, right_on="input_id", left_on="id_y", how="right")
        df_input.drop(columns=["id_x", "id_y"], inplace=True)
        df_input = df_input.merge(df_rec, right_on="id", left_on="recipe_id", how="left")
        df_input.drop(columns=["id_x", "id_y", "input_id"], inplace=True)
        df_input.insert(0, "type_id", df_input.pop("type_id"))
        cache.set(cache_key, df_input, timeout=60*60)

    return df_input

def data_filter(df, filter_mins, filter_maxs, filter_values):
    if len(filter_mins)>0:
        for f in filter_mins:
            for k in f.keys():
                col = k
                val = float(f[k])
                df = df[df[col]>=val]
            
    if len(filter_maxs)>0:
        for f in filter_maxs:
            for k in f.keys():
                col = k
                val = float(f[k])
                df = df[df[col]<=val]
    
    if len(filter_values)>0:
        for f in filter_values:
            for k in f.keys():
                col = k
                val = f[k]
                df = df[df[col]==val]
    return df

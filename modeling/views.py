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
from engine.models import LoggedInUser
from .image_processor import Preprocessor
from .utils import util, discretes, label
from engine.utils import load_config
from django.core.cache import cache

raw_image_path = os.path.join(settings.MEDIA_ROOT, "data", "raw")
hypo_image_path = os.path.join(settings.MEDIA_ROOT, "data", "hypo")
output_image_path = os.path.join(settings.MEDIA_ROOT, "data", "output")

image_cache_path = os.path.join(settings.MEDIA_ROOT, "modeling", "image", "cache")
csv_all_path = os.path.join(settings.MEDIA_ROOT, "modeling", "csv", "all")
csv_training_path = os.path.join(settings.MEDIA_ROOT, "modeling", "csv", "training")
csv_test_path = os.path.join(settings.MEDIA_ROOT, "modeling", "csv", "test")


def main_page(req):
    if req.user.is_authenticated:
        return render(req, "modeling_main_page.html")
    else:
        return HttpResponseRedirect("/login/?next=/modeling/")

    
def data_settings(req, profile="unknownprofile"):

    if req.user.is_authenticated:
        safe_folder_name = req.user.email.replace("@", "_at_").replace(".", "_dot_")
        safe_profiles = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "profiles")
        files = sorted([f.split(".")[0] for f in os.listdir(safe_profiles) if os.path.isfile(os.path.join(safe_profiles, f))])
        conf = {}
        if profile and profile in files:
            saved_profile_path = os.path.join(safe_profiles, profile+".yaml")
            if os.path.exists(saved_profile_path):
                conf = load_config.load_config(saved_profile_path)
                conf = conf or {}              

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

        if req.method == "POST":
            postdata = req.POST
            
            input_list, target_list, input_types, input_maxs, input_mins, categories = [], [], [], [], [], []
            for p in postdata:
                if p.split("_")[0]=="inp":
                    input_list.append(postdata[p])
                    if postdata[p] in discretes:
                        cats = df_input[postdata[p]].value_counts().keys().values
                        categories.append("|".join(cats))
                        input_types.append("disc")
                        input_maxs.append("0")
                        input_mins.append("0")
                    else:
                        categories.append("0")
                        input_types.append("cont")
                        input_maxs.append(str(df_input[postdata[p]].max()))
                        input_mins.append(str(df_input[postdata[p]].min()))
                elif p.split("_")[0]=="out":
                    target_list.append(postdata[p])
            
            df_features = df_input[input_list]
            df_target = df_input[target_list]

            conf["n_features"] = len(input_list)
            conf["input_features"] = ",".join(input_list)
            conf["input_feature_types"] = ",".join(input_types)
            conf["input_maxs"] = ",".join(input_maxs)
            conf["input_mins"] = ",".join(input_mins)
            conf["target_features"] = ",".join(target_list)
            conf["input_categories"] = ",".join(categories)
            conf["test_size"] = float(postdata.get("test_size", 0.0))
            conf["random_state"] = int(postdata.get("random_state", 0))
            conf["input_scaling"] = postdata.get("input_scaling", "off")
            conf["model_type"] = postdata.get("model", "")
            conf["retrain"] = postdata.get("retrain", "off")
            conf["batch_size"] = int(postdata.get("batch_size", 0))
            conf["n_epoch"] = int(postdata.get("n_epoch", 0))
            conf["loss_function"] = postdata.get("loss_function", "mse")
            conf["learning_rate"] = float(postdata.get("learning_rate", 0))
            conf["single_shot_validation"] = postdata.get("single_val", "off")
            conf["val_size"] = float(postdata.get("val_size", 0.0))
            conf["username"] = req.user.username
            conf["checkpoint_path"] = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "checkpoints")
            conf["device"] = "cuda:0"

            df_dataset = pd.concat([df_features, df_target], axis=1)
            cache.set(f"cached_dataset_{req.user.username}", df_dataset)

            # if float(postdata.get("test_size"))>0:
                
            #     n = len(df_dataset)
                
            #     rs = int(postdata.get("random_state", None)) if postdata.get("random_state", None) else None
            #     _, test_ids = train_test_split(range(n), test_size=float(postdata.get("test_size")), random_state=rs)
            #     test_ids = np.sort(test_ids)

            #     train_df = df_dataset[~df_dataset.index.isin(test_ids)]
            #     test_df = df_dataset[df_dataset.index.isin(test_ids)]

            # else:
            #     train_df = df_dataset
            #     test_df = pd.DataFrame()

            # train_path = os.path.join(csv_training_path, conf["input_file_name"])
            # train_df.to_csv(train_path, index=False)
            # test_path = os.path.join(csv_test_path, conf["input_file_name"])
            # test_df.to_csv(test_path, index=False)

            if postdata.get("saveupdate"):
                profile_name = postdata.get("currentProfileName")
                saved_profile_path = os.path.join(safe_profiles, profile_name + ".yaml")
                load_config.save_config(conf, saved_profile_path)
                return JsonResponse({"profile": profile_name})

            saved_profile_path = os.path.join(safe_profiles, "unknownprofile.yaml")
            load_config.save_config(conf, saved_profile_path)    
            return JsonResponse({"profile": "unknownprofile"})
    
        if req.method == "GET":
            
            attr_type, d_type = None, None
            columns, input_features, target_features = [], [], []
            if conf:
                input_features = conf["input_features"].split(",")
                target_features = conf["target_features"].split(",")
            for c in df_input.columns:

                d_type = df_input[c].dtype
                attr_type = "categorical" if len(df_input[c].value_counts())<=50 or d_type=="object" else "continuous"
                input_checked = True if c in input_features else False
                target_checked = True if c in target_features else False

                columns.append({"column": c,
                                "label": label.get(c, c),
                                "attr_type": attr_type,
                                "d_type": d_type,
                                "input_checked": input_checked,
                                "target_checked": target_checked})

            df_input = df_input.head(10)

            if profile in files:
                current_profile = profile
            else:
                current_profile = None

            context = {
                "columns": columns,
                "rows": df_input.values,
                "profiles": files,
                "profile_name": current_profile,
                "conf": conf
            }

            return render(req, "data_process.html", context)
        else:
            return HttpResponseRedirect("/login/?next=/modeling/dataset/settings/")

def training_settings(req):

    conf = dict()
    with open("config.conf") as c:
        for l in c.read().split("\n"):
            e = l.split("=")
            if len(e)==2:
                conf[e[0].strip()] = e[1].strip()
    
    if req.method == "POST":
        data = req.POST
        with open("config.conf") as c:
            for l in c.read().split("\n"):
                e = l.split("=")
                if len(e)==2:
                    conf[e[0].strip()] = e[1].strip()
                    if data.get(e[0].strip()):   
                        conf[e[0].strip()] = data[e[0].strip()]
        
        if not "max_steps" in data.keys():
            conf["max_steps"] = "off"

        conf["target_scaling"] = "off"

        with open("config.conf", "w") as c:
            c.truncate()
            for x in conf:
                c.write(x + " = " + conf[x] + "\n")

        # config_dest = os.path.join(settings.ENG_URL, "config.conf")
        # image_raw_dest = os.path.join(settings.ENG_URL, "data", "raw")
        # image_hypo_dest = os.path.join(settings.ENG_URL, "data", "hypo")
        # image_out_dest = os.path.join(settings.ENG_URL, "data", "output")
        # csv_train_dest = os.path.join(settings.ENG_URL, "data", "train")
        # csv_test_dest = os.path.join(settings.ENG_URL, "data", "test")

        # shutil.copyfile("config.conf", config_dest)
        
        # if os.path.exists(image_raw_dest):
        #     shutil.rmtree(image_raw_dest)
        # shutil.copytree(raw_image_path, image_raw_dest)
        # if os.path.exists(image_hypo_dest):
        #     shutil.rmtree(image_hypo_dest)
        # shutil.copytree(hypo_image_path, image_hypo_dest)
        # if os.path.exists(image_out_dest):
        #     shutil.rmtree(image_out_dest)
        # shutil.copytree(output_image_path, image_out_dest)

        # if os.path.exists(csv_train_dest):
        #     shutil.rmtree(csv_train_dest)
        # shutil.copytree(csv_training_path, csv_train_dest)
        # if os.path.exists(csv_test_dest):
        #     shutil.rmtree(csv_test_dest)
        # shutil.copytree(csv_test_path, csv_test_dest)

        vs = data.get("saved_model")
        mt = data.get("model")
        conf["posted"] = True
        if vs:
            conf["hlink"] = "http://127.0.0.1:5000?model=" + mt + "&version=" + vs 
            conf["version"] = vs
            return HttpResponseRedirect("/engine/?model=" + mt + "&version=" + vs)

        #return render(req, "modeling_configuration.html", conf)
        return HttpResponseRedirect("/engine/")

    if req.method == "GET":
 
        return render(req, "modeling_configuration.html", conf)

def download_train(req):

    conf = dict()
    with open("config.conf") as c:
        for l in c.read().split("\n"):
            e = l.split("=")
            if len(e)==2:
                conf[e[0].strip()] = e[1].strip()

    train_path = os.path.join(csv_training_path, conf["input_file_name"])
    train_df = pd.read_csv(train_path)
    
    response = HttpResponse(content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response['Content-Disposition'] = 'attachment; filename="train.xlsx"'

    wb = Workbook()
    ws = wb.active
    ws.title = "sheet1"

    ws.append(train_df.columns.tolist())

    for row in train_df.itertuples(index=False):
        ws.append(row)

    wb.save(response)
    return response

def download_test(req):
    conf = dict()
    with open("config.conf") as c:
        for l in c.read().split("\n"):
            e = l.split("=")
            if len(e)==2:
                conf[e[0].strip()] = e[1].strip()

    test_path = os.path.join(csv_test_path, conf["input_file_name"])
    test_df = pd.read_csv(test_path)

    response = HttpResponse(content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response['Content-Disposition'] = 'attachment; filename="test.xlsx"'

    wb = Workbook()
    ws = wb.active
    ws.title = "sheet1"

    ws.append(test_df.columns.tolist())

    for row in test_df.itertuples(index=False):
        ws.append(row)

    wb.save(response)
    return response
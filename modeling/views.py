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
from django.http import HttpResponse, HttpResponseRedirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.db.models import Q
from dataops.models import Experiment, Input, Fabric, Recipe
from engine.models import LoggedInUser
from .image_processor import Preprocessor
from .utils import util, discretes, label

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

    
def data_settings(req):

    conf = dict()
    with open("config.conf") as c:
        for l in c.read().split("\n"):
            e = l.split("=")
            if len(e)==2:
                conf[e[0].strip()] = e[1].strip()
    
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
        
        n_features = len(input_list)
        df_features = df_input[input_list]
        df_target = df_input[target_list]

        conf["n_features"] = str(n_features)
        conf["input_features"] = ",".join(input_list)
        conf["input_feature_types"] = ",".join(input_types)
        conf["input_maxs"] = ",".join(input_maxs)
        conf["input_mins"] = ",".join(input_mins)
        conf["target_features"] = ",".join(target_list)
        conf["input_categories"] = ",".join(categories)

        with open("config.conf", "w") as c:
            c.truncate()
            for x in conf:
                c.write(x + " = " + conf[x] + "\n")

        df_dataset = pd.concat([df_features, df_target], axis=1)
        if float(postdata.get("test_size"))>0:
            
            n = len(df_dataset)
            
            rs = int(postdata.get("random_state", None)) if postdata.get("random_state", None) else None
            _, test_ids = train_test_split(range(n), test_size=float(postdata.get("test_size")), random_state=rs)
            test_ids = np.sort(test_ids)

            train_df = df_dataset[~df_dataset.index.isin(test_ids)]
            test_df = df_dataset[df_dataset.index.isin(test_ids)]

        else:
            train_df = df_dataset
            test_df = pd.DataFrame()

        train_path = os.path.join(csv_training_path, conf["input_file_name"])
        train_df.to_csv(train_path, index=False)
        test_path = os.path.join(csv_test_path, conf["input_file_name"])
        test_df.to_csv(test_path, index=False)

        return HttpResponseRedirect("/modeling/dataset/settings/")
    
    if req.method == "GET":

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
        attr_type, d_type = None, None
        columns = []
        for c in df_input.columns:
            if len(df_input[c].value_counts())<=50:
                attr_type = "categorical"
            else:
                attr_type = "continuous"
            d_type = df_input[c].dtype
            if d_type=="object":
                attr_type = "categorical"
            columns.append({"column": c,
                            "label": label.get(c, c),
                            "attr_type": attr_type,
                            "d_type": d_type})

        df_input = df_input.head(10)
        context = {
            "columns": columns,
            "rows": df_input.values
        }

        return render(req, "data_process.html", context)

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

        config_dest = os.path.join(settings.ENG_URL, "config.conf")
        image_raw_dest = os.path.join(settings.ENG_URL, "data", "raw")
        image_hypo_dest = os.path.join(settings.ENG_URL, "data", "hypo")
        image_out_dest = os.path.join(settings.ENG_URL, "data", "output")
        csv_train_dest = os.path.join(settings.ENG_URL, "data", "train")
        csv_test_dest = os.path.join(settings.ENG_URL, "data", "test")

        shutil.copyfile("config.conf", config_dest)
        
        if os.path.exists(image_raw_dest):
            shutil.rmtree(image_raw_dest)
        shutil.copytree(raw_image_path, image_raw_dest)
        if os.path.exists(image_hypo_dest):
            shutil.rmtree(image_hypo_dest)
        shutil.copytree(hypo_image_path, image_hypo_dest)
        if os.path.exists(image_out_dest):
            shutil.rmtree(image_out_dest)
        shutil.copytree(output_image_path, image_out_dest)

        if os.path.exists(csv_train_dest):
            shutil.rmtree(csv_train_dest)
        shutil.copytree(csv_training_path, csv_train_dest)
        if os.path.exists(csv_test_dest):
            shutil.rmtree(csv_test_dest)
        shutil.copytree(csv_test_path, csv_test_dest)


        vs = data.get("saved_model")
        mt = data.get("model")
        conf["posted"] = True
        if vs:
            conf["hlink"] = "http://127.0.0.1:5000?model=" + mt + "&version=" + vs 
            conf["version"] = vs

        return render(req, "modeling_configuration.html", conf)

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
import pandas as pd
import os
import shutil
from django.shortcuts import render
from django.views import View
from django.http import HttpResponse, HttpResponseRedirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.db.models import Q
from dataops.models import Experiment, Input, Fabric, Recipe
from .image_processor import Preprocessor
from .utils import util

raw_image_path = os.path.join(settings.MEDIA_ROOT, "data", "raw")
hypo_image_path = os.path.join(settings.MEDIA_ROOT, "data", "hypo")
output_image_path = os.path.join(settings.MEDIA_ROOT, "data", "output")
image_training_path = os.path.join(settings.MEDIA_ROOT, "modeling", "image", "training")
image_input_path = os.path.join(image_training_path, "input")
image_target_path = os.path.join(image_training_path, "target")
csv_training_path = os.path.join(settings.MEDIA_ROOT, "modeling", "csv", "training")
csv_input_path = os.path.join(csv_training_path, "input")
csv_target_path = os.path.join(csv_training_path, "target")

def main_page(req):

    return render(req, "modeling_main_page.html")

def image_settings(req):
    
    output, raw, hypo, total = 0, 0, 0, 0
    output += Experiment.objects.filter(~Q(output_image_0="")).count()
    output += Experiment.objects.filter(~Q(output_image_1="")).count()
    output += Experiment.objects.filter(~Q(output_image_2="")).count()
    output += Experiment.objects.filter(~Q(output_image_3="")).count()
    output += Experiment.objects.filter(~Q(output_image_4="")).count()
    raw += Input.objects.filter(~Q(raw_image_0="")).count()
    raw += Input.objects.filter(~Q(raw_image_1="")).count()
    raw += Input.objects.filter(~Q(raw_image_2="")).count()
    raw += Input.objects.filter(~Q(raw_image_3="")).count()
    raw += Input.objects.filter(~Q(raw_image_4="")).count()
    hypo += Input.objects.filter(~Q(hypo_image_0="")).count()
    hypo += Input.objects.filter(~Q(hypo_image_1="")).count()
    hypo += Input.objects.filter(~Q(hypo_image_2="")).count()
    hypo += Input.objects.filter(~Q(hypo_image_3="")).count()
    hypo += Input.objects.filter(~Q(hypo_image_4="")).count()
    total = output + raw + hypo

    if req.method == "GET":
        return render(req, "image_process.html", 
                    {"i": {"total": total, "output": output, "raw": raw, "hypo": hypo}})
    
    if req.method == "POST":
        image_extensions = ('.png', '.jpg', '.jpeg')
        folder_path = os.path.join(settings.MEDIA_ROOT, "data")
        save_path = os.path.join(settings.MEDIA_ROOT, "modeling", "images")
        err = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    file_path = os.path.join(root, file)
                    folder_name = file_path.split("/")[-2]
                    save_folder = os.path.join(save_path, folder_name)
                    if not os.path.exists(save_folder):
                        os.mkdir(save_folder)
                    p = Preprocessor(file_path, save_folder)
                    r, f = p.process()
                    if r == 0:
                        print(r, f)
                        err.append(f)

        return render(req, "image_process.html", 
                    {"i": {"total": total, "output": output, "raw": raw, "hypo": hypo, "err": err}})
    
def data_settings(req):
    
    if not os.path.exists(image_training_path):   
        os.makedirs(image_input_path)
        os.makedirs(image_target_path)
    if not os.path.exists(csv_training_path):
        os.makedirs(csv_input_path)
        os.makedirs(csv_target_path)

    image_extensions = ('.png', '.jpg', '.jpeg')

    if req.method == "POST":
        postdata = req.POST
        qs = Experiment.objects.values("id")
        df_input = pd.DataFrame.from_records(qs)
        df_target = pd.DataFrame.from_records(qs)
        for p in postdata:
            if util.get(p):
                if util.get(p) in [e.name for e in Experiment._meta.get_fields()]:
                    qt = Experiment.objects.values(util.get(p))
                    qs = pd.DataFrame.from_records(qt)
                    df_target = df_target.join(qs)
                    
                else:
                    if util.get(p) in [i.name for i in Input._meta.get_fields()]:
                        field = "input__" + str(util.get(p))
                        qi = Experiment.objects.values(field)
                    if util.get(p) in [r.name for r in Recipe._meta.get_fields()]:
                        field = "recipe__" + str(util.get(p))
                        qi = Experiment.objects.values(field)
                    if util.get(p) in [f.name for f in Fabric._meta.get_fields()]:
                        field = "input__type__" + str(util.get(p))
                        qi = Experiment.objects.values(field)

                    qs = pd.DataFrame.from_records(qi)
                    df_input = df_input.join(qs)
        
        df_input["type"] = df_input["id"].apply(lambda x: int(x.split("-")[0]))
        df_input["recipe"] = df_input["id"].apply(lambda x: int(x.split("-")[1]))
        df_input = df_input.sort_values(["type", "recipe"]).drop(columns=["id"])
        new_cols = df_input.columns.tolist()[:-2]
        new_cols.insert(0, df_input.columns.tolist()[-1])
        new_cols.insert(0, df_input.columns.tolist()[-2])
        df_input = df_input[new_cols]
        if not "type" in postdata:
           df_input = df_input.drop(columns="type")
        if not "recipe" in postdata:
            df_input = df_input.drop(columns="recipe")
        if postdata["scaling_type"]=="norm1":
            df_input = (df_input - df_input.min()) / df_input.max()
        if postdata["scaling_type"]=="norm2":
            df_input = 2 * (df_input - df_input.min()) / (df_input.max() - df_input.min()) - 1
        features_path = os.path.join(csv_input_path, "training_features.csv")
        df_input.to_csv(features_path, index=False)

        df_target["type"] = df_target["id"].apply(lambda x: int(x.split("-")[0]))
        df_target["recipe"] = df_target["id"].apply(lambda x: int(x.split("-")[1]))
        df_target = df_target.sort_values(["type", "recipe"]).drop(columns=["id"])
        new_cols = df_target.columns.tolist()[:-2]
        new_cols.insert(0, df_target.columns.tolist()[-1])
        new_cols.insert(0, df_target.columns.tolist()[-2])
        df_target = df_target[new_cols]
        target_path = os.path.join(csv_target_path, "training_target.csv")
        df_target.to_csv(target_path, index=False)

        it = int(postdata["image_type"].split(" ")[-1])
        if "raw_image" in postdata:
            for root, dirs, files in os.walk(image_input_path):
                for f in files:
                    os.unlink(os.path.join(root, f))
            for root, dirs, files in os.walk(raw_image_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        if len(file.split("."))==2:
                            tip = 0
                        if len(file.split("."))==3:
                            tip = int(file.split(".")[1])
                        
                        if tip==it:
                            source_path = os.path.join(root, file)
                            dest_path = os.path.join(image_input_path, file)
                            shutil.copyfile(source_path, dest_path)
                            p = Preprocessor(dest_path, image_input_path)
                            r, f = p.process()
                            os.unlink(dest_path)
                            if r == 0:
                                print(r, f)
        
        if "hypo_image" in postdata:
            for root, dirs, files in os.walk(hypo_image_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        if len(file.split("."))==2:
                            tip = 0
                        if len(file.split("."))==3:
                            tip = int(file.split(".")[1])
                        
                        if tip==it:
                            source_path = os.path.join(root, file)
                            dest_path = os.path.join(image_input_path, file)
                            shutil.copyfile(source_path, dest_path)
                            p = Preprocessor(dest_path, image_input_path)
                            r, f = p.process()
                            os.unlink(dest_path)
                            if r == 0:
                                print(r, f)

        if "hypo_image" in postdata or "raw_image" in postdata:

            for root, dirs, files in os.walk(image_target_path):
                for f in files:
                    os.unlink(os.path.join(root, f))

            for root, dirs, files in os.walk(output_image_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        
                        if len(file.split("."))==2:
                            tip = 0
                        if len(file.split("."))==3:
                            tip = int(file.split(".")[1])
                        
                        if tip==it:
                            image_name = ".".join(file.split(".")[:-1]) 
                            folder_name = root.split("/")[-1]
                            source_path = os.path.join(root, file)
                            dest_path = os.path.join(image_target_path, image_name + folder_name + ".JPG")
                            shutil.copyfile(source_path, dest_path)
                            p = Preprocessor(dest_path, image_target_path)
                            r, f = p.process()
                            os.unlink(dest_path)
                            if r == 0:
                                print(r, f)

        return HttpResponseRedirect("/modeling/dataset/settings/")
    
    if req.method == "GET":
        input_sample, target_sample = [], []
        image_input_num, image_target_num = 0, 0
        _file_path = os.path.join(settings.BASE_DIR, "industrial", "static", "sample")
        for root, dirs, files in os.walk(_file_path):
            for f in files:
                os.unlink(os.path.join(root, f))
        for root, dirs, files in os.walk(image_training_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    folder_name = root.split("/")[-1]
                    if folder_name=="input":
                        image_input_num += 1
                        if image_input_num<10:
                            source_path = os.path.join(root, file)
                            file_path = os.path.join(_file_path, "input", file)
                            shutil.copyfile(source_path, file_path)
                            input_sample.append(file)
                    if folder_name=="target":
                        image_target_num += 1
                        if image_target_num<10:
                            source_path = os.path.join(root, file)
                            file_path = os.path.join(_file_path, "target", file)
                            shutil.copyfile(source_path, file_path)
                            target_sample.append(file)

        df_input, df_target = pd.DataFrame(), pd.DataFrame()
        for root, dirs, files in os.walk(csv_training_path):
            for file in files:
                if file.lower().endswith("csv"):
                    folder_name = root.split("/")[-1]
                    csv_path = os.path.join(root, file)
                    if folder_name=="input":
                        df_input = pd.read_csv(csv_path)
                    if folder_name=="target":
                        df_target = pd.read_csv(csv_path)

        return render(req, "data_process.html",
                       {"iip": "modeling -> image -> training -> input",
                        "iin": image_input_num,
                        "itn": image_target_num,
                        "itp": "modeling -> image -> training -> target",
                        "cip": "modeling -> csv -> training -> input",
                        "ctp": "modeling -> csv -> training -> target",
                        "is": input_sample,
                        "ts": target_sample,
                        "dfi": df_input.head(),
                        "dft": df_target.head()})

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

        with open("config.conf", "w") as c:
            c.truncate()
            for x in conf:
                c.write(x + " = " + conf[x] + "\n")

        config_dest = os.path.join(settings.ENG_URL, "config.conf")
        image_train_dest = os.path.join(settings.ENG_URL, "dataset", "train", "image")
        csv_train_dest = os.path.join(settings.ENG_URL, "dataset", "train", "csv")
        shutil.copyfile("config.conf", config_dest)
        if os.path.exists(image_train_dest):
            shutil.rmtree(image_train_dest)
        shutil.copytree(image_training_path, image_train_dest)
        if os.path.exists(csv_train_dest):
            shutil.rmtree(csv_train_dest)
        shutil.copytree(csv_training_path, csv_train_dest)
        
        return render(req, "modeling_configuration.html", conf)

    if req.method == "GET":
 
        return render(req, "modeling_configuration.html", conf)
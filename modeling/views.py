import pandas as pd
import os
import zipfile
from django.shortcuts import render
from django.views import View
from django.http import HttpResponse, HttpResponseRedirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.db.models import Q
from dataops.models import Experiment, Input
from .image_processor import Preprocessor
from .utils import util

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
    image_training_path = os.path.join(settings.MEDIA_ROOT, "modeling", "image", "training")
    image_input_path = os.path.join(image_training_path, "input")
    image_target_path = os.path.join(image_training_path, "target")
    csv_training_path = os.path.join(settings.MEDIA_ROOT, "modeling", "csv", "training")
    csv_input_path = os.path.join(csv_training_path, "input")
    csv_target_path = os.path.join(csv_training_path, "target")
    if not os.path.exists(image_training_path):   
        os.makedirs(image_input_path)
        os.makedirs(image_target_path)
    if not os.path.exists(csv_training_path):
        os.makedirs(csv_input_path)
        os.makedirs(csv_target_path)

    image_extensions = ('.png', '.jpg', '.jpeg')
    image_input_num, image_target_num = 0, 0
    for root, dirs, files in os.walk(image_training_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                folder_name = root.split("/")[-1]
                if folder_name=="input":
                    image_input_num += 1
                if folder_name=="target":
                    image_target_num += 1 

    if req.method == "POST":
        postdata = req.POST
        print([util.get(p) for p in postdata])
        return render(req, "data_process.html")
    
    if req.method == "GET":
        return render(req, "data_process.html",
                       {"iip": image_input_path,
                        "iin": image_input_num,
                        "itn": image_target_num,
                        "itp": image_target_path,
                        "cip": csv_input_path,
                        "ctp": csv_target_path})
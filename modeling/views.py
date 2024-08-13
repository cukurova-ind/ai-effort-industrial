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

    return render(req, "data_process.html")
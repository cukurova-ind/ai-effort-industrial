import os
import shutil
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseRedirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

def main_page(req):
    if req.user.is_authenticated:
        return render(req, "prompt_main_page.html")
    else:
        return HttpResponseRedirect("/login/?next=/prompting/")

def filtering_page(req, *args, **kwargs):

    whatfor = kwargs.get("for")
    wtf = None
    if whatfor=="test":
        wtf = "Test"
    elif whatfor=="inference":
        wtf = "Tahmin"
    else:
        wtf = None

    if req.method == "POST":
        data = req.POST
        prompt_type = data["prompt_type"]
        return render(req, "filtering_page.html", {"for":wtf, "wtf": whatfor})
    else:
        if wtf:
            return render(req, "filtering_page.html", {"exp":None, "for":wtf, "wtf": whatfor})
        else:
            return HttpResponseRedirect("/prompting/")
    
def selection_change(req):
    if req.method == "POST":
        safe_folder_name = req.user.email.replace("@", "_at_").replace(".", "_dot_")
        saved_models = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "saved_models")
        
        generator_labels = ["Simple Text2Gan"]
        generator_values = ["simple_gan"]
        predictor_labels = ["Multi-layer Perceptron",
                            "CNN-Augmented Mlp",
                            "Machine Learning Models"]
        predictor_values = ["mlp", "cnnmlp", "ml"]
        data = req.POST
        status = None
        profile = None
        type_options, versions = [], []
        prompt_type = data.get("prompt_type")
        model_type = data.get("prompt_model_type")
        model_version = data.get("prompt_model_version")
        load_model_type = data.get("model")
 
        if load_model_type:
            model_path = os.path.join(saved_models, load_model_type)
            if os.path.exists(model_path):
                versions = sorted([f for f in os.listdir(model_path)])
                status = "complete"
        if prompt_type:
            if prompt_type=="generator":
                for l, v in zip(generator_labels, generator_values):
                    opt = {"label": l, "value": v}
                    type_options.append(opt)
            if prompt_type=="predictor":
                for l, v in zip(predictor_labels, predictor_values):
                    opt = {"label": l, "value": v}
                    type_options.append(opt)
            if model_type:
                model_path = os.path.join(saved_models, model_type)
                if os.path.exists(model_path):
                    versions = sorted([f for f in os.listdir(model_path)])
                if model_version:
                    model_path = os.path.join(saved_models, model_type, model_version)
                    for f in os.listdir(model_path):
                        if os.path.isfile(os.path.join(model_path, f)) and f.split(".")[-1]=="yaml":
                            profile = ".".join(f.split(".")[:-1])
                    status = "complete"

        return JsonResponse({"type_options": type_options,
                                "model": model_type,
                                "versions": versions,
                                "profile": profile,
                                "status": status})
    
def model_delete(req):

    if req.user.is_authenticated:
        if req.method == "GET":
            profile_name = req.GET.get("profile_name")
            model_name = req.GET.get("model_name")
            model_type = req.GET.get("model_type")
            whatfor = req.GET.get("path")
            safe_folder_name = req.user.email.replace("@", "_at_").replace(".", "_dot_")
            saved_models = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "saved_models")
            model_path = os.path.join(saved_models, model_type, model_name)
            if os.path.exists(model_path):
                shutil.rmtree(model_path)         
            return HttpResponseRedirect(f"/prompting/filtering/{whatfor}")
    else:
        return HttpResponseRedirect("/login/?next=/prompting/")

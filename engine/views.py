import os
import shutil
import pandas as pd
from openpyxl import Workbook
from django.shortcuts import render
from django.views import View
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.conf import settings
from .utils.load_config import load_config
from .utils.metadata import labels, ct_options


def main_board(req):
    if req.user.is_authenticated:
        profile_name = req.GET.get("profile")
        model_type = "cnn"
        if profile_name:
            safe_folder_name = req.user.email.replace("@", "_at_").replace(".", "_dot_")
            safe_profiles = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "profiles")
            files = sorted([f.split(".")[0] for f in os.listdir(safe_profiles) if os.path.isfile(os.path.join(safe_profiles, f))])
            if str(profile_name) in files or profile_name=="unknownprofile":    
                profile_name = profile_name
                profile_path = os.path.join(safe_profiles, profile_name + ".yaml")
                conf = load_config(profile_path)
                model_type = conf["model_type"]
                retrain = conf["retrain"]
                retraining_model_name = conf.get("retraining_model_name") if retrain else None
            else:
                return HttpResponseRedirect("/modeling/")
        else:
            return HttpResponseRedirect("/modeling/")
        
        return render(req, "trainboard.html", {"profile_name":profile_name, 
                                               "model_type": model_type, 
                                               "retraining_model_name": retraining_model_name})
    else:
        return HttpResponseRedirect("/login/?next=/engine/")

def model_save(req):
    
    if req.method == "POST":
        safe_folder_name = req.user.email.replace("@", "_at_").replace(".", "_dot_")
        safe_profiles = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "profiles")
        checkpoints = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "checkpoints")
        saved_models = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "saved_models")
        files = [f.split(".")[0] for f in os.listdir(safe_profiles) if os.path.isfile(os.path.join(safe_profiles, f))]

        alert = None
        status = "success"
        postdata = req.POST
        model_profile = postdata.get("profile_name")
        model_name = postdata.get("model_name")
        retrain = postdata.get("retrain")
        
        if model_profile and model_name:
            if str(model_profile) in files:
                profile_path = os.path.join(safe_profiles, model_profile + ".yaml")
                conf = load_config(profile_path)
            elif str(model_profile)=="unknownprofile": 
                profile_path = os.path.join(safe_profiles, "unknownprofile.yaml")
                conf = load_config(profile_path)
            else:
                alert = "kayıtlı profil bulunamadı."
                status = "error"
                return JsonResponse({"status": status, "alert": alert})
            
            main_folder = os.path.join(saved_models, str(conf["model_type"]))
            if not os.path.exists(main_folder):
                os.makedirs(main_folder)
            else:
                model_folder_name = "_".join(model_name.lower().split(" "))
                version_folder = os.path.join(main_folder, model_folder_name)
                if not os.path.exists(version_folder) or retrain:
                    os.makedirs(version_folder, exist_ok=True)
                    for c in os.listdir(checkpoints):
                        source = os.path.join(checkpoints, c)
                        dest = os.path.join(version_folder, c)
                        if os.path.isfile(source):
                            shutil.copyfile(source, dest)
                    profile_dest = os.path.join(version_folder, model_profile + ".yaml")
                    shutil.copyfile(profile_path, profile_dest)
                    alert = str(model_folder_name)
                else:
                    status = "error"
                    alert = "böyle bir model mevcuttur."
        else:
            status = "error"
            alert = "bir isim gönderiniz."
    return JsonResponse({"status": status, "alert": alert})


def inference_page(req):
    if req.user.is_authenticated:
        profile_name = req.GET.get("profile_name")
        model_name = req.GET.get("model_name")
        model_type = req.GET.get("model_type")
        if profile_name and model_name and model_type:
            safe_folder_name = req.user.email.replace("@", "_at_").replace(".", "_dot_")
            safe_profiles = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "profiles")
            files = sorted([f.split(".")[0] for f in os.listdir(safe_profiles) if os.path.isfile(os.path.join(safe_profiles, f))])
            if str(profile_name) in files or profile_name=="unknownprofile":    
                profile_name = profile_name
            else:
                return HttpResponseRedirect("/modeling/")
            saved_models = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "saved_models")
            dirs = sorted([d for _, dirs, _ in os.walk(saved_models) for d in dirs])
            if str(model_name) in dirs and str(model_type) in dirs:    
                model_name = model_name
                model_type = model_type
                safe_profile_path = os.path.join(saved_models, model_type, model_name, profile_name + ".yaml")
            else:
                return HttpResponseRedirect("/modeling/")
            
            conf = load_config(safe_profile_path)
            inputs, targets = [], []

            selected_columns = conf["column_list"]
            input_features = conf["input_features"]
            target_features = conf["target_features"]
            feature_types = conf["input_feature_types"]
            maxs = conf["input_maxs"]
            mins = conf["input_mins"]
            cats = conf["input_categories"]

            for i, f in enumerate(input_features):
                new_cat = []
                if cats.get(f):
                    for c in cats.get(f):
                        new_cat.append({"option": ct_options.get(c), "val": c})
                inputs.append({"feature": f,
                                "label": labels.get(f),
                                "d_type": feature_types[i],
                                "max": maxs[i],
                                "min": mins[i],
                                "cats": new_cat
                                })
            if "input_image" in selected_columns:
                inputs.insert(0, {"feature": "input_image"})
            for t in target_features:
                targets.append({"label": labels.get(t), "target": t})
            if "output_image" in target_features:
                targets = [{"label":"Tahmini Çıktı", "target": "output_image"}]
        return render(req, "inferenceboard.html", {"profile_name":profile_name,
                                                    "model_name": model_name,
                                                    "model_type": model_type,
                                                    "inputs": inputs,
                                                    "targets": targets})
    else:
        return HttpResponseRedirect("/login/?next=/engine/")
    
@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('raw_image'):
        image = request.FILES['raw_image']
        safe_folder_name = request.user.email.replace("@", "_at_").replace(".", "_dot_")
        file_path = os.path.join('media/modeling', safe_folder_name, "upload_images", image.name)
        with open(file_path, 'wb+') as f:
            for chunk in image.chunks():
                f.write(chunk)
        return JsonResponse({'file_path': image.name})
    return JsonResponse({'error': 'Invalid request'}, status=400)
    

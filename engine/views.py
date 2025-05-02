import os
import shutil
import pandas as pd
from openpyxl import Workbook
from django.shortcuts import render
from django.views import View
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.contrib.auth.models import User
from django.conf import settings
from .utils import load_config


def main_board(req):
    if req.user.is_authenticated:
        profile_name = req.GET.get("profile")
        if profile_name:
            safe_folder_name = req.user.email.replace("@", "_at_").replace(".", "_dot_")
            safe_profiles = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "profiles")
            files = sorted([f.split(".")[0] for f in os.listdir(safe_profiles) if os.path.isfile(os.path.join(safe_profiles, f))])
            if str(profile_name) in files or profile_name=="unknownprofile":    
                profile_name = profile_name
            else:
                return HttpResponseRedirect("/modeling/")
        else:
            return HttpResponseRedirect("/modeling/")
        
        return render(req, "trainboard.html", {"profile_name":profile_name})
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
 
        if model_profile and model_name:
            if str(model_profile) in files:
                profile_path = os.path.join(safe_profiles, model_profile + ".yaml")
                conf = load_config.load_config(profile_path)
            elif str(model_profile)=="unknownprofile": 
                profile_path = os.path.join(safe_profiles, "unknownprofile.yaml")
                conf = load_config.load_config(profile_path)
            else:
                alert = "kayıtlı profil bulunamadı."
                status = "error"
                return JsonResponse({"status": status, "alert": alert})
            
            main_folder = os.path.join(saved_models, str(conf["model_type"]))
            if not os.path.exists(main_folder):
                os.makedirs(main_folder)
            else:
                version_folder = os.path.join(main_folder, model_name)
                if not os.path.exists(version_folder):
                    os.makedirs(version_folder)
                    for c in os.listdir(checkpoints):
                        source = os.path.join(checkpoints, c)
                        dest = os.path.join(version_folder, c)
                        if os.path.isfile(source):
                            shutil.copyfile(source, dest)
                    profile_dest = os.path.join(version_folder, model_profile + ".yaml")
                    shutil.copyfile(profile_path, profile_dest)
                    alert = str(model_name)
                else:
                    status = "error"
                    alert = "böyle bir model mevcuttur."
        else:
            status = "error"
            alert = "bir isim gönderiniz."
    return JsonResponse({"status": status, "alert": alert})

import os
import shutil
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseRedirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

def main_page(req):
    conf = dict()
    with open("config.conf") as c:
        for l in c.read().split("\n"):
            e = l.split("=")
            if len(e)==2:
                conf[e[0].strip()] = e[1].strip()

    if req.method == "POST":

        data = req.POST
        prompt_type = data["prompt_type"]
        return render(req, "prompt_main_page.html", conf)
    else:
        return render(req, "prompt_main_page.html", {"exp":None})
    
def selection_change(req):
    if req.method == "POST":
        generator_labels = ["DCGAN", "SRGAN", "UnetGan", "Unet++",
                            "Unet++ Conditional", "Unet++ Gan",
                            "Unet++ Gan Conditional"]
        generator_values = ["dcgan", "srgan", "image_cond",
                            "unet_plus", "unet_plus_cond", "unet_plus_gan",
                            "unet_plus_gan_cond"]
        predictor_labels = ["Multi-layer Perceptron",
                            "CNN-Augmented Mlp",
                            "Machine Learning Models"]
        predictor_values = ["mlp", "cnnmlp", "ml"]
        data = req.POST
        status = ""
        type_options, versions = [], []
        prompt_type = data.get("prompt_type")
        model_type = data.get("prompt_model_type")
        model_version = data.get("prompt_model_version")
        load_model = data.get("model")

        if load_model:
            model_path = os.path.join(settings.ENG_URL, "saved_models", load_model)
            if os.path.exists(model_path):
                for mv in os.listdir(model_path):
                    versions.append(mv)
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
                model_path = os.path.join(settings.ENG_URL, "saved_models", model_type)
                if os.path.exists(model_path):
                    for mv in os.listdir(model_path):
                        versions.append(mv)
                if model_version:
                    status = "complete"
        return JsonResponse({"type_options": type_options,
                            "model": model_type,
                            "versions": versions,
                            "status": status})

def generator_model(req):
    p_shot = None
    tmp = os.path.join(settings.MEDIA_ROOT, "prompt/input")
    if req.method == "POST":
        _, old_files = default_storage.listdir(tmp)
        for r in old_files:
            f = os.path.join(tmp, r)
            default_storage.delete(f)

        raw_image = req.FILES.get("raw_image")
        type_number = req.POST.get("type_number")
        step = req.POST.get("step")
        duration = req.POST.get("duration")
        concentration = req.POST.get("concentration")
        if raw_image:
            content = raw_image.content_type
            tmp = os.path.join("prompt", "input", raw_image.name)
            f = default_storage.save(tmp, raw_image)
            input = {"file": os.path.join("media", f),
                     "type": type_number,
                     "types": range(1,51),
                     "step": step,
                     "duration": duration,
                     "concentration": concentration}
            #p_shot = generate_shot(gen_model, input)
            p_shot = "media/prompt/output/predshot7.png"
            input["photo"] = p_shot
            return render(req, "generator_page.html", input)
        else:
            return render(req, "generator_page.html", {"types":range(1,51)})
    else:
        return render(req, "generator_page.html", {"types":range(1,51)})

def predictor_model(req, model=None):
    if model:
        return render(req, "predictor_page.html", {"types":range(1,51), "model": model})
    else:
        return HttpResponseRedirect("/prompting/predictor/gramaj")

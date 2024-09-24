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
        with open("config.conf") as c:
            for l in c.read().split("\n"):
                e = l.split("=")
                if len(e)==2:
                    conf[e[0].strip()] = e[1].strip()
                    if data.get(e[0].strip()):   
                        conf[e[0].strip()] = data[e[0].strip()]

        with open("config.conf", "w") as c:
            c.truncate()
            for x in conf:
                c.write(x + " = " + conf[x] + "\n")

        config_dest = os.path.join(settings.ENG_URL, "config.conf")

        shutil.copyfile("config.conf", config_dest)
        
        return render(req, "prompt_main_page.html", conf)
    else:
        return render(req, "prompt_main_page.html", {"exp":None})

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

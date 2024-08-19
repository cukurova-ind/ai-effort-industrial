import os
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
# import tensorflow as tf
from .apps import gen_model

# def generate_shot(model, input):
#     x = tf.io.read_file(input["file"])
#     x_decode = tf.image.decode_jpeg(x, channels=3)
#     img = tf.image.resize(x_decode, [128, 128])
#     img = (img - 127.5) / 127.5
#     img = tf.expand_dims(img, axis=0)
#     print(model.output_channels)
#     prediction = model.generator([img])

#     p_file = "media/prompt/output/predshot.png"
#     tf.keras.utils.save_img(p_file, prediction[0] * 0.5 + 0.5)
#     return p_file

def main_page(req):
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
            tmp = os.path.join(tmp, raw_image.name)
            f = default_storage.save(tmp, ContentFile(raw_image.read()))
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

import pandas as pd
import os
import zipfile
import xlwt
from django.shortcuts import render
from django.views import View
from django.http import HttpResponse, HttpResponseRedirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.db.models import Q
from .models import Recipe, Input, Experiment, Fabric
from .forms import RecipeForm, FabricForm, ExperimentForm, Folder

def main_page(req):
    r1 = Recipe.objects.count()
    f1 = Fabric.objects.count()
    e1 = Experiment.objects.count()
    g = Experiment.objects.filter(Q(gramaj__isnull=False)).count()
    we = Experiment.objects.filter(Q(tearing_strength_weft__isnull=False)).count()
    wa = Experiment.objects.filter(Q(tearing_strength_warp__isnull=False)).count()
    el = Experiment.objects.filter(Q(elasticity__isnull=False)).count()
    p = Experiment.objects.filter(Q(pot__isnull=False)).count()
    cl = Experiment.objects.filter(Q(cielab_l__isnull=False)).count()
    ca = Experiment.objects.filter(Q(cielab_a__isnull=False)).count()
    cb = Experiment.objects.filter(Q(cielab_b__isnull=False)).count()
    output, raw, hypo, total = 0, 0, 0, 0
    raw_path = os.path.join(settings.MEDIA_ROOT, "data", "raw")
    raw = len([r for r in os.listdir(raw_path) if os.path.isfile(os.path.join(raw_path, r))])
    hypo_path = os.path.join(settings.MEDIA_ROOT, "data", "hypo")
    hypo = len([h for h in os.listdir(hypo_path) if os.path.isfile(os.path.join(hypo_path, h))])
    exp_path = os.path.join(settings.MEDIA_ROOT, "data", "output")
    output = len([e for e in os.listdir(exp_path) if os.path.isfile(os.path.join(exp_path, e))])
    total = output + raw + hypo
    return render(req, "data_main_page.html", 
                  {"r": r1, "f": f1,
                    "e": {"total": e1, "gramaj": g, "weft": we, "warp": wa, "elas": el, "pot": p},
                    "cielab": {"l": cl, "a": ca, "b": cb},
                    "i": {"total": total, "output": output, "raw": raw, "hypo": hypo}})

class Import(View):

    template_name = "import_page.html"
    whats = ["recipe", "fabric", "cielab", "gramaj", "weft", "warp", "elasticity", "potluk", "overall"]

    def get(self, req, *args, **kwargs):
        what = kwargs.get("what")
        if what:
            if what in self.whats:
                context = {"what": what}
                return render(req, self.template_name, context)
            else:
                return HttpResponseRedirect("recipe")
        else:
            context = {"message": "ok"}
            return render(req, self.template_name, context)
    
    def post(self, req, *args, **kwargs):
        what = kwargs.get("what")
        if what:
            if what == "recipe":
                message = self.recipe_import(req)
            elif what == "fabric":
                message = self.fabric_import(req)
            else:
                message = self.result_import(req, imp=what)
                if message=="success":
                    return HttpResponseRedirect("/dataops")
            context = {"message": message, "what": what} 
            return render(req, self.template_name, context)
    
    def recipe_import(self, req):
        if req.FILES:
            myfile = req.FILES["data_file"]
            content = myfile.content_type
            contents = ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]
            if content not in contents:
                message = "error: provide a csv or xlsx file"
            else:
                try:
                    if content == "text/csv":
                        df = pd.read_csv(myfile)
                    else:
                        df = pd.read_excel(myfile)
                    Recipe.objects.all().delete()
                    for d in df.itertuples():
                        recipe = Recipe.objects.create(
                            id=d.id,
                            bleaching=d.bleaching,
                            duration=d.duration, 
                            concentration=d.concentration)           
                        recipe.save()
                    count = Recipe.objects.count()
                    message = "success: " + str(count) + " data imported"
                except Exception as e:
                    message = e   
        else:
            message = "error: provide a csv file"
        return message
    
    def fabric_import(self, req):
        if req.FILES:
            myfile = req.FILES["data_file"]
            content = myfile.content_type
            contents = ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]
            if content not in contents:
                message = "error: provide a csv or xlsx file"
            else:
                try:
                    if content == "text/csv":
                        df = pd.read_csv(myfile)
                    else:
                        df = pd.read_excel(myfile)
                    Fabric.objects.all().delete()
                    for d in df.itertuples():
                        fabric = Fabric.objects.create(
                            id=d.tip,
                            coloring_type=d.boyama_tipi,
                            fabric_elasticity=d.elastikiyet,
                            yarn_number=d.iplik_no_ne,
                            frequency=d.siklik,
                            knitting=d.orgu,
                            onzyd=d.onzyd2)           
                        fabric.save()
                    count = Fabric.objects.count()
                    message = "success: " + str(count) + " data imported"
                except Exception as e:
                    message = e   
        else:
            message = "error: provide a csv file"
        return message
    
    def result_import(self, req, imp=None):
        if req.FILES and req.FILES["data_file"]:
            myfile = req.FILES["data_file"]
            message = ""
            content = myfile.content_type
            contents = ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]
            if content not in contents:
                message = "error: provide a csv or xlsx file"
            else:
                try:
                    if imp=="overall":
                        if content == "text/csv":
                            df = pd.read_csv(myfile)
                        else:
                            df = pd.read_excel(myfile)
                        
                        for d2 in df.itertuples():
                        
                            fabric = Fabric.objects.get(pk=d2.type)
                            inp = {"type": fabric}

                            inp["gramaj_raw"] = d2.gramage_raw
                            inp["gramaj_hypo"] = d2.gramage_prehypo

                            inp["tearing_strength_weft_raw"] = d2.tearing_strength_weft_raw
                            inp["tearing_strength_weft_hypo"] = d2.tearing_strength_weft_prehypo

                            inp["tearing_strength_warp_raw"] = d2.tearing_strength_warp_raw
                            inp["tearing_strength_warp_hypo"] = d2.tearing_strength_warp_prehypo

                            inp["breaking_strength_weft_raw"] = d2.breaking_strength_weft_raw
                            inp["breaking_strength_weft_hypo"] = d2.breaking_strength_weft_prehypo

                            inp["breaking_strength_warp_raw"] = d2.breaking_strength_warp_raw
                            inp["breaking_strength_warp_hypo"] = d2.breaking_strength_warp_prehypo

                            inp["elasticity_raw"] = d2.elasticity_raw
                            inp["elasticity_hypo"] = d2.elasticity_prehypo

                            inp["pot_raw"] = d2.pot_raw
                            inp["pot_hypo"] = d2.pot_prehypo

                            inp["cielab_l_raw"] = d2.cielab_l_raw
                            inp["cielab_a_raw"] = d2.cielab_a_raw
                            inp["cielab_b_raw"] = d2.cielab_b_raw
                            inp["cielab_l_hypo"] = d2.cielab_l_prehypo
                            inp["cielab_a_hypo"] = d2.cielab_a_prehypo
                            inp["cielab_b_hypo"] = d2.cielab_b_prehypo

                            ip, _ = Input.objects.update_or_create(
                                    type=fabric,
                                    defaults=inp)

                            Input.save(ip)

                            oinput = Input.objects.get(type=d2.type)
                            recipe = Recipe.objects.get(id=d2.recipe)
                            
                            exp = {"input": oinput,
                                    "recipe": recipe}

                            exp["gramaj"] = d2.gramage_posthypo
                            exp["tearing_strength_weft"] = d2.tearing_strength_weft_posthypo
                            exp["tearing_strength_warp"] = d2.tearing_strength_warp_posthypo
                            exp["breaking_strength_weft"] = d2.breaking_strength_weft_posthypo
                            exp["breaking_strength_warp"] = d2.breaking_strength_warp_posthypo
                            exp["elasticity"] = d2.elasticity_posthypo
                            exp["pot"] = d2.pot_posthypo
                            exp["cielab_l"] = d2.cielab_l_posthypo
                            exp["cielab_a"] = d2.cielab_a_posthypo
                            exp["cielab_b"] = d2.cielab_b_posthypo

                            ex, _ = Experiment.objects.update_or_create(
                                    pk=str(d2.type) + "-" + str(d2.recipe),
                                    defaults=exp)

                            Experiment.save(ex)

                            message = "success"

                except Exception as e:
                    message = e
        else:
            message = "error: provide a csv file"
        return message
    
class Entry(View):

    types = range(1,51)
    recipe_form = "recipe_form.html"
    fabric_form = "fabric_form.html"
    experiment_form = "experiment_form.html"
    whats = ["recipe", "fabric", "experiment"]

    def get_ids(self):
        r_ids = []
        r1 = Recipe.objects.all().order_by("id")
        if r1:
            for r  in r1:
                r_ids.append((r.id, r.id))
            r_ids.insert(0, (r.id+1, "Yeni Reçete"))
        else:
            r_ids.insert(0, (1, "Yeni Reçete"))
        f_ids = []
        f1 = Fabric.objects.all().order_by("id")
        if f1:
            for f  in f1:
                f_ids.append((f.id, f.id))
            f_ids.insert(0, (f.id+1, "Yeni Kumaş"))
        else:
            f_ids.insert(0, (1, "Yeni Kumaş"))
        return r_ids, f_ids
    
    def get(self, req, *args, **kwargs):
        what = kwargs.get("what")
        if what:
            r, f = self.get_ids()            
            if what == "recipe":
                context = {"what": what, "form": RecipeForm(ids=r)}
                return render(req, self.recipe_form, context)
            elif what == "fabric":
                context = {"what": what, "form": FabricForm(ids=f)}
                return render(req, self.fabric_form, context)
            elif what == "experiment":
                f.remove(f[0])
                r.remove(r[0])
                context = {"what": what, "form": ExperimentForm(ids1=f, ids2=r)}
                return render(req, self.experiment_form, context)
            else:
                return HttpResponseRedirect("recipe")
        else:
            return HttpResponseRedirect("recipe")
        
    def post(self, req, *args, **kwargs):
        what = kwargs.get("what")
        if what:
            r, f = self.get_ids() 
            if what == "recipe":
                form = RecipeForm(data=req.POST, ids=r)
                if form.is_valid():
                    rec = Recipe(form.cleaned_data["recipe_id"],
                                 form.cleaned_data["bleaching"],
                                 form.cleaned_data["duration"],
                                 form.cleaned_data["concentration"])
                    rec.save()
                return HttpResponseRedirect("/dataops")
            elif what == "fabric":
                form = FabricForm(data=req.POST, ids=f)
                if form.is_valid():
                    fab = Fabric(form.cleaned_data["fabric_id"],
                                 form.cleaned_data["coloring_type"],
                                 form.cleaned_data["fabric_elasticity"],
                                 form.cleaned_data["yarn_number"],
                                 form.cleaned_data["frequency"],
                                 form.cleaned_data["knitting"],
                                 form.cleaned_data["onzyd"],)
                    fab.save()
                return HttpResponseRedirect("/dataops")
            else:
                form = ExperimentForm(data=req.POST, ids1=f, ids2=r)
                if form.is_valid():
                    fab = Fabric.objects.get(form.cleaned_data["type_number"])
                    rec = Recipe.objects.get(form.cleaned_data["recipe_id"])
                    exp = Experiment(fab,
                                    rec,
                                    form.cleaned_data["gramaj"],
                                    form.cleaned_data["tearing_strength_weft"],
                                    form.cleaned_data["tearing_strength_warp"],
                                    form.cleaned_data["elasticity"],
                                    form.cleaned_data["pot"],
                                    form.cleaned_data["cielab_l"],
                                    form.cleaned_data["cielab_a"],
                                    form.cleaned_data["cielab_b"])
                    exp.save()
                return HttpResponseRedirect("/dataops")

def image_upload(req):
    form = Folder()
    if req.method == "POST":
        image_extensions = ("png", "jpg", "jpeg", "JPG")
        if req.FILES:
            message = "ok"
            try:
                raw_images = req.FILES.getlist("raw_folder")
                if raw_images:
                    raw_path = os.path.join(settings.MEDIA_ROOT, "data", "raw")
                    for root, _, files in os.walk(raw_path):
                        for f in files:
                            os.unlink(os.path.join(root, f))
                    for file in raw_images:
                        if file.name.lower().endswith(image_extensions):
                            raw_name = os.path.join("data", "raw", file.name)
                            default_storage.save(raw_name, file)

                hypo_images = req.FILES.getlist("hypo_folder")
                if hypo_images:
                    hypo_path = os.path.join(settings.MEDIA_ROOT, "data", "hypo")
                    for root, _, files in os.walk(hypo_path):
                        for f in files:
                            os.unlink(os.path.join(root, f))
                    for file in hypo_images:
                        if file.name.lower().endswith(image_extensions):
                            hypo_name = os.path.join("data", "hypo", file.name)
                            default_storage.save(hypo_name, file)
                exp_images = req.FILES.getlist("exp_folder")
                if exp_images:
                    exp_path = os.path.join(settings.MEDIA_ROOT, "data", "output")
                    for root, _, files in os.walk(exp_path):
                        for f in files:
                            os.unlink(os.path.join(root, f))
                    for file in exp_images:
                        if file.name.lower().endswith(image_extensions):
                            exp_name = os.path.join("data", "output", file.name)
                            default_storage.save(exp_name, file)
                return HttpResponseRedirect("/dataops")
            except Exception as e:
                message = e  
        else:
            message = "error: provide image folders"
        return render(req, "upload_page.html", {"form": form, "message": message})
    else:
        return render(req, "upload_page.html", {"form": form})
    
def download_format(req, *args, **kwargs):
    what = kwargs.get("what")

    response = HttpResponse(content_type="application/ms-excel")
    response['Content-Disposition'] = f'attachment; filename="{what} format.xlsx"'
 
    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet("sheet1")
    row_num = 0
    font_style = xlwt.XFStyle()
    font_style.font.bold = True
    columns = ['Column 1', 'Column 2', 'Column 3', 'Column 4', ]
    
    for col_num in range(len(columns)):
        ws.write(row_num, col_num, columns[col_num], font_style)
    
    font_style = xlwt.XFStyle()
    wb.save(response)
    return response
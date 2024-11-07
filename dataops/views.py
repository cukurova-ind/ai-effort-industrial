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
    return render(req, "data_main_page.html", 
                  {"r": r1, "f": f1,
                    "e": {"total": e1, "gramaj": g, "weft": we, "warp": wa, "elas": el, "pot": p},
                    "cielab": {"l": cl, "a": ca, "b": cb},
                    "i": {"total": total, "output": output, "raw": raw, "hypo": hypo}})

class Import(View):

    template_name = "import_page.html"
    whats = ["recipe", "fabric", "cielab", "gramaj", "weft", "warp", "elasticity", "potluk"]

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
            if content != "text/csv":
                message = "error: provide a csv file"
            else:
                try:
                    df = pd.read_csv(myfile)
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
            if content != "text/csv":
                message = "error: provide a csv file"
            else:
                try:
                    #df = pd.read_excel(myfile)
                    df = pd.read_csv(myfile)
                    Fabric.objects.all().delete()
                    for d in df.itertuples():
                        fabric = Fabric.objects.create(
                            id=d.tip,
                            material=d.malzeme,
                            material_text=d.malzeme_metni, 
                            coloring=d.boyama,
                            coloring_type=d.boyama_tipi,
                            elastan=d.elastanlik,
                            elasticity=d.elastikiyet,
                            composition=d.kompozisyon,
                            yarn_number=d.iplik_no_ne,
                            frequency=d.siklik,
                            knitting=d.orgu,
                            onyzd=d.onzyd2,
                            onyzd_washed=d.onzyd2_washed,
                            product_color=d.mamul_renk,
                            width=d.en)           
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
            content = myfile.content_type
            if content != "text/csv":
                message = "error: provide a csv file"
            else:
                try:
                    if imp=="cielab":
                        df = pd.read_csv(myfile, header=[0,1])

                        df2 = df.iloc[:, :7]
                        df2.columns = ["Tip", "ham_L", "ham_a", "ham_b", "prehypo_L", "prehypo_a", "prehypo_b"]
                        df2["Tip"] = df2["Tip"].str.partition("p")[2].astype(int)
                        df2 = df2.sort_values(["Tip"])
                        df2.fillna(0, inplace=True)
                        df2["ham_L"] = df2["ham_L"].astype(str).str.replace(",", ".").astype(float)
                        df2["ham_a"] = df2["ham_a"].astype(str).str.replace(",", ".").astype(float)
                        df2["ham_b"] = df2["ham_b"].astype(str).str.replace(",", ".").astype(float)
                        df2["prehypo_L"] = df2["prehypo_L"].astype(str).str.replace(",", ".").astype(float)
                        df2["prehypo_a"] = df2["prehypo_a"].astype(str).str.replace(",", ".").astype(float)
                        df2["prehypo_b"] = df2["prehypo_b"].astype(str).str.replace(",", ".").astype(float)

                        df = df.set_index(df.columns[0])
                        df1 = df.iloc[:, 6:]

                        df1 = df1.stack(1).reset_index()
                        df1 = df1.set_index(df1.columns[0])

                        df1 = df1.melt(id_vars=df1.columns[:1], var_name="recipe", value_name='value', ignore_index=False)
                        df1 = df1.rename_axis("Tip").reset_index()
                        df1 = df1.rename(columns={"level_1":"cielab"})
                        df1.loc[df1["recipe"].str.contains("ete"), "recipe"] = df1.loc[df1["recipe"].str.contains("ete"), "recipe"].str.partition("te")[2]
                        df1["Tip"] = df1["Tip"].str.partition("p")[2].astype(int)
                        df1["recipe"] = df1["recipe"].astype(int)
                        df1 = df1.sort_values(["recipe", "Tip"])
                        df1 = df1.set_index(["Tip", "recipe", "cielab"]).unstack(2).reset_index()
                        df1.columns = ["Tip", "recipe", "L", "a", "b"]
                        df1.fillna(0, inplace=True)

                        df1["L"] = df1["L"].astype(str).str.replace(",", ".").astype(float)
                        df1["a"] = df1["a"].astype(str).str.replace(",", ".").astype(float)
                        df1["b"] = df1["b"].astype(str).str.replace(",", ".").astype(float)
                    else:
                        df = pd.read_csv(myfile, header=[0])
                        
                        df1 = df.set_index(df.columns[0]).iloc[:,2:]
                        df2 = df.set_index(df.columns[0]).iloc[:,:2]
                        df1 = df1.melt(var_name='recipe', value_name='value', ignore_index=False)
                        df1 = df1.rename_axis("Tip").reset_index()
                        df2 = df2.rename_axis("Tip").reset_index()
                        df2["Tip"] = df2["Tip"].str.partition("p")[2].astype(int)
                        
                        df1.loc[df1["recipe"].str.contains("ete"), "recipe"] = df1.loc[df1["recipe"].str.contains("ete"), "recipe"].str.partition("te")[2]
                        df1["Tip"] = df1["Tip"].str.partition("p")[2].astype(int)
                        df1["recipe"] = df1["recipe"].astype(int)
                        df1 = df1.sort_values(["recipe", "Tip"])
                        df1.fillna(0, inplace=True)

                    for d2 in df2.itertuples():
                        
                        fabric = Fabric.objects.get(pk=d2.Tip)
                        inp = {"type": fabric}

                        if imp=="gramaj":
                            inp["gramaj_raw"] = d2.ham
                            inp["gramaj_hypo"] = d2.prehypo
                        if imp=="weft":
                            inp["tearing_strength_weft_raw"] = d2.ham
                            inp["tearing_strength_weft_hypo"] = d2.prehypo
                        if imp=="warp":
                            inp["tearing_strength_warp_raw"] = d2.ham
                            inp["tearing_strength_warp_hypo"] = d2.prehypo
                        if imp=="elasticity":
                            inp["elasticity_raw"] = d2.ham
                            inp["elasticity_hypo"] = d2.prehypo
                        if imp=="potluk":
                            inp["pot_raw"] = d2.ham
                            inp["pot_hypo"] = d2.prehypo
                        if imp=="cielab":
                            inp["cielab_l_raw"] = d2.ham_L
                            inp["cielab_a_raw"] = d2.ham_a
                            inp["cielab_b_raw"] = d2.ham_b
                            inp["cielab_l_hypo"] = d2.prehypo_L
                            inp["cielab_a_hypo"] = d2.prehypo_a
                            inp["cielab_b_hypo"] = d2.prehypo_b

                        ip, _ = Input.objects.update_or_create(
                                type=fabric,
                                defaults=inp)

                        Input.save(ip)

                    for d1 in df1.itertuples():
                        
                        oinput = Input.objects.get(type=d1.Tip)
                        recipe = Recipe.objects.get(id=d1.recipe)
                        
                        exp = {"input": oinput,
                               "recipe": recipe}
                        
                        if imp=="gramaj":
                            exp["gramaj"] = d1.value
                        if imp=="weft":
                            exp["tearing_strength_weft"] = d1.value
                        if imp=="warp":
                            exp["tearing_strength_warp"] = d1.value
                        if imp=="elasticity":
                            exp["elasticity"] = d1.value
                        if imp=="potluk":
                            exp["pot"] = d1.value
                        if imp=="cielab":
                            exp["cielab_l"] = d1.L
                            exp["cielab_a"] = d1.a
                            exp["cielab_b"] = d1.b

                        ex, _ = Experiment.objects.update_or_create(
                                pk=str(d1.Tip) + "-" + str(d1.recipe),
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
                                 form.cleaned_data["temperature"],
                                 form.cleaned_data["concentration"])
                    rec.save()
                return HttpResponseRedirect("/dataops")
            elif what == "fabric":
                form = FabricForm(data=req.POST, ids=f)
                if form.is_valid():
                    fab = Fabric(form.cleaned_data["fabric_id"],
                                 form.cleaned_data["material"],
                                 form.cleaned_data["material_text"],
                                 form.cleaned_data["coloring"],
                                 form.cleaned_data["coloring_type"],
                                 form.cleaned_data["elastan"],
                                 form.cleaned_data["elasticity"],
                                 form.cleaned_data["composition"],
                                 form.cleaned_data["yarn_number"],
                                 form.cleaned_data["frequency"],
                                 form.cleaned_data["knitting"],
                                 form.cleaned_data["onyzd"],
                                 form.cleaned_data["onyzd_washed"],
                                 form.cleaned_data["product_color"],
                                 form.cleaned_data["width"],)
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
    if req.method == "POST":
        if req.FILES:
            message = "ok"
            folder_zip = req.FILES["folder"]
            content = folder_zip.content_type
            contents = ["application/zip", "application/x-zip-compressed"]
            if not content in contents:
                message = "error: provide a .zip file"
            else:
                try:
                    with zipfile.ZipFile(folder_zip, 'r') as zip_ref:

                        for i, info in enumerate(zip_ref.infolist()):
                            if i>0 and not info.is_dir():
                                file_type = info.filename.split('.')[-1]
                                file_type = file_type.lower()
                                if file_type in ["png", "jpg", "jpeg"]:
                                       
                                    folder = info.filename.split("/")[-2]
                                    tip = int(info.filename.split("/")[-1].split(".")[0].split("p")[-1])
                                    region = info.filename.split("/")[-1].split(".")[1]
                                    if region=="JPG":
                                        region = 0
                                    else:
                                        region = int(region)

                                    if folder=="ham":
                                        img_file = zip_ref.extract(info.filename, os.path.join(settings.MEDIA_ROOT, "data", "raw")) 
                                        fabric = Fabric.objects.get(pk=tip)
                                        inp = {"raw_image_" + str(region): os.path.join("data", "raw", info.filename)}
                                        ip, _ = Input.objects.update_or_create(
                                                    type=fabric,
                                                    defaults=inp)
                                        Input.save(ip)
                                    if folder=="hypo":
                                        img_file = zip_ref.extract(info.filename, os.path.join(settings.MEDIA_ROOT, "data", "hypo")) 
                                        fabric = Fabric.objects.get(pk=tip)
                                        inp = {"hypo_image_" + str(region): os.path.join("data", "hypo", info.filename)}
                                        ip, _ = Input.objects.update_or_create(
                                                    type=fabric,
                                                    defaults=inp)
                                        Input.save(ip)
                                    if folder.startswith("recete"):
                                        img_file = zip_ref.extract(info.filename, os.path.join(settings.MEDIA_ROOT, "data", "output"))
                                        recipe = folder.split("ete")[-1]
                                        exp = {"output_image_" + str(region): os.path.join("data", "output", info.filename)}
                                        ex, _ = Experiment.objects.update_or_create(
                                                    id=str(tip) + "-" + recipe,
                                                    defaults=exp)
                                        Experiment.save(ex)
                    return HttpResponseRedirect("/dataops")
                except Exception as e:
                    message = e   
        else:
            message = "error: provide a csv file"
        return render(req, "upload_page.html", {"message": message})
    else:
        form = Folder()
        return render(req, "upload_page.html", {"form": form})
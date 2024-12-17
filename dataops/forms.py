from typing import Any, Mapping
from django import forms
from django.forms.renderers import BaseRenderer
from django.forms.utils import ErrorList
from .models import Recipe, Fabric

class RecipeForm(forms.Form):

    def __init__(self, ids, *args, **kwargs):
        super(RecipeForm, self).__init__(*args, **kwargs)
        self.fields['recipe_id'].choices = ids

    recipe_id = forms.ChoiceField(
        widget=forms.Select(
            attrs={
                    "class": "form-select",
                    "name": "recipe_id",
                }), choices=()) 
    
    bleaching = forms.CharField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0,
                "name": "bleaching",
            }))
    
    duration = forms.CharField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0,
                "name": "duration",
            }))
    
    temperature = forms.CharField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0,
                "name": "temperature",
            }))
    
    concentration = forms.CharField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0,
                "name": "concentration",
            }))
    
class FabricForm(forms.Form):

    coloring_types = [
        ("halat", "halat"),
        ("Slasher", "Slasher"),
        ("indigo", "indigo"),
        ("Vat dye", "Vat dye")
    ]

    knittings = [
        ("2/1 Z", "2/1 Z"),
        ("3/1 KIRIK DİMİ", "3/1 KIRIK DİMİ"),
        ("3/1 RHT", "3/1 RHT"),
        ("3/1 Z", "3/1 Z"),
        ("4/1 SATEN 3 ATLA", "4/1 SATEN 3 ATLA"),
    ]

    def __init__(self, ids, *args, **kwargs):
        super(FabricForm, self).__init__(*args, **kwargs)
        self.fields['fabric_id'].choices = ids

    fabric_id = forms.ChoiceField(widget=forms.Select(
        attrs={
                "class": "form-select",
                "name": "id",
            }), choices=())
    
    material = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "type": "text",
                "name": "material",
            }))
    
    material_text = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "type": "text",
                "name": "material_text",
            }))
    
    coloring = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "type": "text",
                "name": "coloring",
            }))
    
    coloring_type = forms.ChoiceField(
        widget=forms.Select(
            attrs={
                    "class": "form-select",
                    "name": "coloring_type",
                }), choices=coloring_types)
        
    fabric_elasticity = forms.IntegerField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0,
                "name": "fabric_elasticity",
            }))
    
    composition = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "type": "text",
                "name": "composition",
            }))
    
    yarn_number = forms.FloatField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0.0,
                "name": "yarn_number",
            }))
    
    frequency = forms.FloatField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0.0,
                "name": "frequency",
            }))
    
    knitting = forms.ChoiceField(widget=forms.Select(
        attrs={
                "class": "form-select",
                "name": "knitting",
            }), choices=knittings)
    
    onzyd = forms.FloatField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0.0,
                "name": "onzyd",
            }))
    
    onyzd_washed = forms.FloatField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0.0,
                "name": "onyzd_washed",
            }))
    
    product_color = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "type": "text",
                "name": "product_color",
            }))
    
    width = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "type": "text",
                "name": "width",
            }))

class ExperimentForm(forms.Form):

    coloring_types = [
        ("halat", "halat"),
        ("Slasher", "Slasher"),
        ("indigo", "indigo"),
        ("Vat dye", "Vat dye")
    ]

    knittings = [
        ("2/1 Z", "2/1 Z"),
        ("3/1 KIRIK DİMİ", "3/1 KIRIK DİMİ"),
        ("3/1 RHT", "3/1 RHT"),
        ("3/1 Z", "3/1 Z"),
        ("4/1 SATEN 3 ATLA", "4/1 SATEN 3 ATLA"),
    ]
    
    def __init__(self, ids1, ids2, *args, **kwargs):
        super(ExperimentForm, self).__init__(*args, **kwargs)
        self.fields['type_number'].choices = ids1
        self.fields['recipe_id'].choices = ids2

    type_number = forms.ChoiceField(widget=forms.Select(
        attrs={
                "class": "form-select",
                "name": "type_number",
            }), choices=())
    
    recipe_id = forms.ChoiceField(widget=forms.Select(
        attrs={
                "class": "form-select",
                "name": "recipe_id",
            }), choices=())
     
    gramaj = forms.IntegerField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0,
                "name": "gramaj",
            }))
    
    tearing_strength_weft = forms.IntegerField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0,
                "name": "tearing_strength_weft",
            }))
    
    tearing_strength_warp = forms.IntegerField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0,
                "name": "tearing_strength_warp",
            }))
    
    breaking_strength_weft = forms.IntegerField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0,
                "name": "breaking_strength_weft",
            }))
    
    breaking_strength_warp = forms.IntegerField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0,
                "name": "breaking_strength_warp",
            }))
    
    elasticity = forms.IntegerField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0,
                "name": "elasticity",
            }))
    
    pot = forms.IntegerField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0,
                "name": "pot",
            }))
    
    cielab_l = forms.FloatField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0.0,
                "name": "cielab_l",
            }))
    
    cielab_a = forms.FloatField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0.0,
                "name": "cielab_a",
            }))
    
    cielab_b = forms.FloatField(
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "type": "number",
                "value": 0.0,
                "name": "cielab_b",
            }))

class Folder(forms.Form):

    raw_folder = forms.FileField(
        required=False,
        widget=forms.FileInput(
            attrs={
                "id": "raw_folder",
                "webkitdirectory": "",
            }))
    
    hypo_folder = forms.FileField(
        required=False,
        widget=forms.FileInput(
            attrs={
                "id": "hypo_folder",
                "webkitdirectory": "",
            }))
    
    exp_folder = forms.FileField(
        required=False,
        widget=forms.FileInput(
            attrs={
                "id": "exp_folder",
                "webkitdirectory": "",
            }))

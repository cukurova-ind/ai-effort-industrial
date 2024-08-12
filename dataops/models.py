from django.db import models
from django.contrib.postgres.fields import ArrayField


class Recipe(models.Model):

    id = models.IntegerField(primary_key=True)
    bleaching = models.IntegerField(null=False, default=1)
    duration = models.IntegerField(null=False, default=5)
    temperature = models.IntegerField(null=False, default=50)
    concentration = models.IntegerField(null=False, default=4000)

    def __str__(self):
        return str(self.id)

class Fabric(models.Model):
    coloring_type = [
        ("halat", "halat"),
        ("Slasher", "Slasher"),
        ("indigo", "indigo"),
        ("Vat dye", "Vat dye")
    ]
    id = models.IntegerField(primary_key=True)
    material = models.CharField(null=True)
    material_text = models.CharField(null=True)
    coloring = models.CharField(null=True)
    coloring_type = models.CharField(null=False, choices=coloring_type)
    elastan = models.CharField(null=True)
    elasticity = models.IntegerField(null=True)
    composition = models.CharField(null=True)
    yarn_number = models.FloatField(null=True)
    frequency = models.FloatField(null=True)
    knitting = models.CharField(null=True)
    onyzd = models.FloatField(null=True)
    onyzd_washed = models.FloatField(null=True)
    product_color = models.CharField(null=True)
    width = models.CharField(null=True)

    def __str__(self):
        return str(self.id)


class Input(models.Model):

    type = models.ForeignKey(Fabric, null=True, on_delete=models.CASCADE)

    raw_image_0 = models.ImageField(upload_to="data/raw/", null=True)
    raw_image_1 = models.ImageField(upload_to="data/raw/", null=True)
    raw_image_2 = models.ImageField(upload_to="data/raw/", null=True)
    raw_image_3 = models.ImageField(upload_to="data/raw/", null=True)
    raw_image_4 = models.ImageField(upload_to="data/raw/", null=True)

    hypo_image_0 = models.ImageField(upload_to="data/hypo/", null=True)
    hypo_image_1 = models.ImageField(upload_to="data/hypo/", null=True)
    hypo_image_2 = models.ImageField(upload_to="data/hypo/", null=True)
    hypo_image_3 = models.ImageField(upload_to="data/hypo/", null=True)
    hypo_image_4 = models.ImageField(upload_to="data/hypo/", null=True)

    gramaj_raw = models.IntegerField(null=True, blank=True)
    gramaj_hypo = models.IntegerField(null=True, blank=True)

    tearing_strength_weft_raw = models.IntegerField(null=True, blank=True)
    tearing_strength_weft_hypo = models.IntegerField(null=True, blank=True)

    tearing_strength_warp_raw = models.IntegerField(null=True, blank=True)
    tearing_strength_warp_hypo = models.IntegerField(null=True, blank=True)

    elasticity_raw = models.IntegerField(null=True, blank=True)
    elasticity_hypo = models.IntegerField(null=True, blank=True)

    pot_raw = models.IntegerField(null=True, blank=True)
    pot_hypo = models.IntegerField(null=True, blank=True)

    cielab_l_raw = models.FloatField(null=True, blank=True)
    cielab_l_hypo = models.FloatField(null=True, blank=True)
    cielab_a_raw = models.FloatField(null=True, blank=True)
    cielab_a_hypo = models.FloatField(null=True, blank=True)
    cielab_b_raw = models.FloatField(null=True, blank=True)
    cielab_b_hypo = models.FloatField(null=True, blank=True)

    def __str__(self):
        return str(self.type)

class Experiment(models.Model):

    id = models.CharField(primary_key=True)

    input = models.ForeignKey(Input, null=True, on_delete=models.CASCADE)    
    recipe = models.ForeignKey(Recipe, null=True, on_delete=models.CASCADE)
    
    output_image_0 = models.ImageField(upload_to="data/output/", null=True)
    output_image_1 = models.ImageField(upload_to="data/output/", null=True)
    output_image_2 = models.ImageField(upload_to="data/output/", null=True)
    output_image_3 = models.ImageField(upload_to="data/output/", null=True)
    output_image_4 = models.ImageField(upload_to="data/output/", null=True)

    gramaj = models.IntegerField(null=True, blank=True)
    tearing_strength_weft = models.IntegerField(null=True, blank=True)
    tearing_strength_warp = models.IntegerField(null=True, blank=True)
    elasticity = models.IntegerField(null=True, blank=True)
    pot = models.IntegerField(null=True, blank=True)
    cielab_l = models.FloatField(null=True, blank=True)
    cielab_a = models.FloatField(null=True, blank=True)
    cielab_b = models.FloatField(null=True, blank=True)

    def __str__(self):
        return str(self.id)
    
    def save(self, *args, **kwargs):
        if not self.id:
            self.id = str(self.input.type) + "-" + str(self.recipe)
        return super().save(*args, **kwargs)
    
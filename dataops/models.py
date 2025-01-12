from django.db import models
from django.contrib.postgres.fields import ArrayField


class Recipe(models.Model):

    id = models.IntegerField(primary_key=True)
    bleaching = models.IntegerField(null=False, default=1)
    duration = models.IntegerField(null=False, default=5)
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
    knitting_type = [
        ("2/1 Z", "2/1 Z"),
        ("3/1 Z", "3/1 Z"),
        ("3/1 RHT", "3/1 RHT"),
        ("4/1 SATEN 3 ATLA", "4/1 SATEN 3 ATLA"),
        ("3/1 KIRIK DİMİ", "3/1 KIRIK DİMİ"),
    ]
    id = models.IntegerField(primary_key=True)
    coloring_type = models.CharField(null=False, choices=coloring_type, max_length=180)
    fabric_elasticity = models.IntegerField(null=True)
    yarn_number = models.FloatField(null=True)
    frequency = models.FloatField(null=True)
    knitting = models.CharField(null=True, choices=knitting_type, max_length=180)
    onzyd = models.FloatField(null=True)

    def __str__(self):
        return str(self.id)


class Input(models.Model):

    type = models.ForeignKey(Fabric, null=True, on_delete=models.CASCADE)

    gramaj_raw = models.IntegerField(null=True, blank=True)
    gramaj_hypo = models.IntegerField(null=True, blank=True)

    tearing_strength_weft_raw = models.IntegerField(null=True, blank=True)
    tearing_strength_weft_hypo = models.IntegerField(null=True, blank=True)

    tearing_strength_warp_raw = models.IntegerField(null=True, blank=True)
    tearing_strength_warp_hypo = models.IntegerField(null=True, blank=True)

    breaking_strength_weft_raw = models.IntegerField(null=True, blank=True)
    breaking_strength_weft_hypo = models.IntegerField(null=True, blank=True)

    breaking_strength_warp_raw = models.IntegerField(null=True, blank=True)
    breaking_strength_warp_hypo = models.IntegerField(null=True, blank=True)

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

    gramaj = models.IntegerField(null=True, blank=True)
    tearing_strength_weft = models.IntegerField(null=True, blank=True)
    tearing_strength_warp = models.IntegerField(null=True, blank=True)
    breaking_strength_weft = models.IntegerField(null=True, blank=True)
    breaking_strength_warp = models.IntegerField(null=True, blank=True)
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
    
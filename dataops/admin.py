from django.contrib import admin
from . import models


admin.site.register(models.Recipe)
admin.site.register(models.Fabric)
admin.site.register(models.Input)
admin.site.register(models.Experiment)

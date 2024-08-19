from django.apps import AppConfig
from .imgtoimg import Img2Img

gen_model = None

class PromptingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'prompting'

    def ready(self):
        global gen_model
        #i2i = Img2Img(image_shape=(128,128,3))
        #i2i.restore_model()
        #gen_model = i2i 

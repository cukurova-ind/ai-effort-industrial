from django.apps import AppConfig

# gen_model = None

class PromptingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'prompting'

    # def ready(self):
    #     global gen_model
    #     i2i = unet_plus_gan.Unet_plus_gan(image_shape=(128,128,3))
    #     #i2i.restore_model()
    #     #gen_model = i2i 

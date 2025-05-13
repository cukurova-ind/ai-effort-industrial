from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [

    path('', views.main_board, name='main_board'),
    path('model-save/', views.model_save, name='model_save'),
    path('inference/', views.inference_page, name='inference_board'),
    path('inference/upload-image/', views.upload_image, name='upload_image'),

]

if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.main_page, name="modeling_main_page"),
    path('image/settings/', views.image_settings, name="image_settings"),
    path('dataset/settings/', views.data_settings, name="data_settings"),
]

if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
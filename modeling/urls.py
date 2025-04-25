from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.main_page, name="modeling_main_page"),
    path('dataset/settings/', views.data_settings, name="data_settings"),
    path('dataset/settings/<str:profile>', views.data_settings, name="data_settings_with_profile"),
    path('dataset/configure/', views.training_settings, name="training_settings"),
    path('dataset/download/train/', views.download_train, name="download_train"),
    path('dataset/download/test/', views.download_test, name="download_test"),
]

if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
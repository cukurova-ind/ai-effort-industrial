from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.main_page, name="prompt_main_page"),
    path('generator/', views.generator_model, name="generator"),
    path('predictor/', views.predictor_model, name="predictor"),
    path('predictor/<str:model>', views.predictor_model, name="predictor"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)

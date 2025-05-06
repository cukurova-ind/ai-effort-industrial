from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.main_page, name="prompt_main_page"),
    path('filtering/<str:for>', views.filtering_page, name="model_filtering"),
    path('selection-change/', views.selection_change, name="selection-change"),
    path('model-delete/', views.model_delete, name='model_delete'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)

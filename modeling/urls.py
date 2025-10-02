from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.main_page, name="modeling_main_page"),
    path('dataset/settings/', views.data_settings, name="data_settings"),
    path('dataset/settings/<str:profile>', views.data_settings, name="data_settings_with_profile"),
    path('profiledelete/<str:profile>', views.profile_delete, name="profile_delete"),
    path('dataset/download/<str:what>', views.download_df, name="dataset_download"),
    path('csv-upload/', views.csv_upload, name="csv_upload"),
    path('clear-csv/', views.clear_csv, name="clear_csv"),
]

if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
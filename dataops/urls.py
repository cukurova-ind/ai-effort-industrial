from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.main_page, name="data_main_page"),
    path('import/file/<str:what>', views.Import.as_view(), name="data_import"),
    path('entry/<str:what>', views.Entry.as_view(), name="data_entry"),
    path('import/image', views.image_upload, name="image_upload"),
    path('format/download/<str:what>', views.download_format, name="download_format"),
    path('drop/<str:what>', views.Drop.as_view(), name="data_drop"),
]

if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
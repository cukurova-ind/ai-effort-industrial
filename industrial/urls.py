from django.contrib import admin
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static
from . import views


urlpatterns = [
    path('admin/', admin.site.urls),
    #path('accounts/', include('django.contrib.auth.urls')),
    path('login/', views.custom_login_view, name="custom_login"),
    path('', views.main, name="main"),
    path('dataops/', include("dataops.urls")),
    path('modeling/', include("modeling.urls")),
    path('prompting/', include("prompting.urls")),   
    path('engine/', include('engine.urls')),

]

if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

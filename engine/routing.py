from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/engine/(?P<user_name>\w+)/$', consumers.EngineConsumer.as_asgi()),
]
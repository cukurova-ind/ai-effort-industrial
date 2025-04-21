import os
import django
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
import engine.routing


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'industrial.settings')
django.setup()

application = ProtocolTypeRouter({
  "http": get_asgi_application(),
  "websocket": AuthMiddlewareStack(
        URLRouter(
            engine.routing.websocket_urlpatterns
        )
    ),
})

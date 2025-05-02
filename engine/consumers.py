import json
import os
import threading
from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer
from django.conf import settings

from .utils.dataset_create import create_custom_dataset
from .model_templates.mlp_regressor import MlpRegressor
from .model_templates import mlp_regressor as m

class EngineConsumer(WebsocketConsumer):
    stop_signal = threading.Event()

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.user_name = None
        self.train_name = None
        self.room = None

    def connect(self):
        self.user_name = self.scope['url_route']['kwargs']['user_name']
        self.train_name = f'train_{self.user_name}'

        self.accept()

        async_to_sync(self.channel_layer.group_add)(
            self.train_name,
            self.channel_name,
        )

    def disconnect(self, close_code):
        self.stop_signal.set()
        async_to_sync(self.channel_layer.group_discard)(
            self.train_name,
            self.channel_name,
        )

    def receive(self, text_data=None, bytes_data=None):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        if message == "stop":
            self.stop_signal.set()
            async_to_sync(self.channel_layer.group_send)(
                self.train_name,
                {
                    "type": "operation_message",
                    "event": "stopped",
                    "message": "training was stopped by the user."
                }
            )
            return
        
        if message.split("-")[0]=="start":
            df, mes = get_cached_dataframe(f"cached_dataset_{self.user_name}")
            email = message.split("-")[3]
            profile = message.split("-")[-1]
            safe_folder_name = email.replace("@", "_at_").replace(".", "_dot_")
            safe_profile_path = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "profiles", profile + ".yaml")
            
            async_to_sync(self.channel_layer.group_send)(
                self.train_name,
                {
                    'type': 'operation_message',
                    'message': mes,
                }
            )

            train_loader, test_loader, val_loader = create_custom_dataset(df, safe_profile_path)
            input_shape = 0
            for i, batch in enumerate(train_loader):
                input_shape = batch[0].shape
                break
            mlpreg = MlpRegressor(input_shape[1])
            async_to_sync(self.channel_layer.group_send)(
                self.train_name,
                {
                    'type': 'operation_message',
                    'message': "model initialized.",
                }
            )
            def train_async():
                m.train(mlpreg, train_loader, val_loader, config_path=safe_profile_path, stop_signal=self.stop_signal)

            self.stop_signal.clear()
            threading.Thread(target=train_async).start()


    def operation_message(self, event):
        self.send(text_data=json.dumps(event))
    
    def train_message(self, event):
        self.send(text_data=json.dumps(event))


from django.core.cache import cache

def get_cached_dataframe(cache_key):
    df = cache.get(cache_key)
    message = "cache hit. cached dataframe used."
    return df, message
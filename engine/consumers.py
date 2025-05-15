import json
import os
import threading
import pandas as pd
from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer
from django.conf import settings
from django.core.cache import cache
from .utils.dataset_create import create_custom_dataset
from .utils.load_config import load_config
from .model_templates.mlp_regressor import MlpRegressor
from .model_templates.cnnmlp_regressor import CnnmlpRegressor
from .model_templates.simple_generator import Generator, Discriminator


class EngineConsumer(WebsocketConsumer):

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.user_name = None
        self.train_name = None
        self.email = None
        self.profile = None
        self.safe_profile_path = None
        self.conf = None
        self.model = None
        self.retraining_name = None
        self.stop_signal = threading.Event()


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

        if text_data_json["message"] == "stop":
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

        if text_data_json["message"] == "test":
            cache_key = f"cached_testset_{self.user_name}"
            df = cache.get(cache_key)
            if df is not None:
                message = "cache hit. cached dataframe used."
            else:
                message = "no cached dataframe"
                return
            async_to_sync(self.channel_layer.group_send)(
                self.train_name,
                {
                    "type": "operation_message",
                    "message": message
                }
            )

            input_image = False
            if self.conf["model_type"]=="mlp":
                from .model_templates import mlp_regressor as m
            elif self.conf["model_type"]=="cnnmlp":
                input_image = True
                from .model_templates import cnnmlp_regressor as m
            elif self.conf["model_type"]=="simple_gan":
                from .model_templates import simple_generator as m

            self.loader = create_custom_dataset(df, self.safe_profile_path, to="test", input_image=input_image)
            async_to_sync(self.channel_layer.group_send)(
                self.train_name,
                {
                    "type": "operation_message",
                    "message": "data loader created."
                },
            )

            if self.conf["model_type"]=="simple_gan":
                clips, rgbd = m.evaluate(self.model[0], self.loader, self.conf["device"])
                clips = round(clips, 2)
                rgbd = round(rgbd, 2)
                async_to_sync(self.channel_layer.group_send)(
                    self.train_name,
                    {
                        "type": "train_message",
                        "event": "test",
                        "test_clip": clips,
                        "test_rgbd": rgbd,
                    }
                )
            else:
                test_mae, test_mse, test_mape = m.evaluate(self.model, self.loader, self.conf["device"])
                mape = round(test_mape, 2)
                mae = round(test_mae, 2)
                mse = round(test_mse, 2)

                async_to_sync(self.channel_layer.group_send)(
                    self.train_name,
                    {
                        "type": "train_message",
                        "event": "test",
                        "test_mae": mae,
                        "test_mse": mse,
                        "test_mape": mape,
                    }
                )

            async_to_sync(self.channel_layer.group_send)(
                self.train_name,
                {
                    "type": "operation_message",
                    "message": "model evaluated.",
                }
            )

            return
        
        if text_data_json["message"]=="reload":
            self.retraining_name = text_data_json["retrainingmodelname"]
            self.email = text_data_json["email"]
            self.profile = text_data_json["profilename"]
            safe_folder_name = self.email.replace("@", "_at_").replace(".", "_dot_")
            self.safe_profile_path = os.path.join(
                settings.MEDIA_ROOT, "modeling", safe_folder_name, "profiles", self.profile + ".yaml")
            self.conf = load_config(self.safe_profile_path)

            async_to_sync(self.channel_layer.group_send)(
                self.train_name,
                {
                    "type": "operation_message",
                    "event": "retrain",
                    "message": f"retraining for {self.retraining_name}."
                }
            )

            return

        if text_data_json["message"]=="startTrain":
            self.email = text_data_json["email"]
            self.profile = text_data_json["profilename"]
            safe_folder_name = self.email.replace("@", "_at_").replace(".", "_dot_")

            self.safe_profile_path = os.path.join(
                settings.MEDIA_ROOT, "modeling", safe_folder_name, "profiles", self.profile + ".yaml")
            self.conf = load_config(self.safe_profile_path)

            cache_key = f"cached_trainset_{self.user_name}"
            df = cache.get(cache_key)
            if df is not None:
                message = "cache hit. cached dataframe used."
            else:
                message = "no cached dataframe"

            async_to_sync(self.channel_layer.group_send)(
                self.train_name,
                {
                    'type': 'operation_message',
                    'message': message,
                }
            )

            input_image = False
            if self.conf["model_type"]=="mlp":
                self.model = MlpRegressor(int(self.conf["n_features"]))
                from .model_templates import mlp_regressor as m
            elif self.conf["model_type"]=="cnnmlp":
                self.model = CnnmlpRegressor(int(self.conf["n_features"]))
                input_image = True
                from .model_templates import cnnmlp_regressor as m
            elif self.conf["model_type"]=="simple_gan":
                self.model = (Generator(self.conf), Discriminator(self.conf)) 
                from .model_templates import simple_generator as m

            async_to_sync(self.channel_layer.group_send)(
                self.train_name,
                {
                    'type': 'operation_message',
                    'message': "model initialized.",
                }
            )

            train_loader, val_loader = create_custom_dataset(df, self.safe_profile_path, input_image=input_image)
            async_to_sync(self.channel_layer.group_send)(
                self.train_name,
                {
                    "type": "operation_message",
                    "message": "data loader created."
                },
            )

            reload = False
            reload_path = None
            if self.retraining_name:
                model_folder = self.conf["model_type"]
                reload_path = os.path.join(
                    settings.MEDIA_ROOT, "modeling", safe_folder_name, "saved_models", model_folder, self.retraining_name)
                reload = True

            def train_async():
                m.train(self.model, train_loader, val_loader, reload=reload, reload_path=reload_path,
                         config_path=self.safe_profile_path, stop_signal=self.stop_signal)

            self.stop_signal.clear()
            threading.Thread(target=train_async).start()

    def operation_message(self, event):
        self.send(text_data=json.dumps(event))
    
    def train_message(self, event):
        self.send(text_data=json.dumps(event))


class InferenceConsumer(WebsocketConsumer):

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.user_name = None
        self.inference_name = None
        self.conf = {}
        self.safe_profile_path = None
        self.model = None
        self.checkpoint_path = None
        self.loader = None

    def connect(self):
        self.user_name = self.scope['url_route']['kwargs']['user_name']
        self.inference_name = f'inference_{self.user_name}'

        self.accept()

        async_to_sync(self.channel_layer.group_add)(
            self.inference_name,
            self.channel_name,
        )

    def disconnect(self, close_code):

        async_to_sync(self.channel_layer.group_discard)(
            self.inference_name,
            self.channel_name,
        )

    def receive(self, text_data=None, bytes_data=None):
        text_data_json = json.loads(text_data)

        if text_data_json["message"]=="data":
            form_dict = text_data_json.get("data", {})
            for k in form_dict.keys():
                if form_dict.get(k) == "":
                    async_to_sync(self.channel_layer.group_send)(
                        self.inference_name,
                        {
                            "type": "operation_message",
                            "message": "error. empty value."
                        },
                    )
                    return
    
            df = pd.DataFrame([form_dict])
            input_image = "input_image" in df.columns
            self.loader = create_custom_dataset(df, self.safe_profile_path, to="inference", input_image=input_image)
            async_to_sync(self.channel_layer.group_send)(
                self.inference_name,
                {
                    "type": "operation_message",
                    "message": "data loader created."
                },
            )

            def infer_async():
                if self.conf["model_type"]=="mlp":
                    from .model_templates import mlp_regressor as m
                    prediction = m.inference(self.model, self.loader, config_path=self.safe_profile_path)
                elif self.conf["model_type"]=="cnnmlp":
                    from .model_templates import cnnmlp_regressor as m
                    prediction = m.inference(self.model, self.loader, config_path=self.safe_profile_path)
                elif self.conf["model_type"]=="simple_gan":
                    from .model_templates import simple_generator as m
                    prediction = m.inference(self.model, self.loader, config_path=self.safe_profile_path)

                parts = self.conf["generated_image_folder"].strip(os.sep).split(os.sep)
                new_dir = os.path.join(*parts[-4:])

                async_to_sync(self.channel_layer.group_send)(
                    self.inference_name,
                    {
                        "type": "inference_message",
                        "event": self.conf["model_type"],
                        "dir": new_dir,
                        "prediction": str(prediction)
                    },
                )

            threading.Thread(target=infer_async).start()
            

        if text_data_json["message"]=="loadModel":
            email = text_data_json["email"]
            profile = text_data_json["profilename"]
            model_folder = text_data_json["modeltype"]
            model_name = text_data_json["modelname"]
            safe_folder_name = email.replace("@", "_at_").replace(".", "_dot_")
            self.safe_profile_path = os.path.join(
                settings.MEDIA_ROOT, "modeling", safe_folder_name, "saved_models", model_folder, model_name, profile + ".yaml")
            self.checkpoint_path = os.path.join(
                settings.MEDIA_ROOT, "modeling", safe_folder_name, "saved_models", model_folder, model_name)
            self.conf = load_config(self.safe_profile_path)

            if self.conf["model_type"]=="mlp":
                self.model = MlpRegressor(int(self.conf["n_features"]))
            elif self.conf["model_type"]=="cnnmlp":
                self.model = CnnmlpRegressor(int(self.conf["n_features"]))
            elif self.conf["model_type"]=="simple_gan":
                self.model = (Generator(self.conf), Discriminator(self.conf)) 

            async_to_sync(self.channel_layer.group_send)(
                self.inference_name,
                {
                    'type': 'operation_message',
                    'message': "model initialized.",
                }
            )

            def load_async():
                if self.conf["model_type"]=="mlp":
                    from .model_templates import mlp_regressor as m
                    self.model, _ = m.load_model(self.model, checkpoint_path=self.checkpoint_path, config_path=self.safe_profile_path)
                elif self.conf["model_type"]=="cnnmlp":
                    from .model_templates import cnnmlp_regressor as m
                    self.model, _ = m.load_model(self.model, checkpoint_path=self.checkpoint_path, config_path=self.safe_profile_path)
                elif self.conf["model_type"]=="simple_gan":
                    from .model_templates import simple_generator as m
                    self.model, _, _ = m.load_model(self.model[0], self.model[1], checkpoint_path=self.checkpoint_path, config_path=self.safe_profile_path)

            threading.Thread(target=load_async).start()
            
    def operation_message(self, event):
        self.send(text_data=json.dumps(event))
    
    def inference_message(self, event):
        self.send(text_data=json.dumps(event))

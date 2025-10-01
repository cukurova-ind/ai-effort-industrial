import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from ..utils.load_config import load_config

class MlpRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MlpRegressor, self).__init__()

        self.input_dim = input_dim
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        y = self.layers(x)
        return y
        

def calculate_errors(real, pred):
    real = real.float()
    pred = pred.float()
    l1_loss = torch.mean(torch.abs(real - pred))
    l2_loss = torch.mean(torch.square(real - pred))
    mape = torch.mean(torch.abs((real - pred) / (real + 1e-8)) * 100.0)
    return l1_loss, l2_loss, mape
    
def save_checkpoint(model, optimizer, epoch, name, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    checkpoint_path = os.path.join(checkpoint_dir, "latest_weights.pt")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        return None
    #checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    return latest_checkpoint

def load_model(model, to="inference", checkpoint_path=None, config_path=None):
    start_epoch = 0
    
    conf = load_config(config_path) 
    device = torch.device(conf["device"])

    if not checkpoint_path:
        print("no saved model")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        model_path = get_latest_checkpoint(checkpoint_path)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

        if to=="retrain":
            learning_rate = float(conf["learning_rate"])
            optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"]

            channel_layer = get_channel_layer()
            channel_name = f"train_{conf["username"]}"
            if channel_name:
                async_to_sync(channel_layer.group_send)(
                        channel_name,
                        {
                            'type': 'operation_message',
                            'message': "model reloaded to train",
                        }
                    )
                
            print(f"Resuming training from epoch {start_epoch}, checkpoint: {checkpoint_path}")
        else:
            model.eval()
            channel_layer = get_channel_layer()
            channel_name = f"inference_{conf["username"]}"
            if channel_name:
                async_to_sync(channel_layer.group_send)(
                        channel_name,
                        {
                            'type': 'operation_message',
                            'message': "model reloaded for inference",
                        }
                    )
            print(f"Model loaded for inference from: {checkpoint_path}")
    else:
        print("No checkpoint found. Starting fresh.")
    
    return model, start_epoch

def evaluate(model, val_loader, device):
    device = torch.device(device)
    model.eval()
    total_mae = 0
    total_mse = 0
    total_mape = 0
    samples = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            
            input_feat = batch[0].to(device)
            target_feat = batch[1].to(device)

            prediction = model(input_feat)
            mae, mse, mape = calculate_errors(target_feat, prediction)

            batch_size = input_feat.size(0)
            total_mae += mae.item() * batch_size
            total_mse += mse.item() * batch_size
            total_mape += mape.item() * batch_size
            samples += batch_size

    avg_mae = total_mae / samples
    avg_mse = total_mse / samples
    avg_mape = total_mape / samples
    
    print(f"Validation results - MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}, MAPE: {avg_mape:.4f}")
    
    model.train()
    return avg_mae, avg_mse, avg_mape
    
def train(model, train_loader, val_loader=None, reload=False, reload_path=None,
          checkpoint_interval=5, val_interval=5, config_path=None,
          name="mlp_regressor", checkpoint_path=None, stop_signal=None):

    conf = load_config(config_path) 
    device = torch.device(conf["device"])
    num_epochs = int(conf["n_epoch"])
    learning_rate = float(conf["learning_rate"])
    loss_function = conf["loss_function"]
    if loss_function == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss_function == "mae":
        loss_fn = torch.nn.L1Loss()
    elif loss_function == "mape":
        loss_fn = MAPELoss()
    channel_layer = get_channel_layer()
    channel_name = f"train_{conf["username"]}"
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    checkpoint_path = conf["checkpoint_path"] if not checkpoint_path else checkpoint_path
    os.makedirs(checkpoint_path, exist_ok=True)
    model.to(device)

    start_epoch = 0
    if reload:
        model, start_epoch = load_model(model, to="retrain", checkpoint_path=reload_path, config_path=config_path)
    
    model.train()

    if channel_name:
        async_to_sync(channel_layer.group_send)(
                channel_name,
                {
                    'type': 'operation_message',
                    'message': "train started.",
                }
            )
    
    for epoch in range(start_epoch, num_epochs + start_epoch):
        if stop_signal and stop_signal.is_set():
            break

        epoch_start_time = time.time()
        running_loss = 0.0
        samples = 0.00001
        
        
        for i, batch in enumerate(train_loader):
            if stop_signal and stop_signal.is_set():
                break

            total_steps = len(train_loader)

            input_feat = batch[0].to(device)
            target_feat = batch[1].to(device)

            optimizer.zero_grad()
            prediction = model(input_feat)
            loss = loss_fn(target_feat, prediction)
            loss.backward()
            optimizer.step()
            batch_size = input_feat.size(0)
            running_loss += loss.item() * batch_size
            samples += batch_size

            if (i + 1) % 10 == 0 or (i + 1) == total_steps:
                step = i + 1
                percent_complete = round((step / total_steps) * 100, 2) 
                loss_value = round(loss.item(),2) if isinstance(loss, torch.Tensor) else round(float(loss),2)

                if channel_name:
                    async_to_sync(channel_layer.group_send)(
                        channel_name,
                        {
                            "type": "train_message",
                            "event": "step_ten",
                            "step": i + 1,
                            "steps": total_steps,
                            "percentage": percent_complete,
                            "loss": loss_value
                        }
                    )

                #print(f"Epoch {epoch+1}/{num_epochs + start_epoch}, Batch {i+1}, "
                #      f"Loss: {running_loss/samples:.4f}")
                
        epoch_loss = round((running_loss / samples), 4)
        epoch_time = time.time() - epoch_start_time
        percentage = round(((epoch + 1) / (num_epochs + start_epoch)) * 100, 2)

        if channel_name:
            async_to_sync(channel_layer.group_send)(
                channel_name,
                {
                    "type": "train_message",
                    "event": "epoch_end",
                    "epoch_loss": epoch_loss,
                    "epoch": epoch + 1,
                    "epochs": num_epochs + start_epoch,
                    "percentage": percentage
                }
            )

        #print(f"Epoch {epoch + 1}/{num_epochs + start_epoch} completed in {epoch_time:.2f}s - "
        #      f"Loss: {epoch_loss:.4f}")
        
        if val_loader:
            if (epoch + 1) % val_interval == 0:
                val_mae, val_mse, val_mape = evaluate(model, val_loader, conf["device"])
                mape = round(val_mape, 2)
                mae = round(val_mae, 2)
                mse = round(val_mse, 2)
                if channel_name:
                    async_to_sync(channel_layer.group_send)(
                        channel_name,
                        {
                            "type": "train_message",
                            "event": "validation",
                            "val_mae": mae,
                            "val_mse": mse,
                            "val_mape": mape,
                        }
                    )

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch+1, name, checkpoint_path)
    
    if channel_name:
        async_to_sync(channel_layer.group_send)(
                channel_name,
                {
                    'type': 'operation_message',
                    'event': 'stopped',
                    'message': "train ended.",
                }
            )
        
    return model
    
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8)))
    

def predict(model, test_loader, device):
    model.eval()
    total_mae = 0
    total_mse = 0
    total_mape = 0
    samples = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            
            input_feat = batch[0].to(device)
            target_feat = batch[1].to(device)

            prediction = model(input_feat)
            mae, mse, mape = calculate_errors(target_feat, prediction)

            batch_size = input_feat.size(0)
            total_mae += mae.item() * batch_size
            total_mse += mse.item() * batch_size
            total_mape += mape.item() * batch_size
            samples += batch_size

    avg_mae = total_mae / samples
    avg_mse = total_mse / samples
    avg_mape = total_mape / samples
    
    print(f"Test results - MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}, MAPE: {avg_mape:.4f}")
    
    model.train()
    return avg_mae, avg_mse, avg_mape

def inference(model, loader, config_path=None):

    conf = load_config(config_path) 
    device = torch.device(conf["device"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_feat = batch[0].to(device)
            prediction = model(input_feat)
        
    return prediction[0].to("cpu").numpy()

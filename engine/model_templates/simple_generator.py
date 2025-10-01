import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import time
import shutil
from datetime import datetime
import os
from PIL import Image
import numpy as np
import clip
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from ..utils.load_config import load_config


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scalar

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        key = self.key(x).view(B, -1, H * W)  # (B, C//8, HW)
        attention = torch.bmm(query, key)  # (B, HW, HW)
        attention = F.softmax(attention, dim=-1)

        value = self.value(x).view(B, -1, H * W)  # (B, C, HW)
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()
        self.channels = int(conf["n_channels"])
        self.noise_dim = int(conf["noise_dim"])
        self.embed_dim = int(conf["embed_dim"])
        self.embed_out_dim = int(conf["embed_out_dim"])
        self.input_dim = int(conf["n_features"])

        self.text_embedding = nn.Sequential(
            nn.Linear(self.input_dim , self.embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.embed_dim, self.embed_out_dim),
            nn.BatchNorm1d(self.embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Linear(self.noise_dim + self.embed_out_dim, 512 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.attn = SelfAttention(64)

        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)

        self.deconv5 = nn.ConvTranspose2d(32, self.channels, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.tanh = nn.Tanh()

    def forward(self, noise, text):
        
        text = self.text_embedding(text)
        text = text.view(text.shape[0], text.shape[1], 1, 1)

        z = torch.cat([text, noise], 1)

        z = self.fc(z.view(z.shape[0], -1))
        z = z.view(z.shape[0], 512, 8, 8)

        z = self.bn1(self.deconv1(z)).relu()
        z = self.bn2(self.deconv2(z)).relu()
        z = self.bn3(self.deconv3(z)).relu()
        z = self.attn(z)
        z = self.bn4(self.deconv4(z)).relu()
        z = self.tanh(self.deconv5(z))

        return z


class Embedding(nn.Module):
    def __init__(self, input_dim, size_in, size_out):
        super(Embedding, self).__init__()
        self.text_embedding = nn.Sequential(
            nn.Linear(input_dim, size_in),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(size_in, size_out),
            nn.BatchNorm1d(size_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, text):
        embed_out = self.text_embedding(text)

        batch_size, channels, height, width = x.size()

        embed_out_resize = embed_out.view(embed_out.size(0), embed_out.size(1), 1, 1)  # Shape: [batch_size, size_out, 1, 1]
        embed_out_resize = embed_out_resize.expand(-1, -1, height, width) 
        out = torch.cat([x, embed_out_resize], 1)

        return out


class Discriminator(nn.Module):
    def __init__(self, conf):
        super(Discriminator, self).__init__()
        self.channels = int(conf["n_channels"])
        self.embed_dim = int(conf["embed_dim"])
        self.embed_out_dim = int(conf["embed_out_dim"])
        self.input_dim = int(conf["n_features"])

        self.conv1 = nn.Conv2d(self.channels, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.attn = SelfAttention(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.text_embedding = Embedding(self.input_dim, self.embed_dim, self.embed_out_dim)

        self.output = nn.Conv2d(512 + self.embed_out_dim, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, text):
        
        x_out = self.relu1(self.bn1(self.conv1(x)))
        x_out = self.relu2(self.bn2(self.conv2(x_out)))
        x_out = self.attn(x_out)
        x_out = self.relu3(self.bn3(self.conv3(x_out)))
        x_out = self.relu4(self.bn4(self.conv4(x_out)))

        out = self.text_embedding(x_out, text)

        out = self.output(out)
        out = self.sigmoid(out)

        return out.squeeze(), x_out
    

def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = {
        "epoch": epoch,
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "g_optimizer_state": g_optimizer.state_dict(),
        "d_optimizer_state": d_optimizer.state_dict(),
    }

    checkpoint_path = os.path.join(checkpoint_dir, "latest_weights.pt")
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def get_clip_embeddings(image, model, preprocess, device):
    """ Get CLIP embeddings for a given image. """

    if image.dtype != np.uint8:
        # Scale from [-1, 1] or [0, 1] to [0, 255] if needed
        image = np.clip((image + 1) / 2 * 255.0, 0, 255).astype(np.uint8)

    # Permute from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    image_input = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)  # Prepare the image
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize the embeddings

def calculate_clip_similarity(fake_image, target_image, model, preprocess, device):
    """ Calculate the cosine similarity between fake and target images using CLIP embeddings. """
    fake_features = get_clip_embeddings(fake_image, model, preprocess, device)
    target_features = get_clip_embeddings(target_image, model, preprocess, device)
    
    similarity = (fake_features @ target_features.T).item()  # Cosine similarity
    return similarity
        
def evaluate(generator, val_loader, device, save_samples=0, savedir=None, saveuser=None):
    device = torch.device(device)
    generator.eval()
    
    total_clip_similarity = 0.0
    total_rgb_distance = 0.0
    num_samples = 0

    model, preprocess = clip.load('ViT-B/32', device)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            
            input_feat = batch[0].to(device)
            target_image = batch[2].to(device)
            noise = torch.randn(input_feat.size(0), 128, 1, 1, device=device)
            batch_size = target_image.size(0)
            fake_images = generator(noise, input_feat).detach()

            fake_images_scaled = (fake_images * 127.5 + 127.5).clamp(0, 255).byte()
            target_image_scaled = (target_image * 127.5 + 127.5).clamp(0, 255).byte()

            clip_batch = 0.0
            rgb_distance_batch = 0.0
            for j in range(batch_size):
                clip_batch += calculate_clip_similarity(
                    fake_images[j].cpu().numpy(),
                    target_image[j].cpu().numpy(),
                    model, preprocess, device
                )

                fake_mean = fake_images_scaled[j].float().view(3, -1).mean(dim=1)
                target_mean = target_image_scaled[j].float().view(3, -1).mean(dim=1)
                rgb_distance = torch.norm(fake_mean - target_mean, p=2).item()
                rgb_distance_batch += rgb_distance

            total_clip_similarity += clip_batch
            total_rgb_distance += rgb_distance_batch
            num_samples += batch_size

            if i == 0 and save_samples>0:
                if os.path.exists(savedir):
                    shutil.rmtree(savedir)
                os.makedirs(savedir, exist_ok=True)
                print(savedir)
                
                fake_vis = (fake_images + 1) / 2
                target_vis = (target_image + 1) / 2
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                vutils.save_image(fake_vis[:save_samples], f"{savedir}/prediction_{timestamp}.png", nrow=4, normalize=True)
                vutils.save_image(target_vis[:save_samples], f"{savedir}/target_{timestamp}.png", nrow=4, normalize=True)

                parts = savedir.strip(os.sep).split(os.sep)
                new_dir = os.path.join(*parts[-4:])

                channel_layer = get_channel_layer()
                channel_name = f"train_{saveuser}"
                if channel_name:
                    async_to_sync(channel_layer.group_send)(
                            channel_name,
                            {
                                'type': 'operation_message',
                                'event': 'imagesave',
                                'message': "sample images saved.",
                                'savedir': new_dir,
                                'timestamp': timestamp
                            }
                        )
        
    avg_clip_similarity = total_clip_similarity / num_samples
    avg_rgb_distance = total_rgb_distance / num_samples

    print(f"Validation results - CLIP: {avg_clip_similarity:.4f}, RGB_Distance: {avg_rgb_distance:.4f}")

    generator.train()
    return avg_clip_similarity, avg_rgb_distance
    
def train(model, train_loader, val_loader=None, reload=False, reload_path=None,
          checkpoint_interval=5, val_interval=5, config_path=None,
          name="simple_gan", checkpoint_path=None, stop_signal=None):
    
    generator = model[0]
    discriminator = model[1]
    conf = load_config(config_path) 
    device = torch.device(conf["device"])
    num_epochs = int(conf["n_epoch"])
    learning_rate = float(conf["learning_rate"])

    d_losses = []
    g_losses = []
    
    criterion = nn.BCELoss().to(device)
    l2_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    channel_layer = get_channel_layer()
    channel_name = f"train_{conf["username"]}"

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    checkpoint_path = conf["checkpoint_path"] if not checkpoint_path else checkpoint_path
    os.makedirs(checkpoint_path, exist_ok=True)

    generator.to(device)
    discriminator.to(device)

    start_epoch = 0
    if reload:
        generator, discriminator, start_epoch = load_model(generator, discriminator, to="retrain",
                                                           checkpoint_path=reload_path, config_path=config_path)

    if channel_name:
        async_to_sync(channel_layer.group_send)(
                channel_name,
                {
                    'type': 'operation_message',
                    'message': "train started.",
                }
            )
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        if stop_signal and stop_signal.is_set():
            break
        
        epoch_start_time = time.time()
        running_d_loss = 0
        running_g_loss = 0
        samples = 0.00001
        
        for i, batch in enumerate(train_loader):   
            if stop_signal and stop_signal.is_set():
                break
            
            total_steps = len(train_loader)

            input_feat = batch[0].to(device)
            target_images = batch[2].to(device)
            wrong_images = batch[3].to(device)
            
            batch_size = target_images.size(0)

            d_optimizer.zero_grad()
            noise = torch.randn(batch_size, 128, 1, 1, device=device)
            fake_images = generator(noise, input_feat)
            real_out, real_act = discriminator(target_images, input_feat)
            d_loss_real = criterion(real_out, torch.full_like(real_out, 1., device=device))
            wrong_out, wrong_act = discriminator(wrong_images, input_feat)
            d_loss_wrong = criterion(wrong_out, torch.full_like(wrong_out, 0., device=device))
            fake_out, fake_act = discriminator(fake_images.detach(), input_feat)
            d_loss_fake = criterion(fake_out, torch.full_like(fake_out, 0., device=device))
            d_loss = d_loss_real + d_loss_wrong + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            noise = torch.randn(batch_size, 128, 1, 1, device=device)
            fake_images = generator(noise, input_feat)
            out_fake, act_fake = discriminator(fake_images, input_feat)
            out_real, act_real = discriminator(target_images, input_feat)
            g_bce = criterion(out_fake, torch.full_like(out_fake, 1., device=device))
            g_l1 = 50 * l1_loss(target_images, fake_images)
            g_l2 = 150 * l2_loss(torch.mean(act_fake, 0), torch.mean(act_real, 0).detach())
            g_loss = g_bce + g_l1 + g_l2
            g_loss.backward()
            g_optimizer.step()

            running_d_loss += d_loss.item() * batch_size
            running_g_loss += g_loss.item() * batch_size
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            samples += batch_size

            if (i + 1) % 10 == 0 or (i + 1) == total_steps:
                step = i + 1
                percent_complete = round((step / total_steps) * 100, 2)
                d_loss_value = round(d_loss.item(),2) if isinstance(d_loss, torch.Tensor) else round(float(d_loss),2)
                g_loss_value = round(g_loss.item(),2) if isinstance(g_loss, torch.Tensor) else round(float(g_loss),2)

                if channel_name:
                    async_to_sync(channel_layer.group_send)(
                        channel_name,
                        {
                            "type": "train_message",
                            "event": "step_ten",
                            "step": i + 1,
                            "steps": total_steps,
                            "percentage": percent_complete,
                            "loss": {"d_loss": d_loss_value, "g_loss": g_loss_value}
                        }
                    )

                #print(f"Epoch {epoch+1}/{num_epochs + start_epoch}, Batch {i+1}, "
                #      f"Loss_d: {running_d_loss/samples:.4f}, Loss_g: {running_g_loss/samples:.4f}")
                
        epoch_d_loss = round((running_d_loss / samples), 4)
        epoch_g_loss = round((running_g_loss / samples), 4)
        epoch_time = time.time() - epoch_start_time
        percentage = round(((epoch + 1) / (num_epochs + start_epoch)) * 100, 2)

        if channel_name:
            async_to_sync(channel_layer.group_send)(
                channel_name,
                {
                    "type": "train_message",
                    "event": "epoch_end",
                    "epoch_loss": {"d_loss": epoch_d_loss, "g_loss": epoch_g_loss},
                    "epoch": epoch + 1,
                    "epochs": num_epochs + start_epoch,
                    "percentage": percentage
                }
            )
        
        if val_loader:
            if (epoch + 1) % val_interval == 0:
                clips, rgbd = evaluate(generator, val_loader, device, save_samples=8,
                                       savedir=conf["generated_image_folder"], saveuser=conf["username"])
                clips = round(clips, 2)
                rgbd = round(rgbd, 2)
                if channel_name:
                    async_to_sync(channel_layer.group_send)(
                        channel_name,
                        {
                            "type": "train_message",
                            "event": "validation",
                            "clips": clips,
                            "rgbd": rgbd,
                        }
                    )
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch+1, checkpoint_path)
    
    if channel_name:
        async_to_sync(channel_layer.group_send)(
                channel_name,
                {
                    'type': 'operation_message',
                    'event': 'stopped',
                    'message': "train ended.",
                }
            )

def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        return None
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    return latest_checkpoint

def load_model(generator, discriminator, to="inference", checkpoint_path=None, config_path=None):

    start_epoch = 0
    
    conf = load_config(config_path) 
    device = torch.device(conf["device"])

    if not checkpoint_path:
        print("no saved model")

    if os.path.exists(checkpoint_path):
        model_path = get_latest_checkpoint(checkpoint_path)
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint["generator_state"])
        discriminator.load_state_dict(checkpoint["discriminator_state"])

        if to=="retrain":
            learning_rate = float(conf["learning_rate"])
            g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            g_optimizer.load_state_dict(checkpoint["g_optimizer_state"])
            d_optimizer.load_state_dict(checkpoint["d_optimizer_state"])
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
            generator.eval()
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

    return generator, discriminator, start_epoch
    
def inference(generator, loader, config_path=None):
    
    conf = load_config(config_path) 
    device = torch.device(conf["device"])
    generator.to(device)
    generator.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):

            input_feat = batch[0].to(device)
            batch_size = input_feat.size(0)
            noise = torch.randn(batch_size, 128, 1, 1, device=device)
            fake_images = generator(noise, input_feat).detach()

            fake_vis = (fake_images + 1) / 2
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fake_vis = F.interpolate(fake_vis, size=(512, 256), mode="bilinear", align_corners=False)
            vutils.save_image(fake_vis[0], f"{conf["generated_image_folder"]}/prediction_{timestamp}.png", normalize=True)

    generator.train()
    return timestamp
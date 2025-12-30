import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from PIL import Image

from utils import xyz_to_latlon


class CLIPModel(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.device = CONFIG['device']
        self.num_classes = CONFIG['classes']
        self.clip_model, transform = clip.load("ViT-L/14@336px", device=self.device, jit=False)
        self.clip_model.float()

        self.train_transform = transform
        self.eval_transform = transform

        self.vision_encoder = self.clip_model.visual
        self.vision_dim = self.vision_encoder.output_dim

        # freeze backbone
        if not CONFIG['backbone_unfrozen']:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Linear(self.vision_dim, self.num_classes)

    def forward(self, images):
        features = self.vision_encoder(images)
        logits = self.classifier(features)
        return logits

    def guess(self, class_centers, config, pil_image: Image.Image):
        device = config["device"]
        transform = self.eval_transform
        image = pil_image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self(image_tensor)
            probs = F.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            pred_conf = probs[0, pred_idx].item()

        x, y, z = class_centers[pred_idx]
        lat, lon = xyz_to_latlon(x, y, z)
        return lat, lon, pred_conf
    
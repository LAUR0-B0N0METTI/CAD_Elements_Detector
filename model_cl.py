import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

# Dataset customizado para o CLIP
class CLIPClassifierDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Abre a imagem e garante que está em RGB (formato do CLIP)
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(self.labels[idx])

# Definição do Modelo
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=26):
        super().__init__()
        self.clip_model = clip_model
        # Congela o backbone do CLIP (Linear Probing)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Cabeça de classificação: 512 do ViT-B/32 para as 26 classes
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, images):
        with torch.no_grad():
            # Extrai características usando o encoder congelado
            feats = self.clip_model.encode_image(images).float()
        
        # Normalização de L2 para manter a escala do CLIP
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return self.classifier(feats)
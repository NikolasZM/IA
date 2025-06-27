
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct


image_size = 48
patch_size = 6
d_model = 64          
num_classes = 7       
num_layers = 1        
hidden_dim = 128      
epochs = 25        
batch_size = 128
lr = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FERBinDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x_data = np.fromfile(x_path, dtype=np.float32).reshape(-1, image_size * image_size)
        self.y_data = np.fromfile(y_path, dtype=np.uint8)


        assert len(self.x_data) == len(self.y_data), "Dimensiones inconsistentes"
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx].reshape(1, image_size, image_size)
        y = self.y_data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

#ViT

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, d_model,
                                     kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.patch_embed(x) 
        x = x.flatten(2)         
        x = x.transpose(1, 2)  
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class SimpleViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = PatchEmbedding()
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, d_model))
        self.encoder_layers = nn.Sequential(*[TransformerEncoderLayer() for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.patch(x)             
        x = x + self.pos_embedding   
        x = self.encoder_layers(x)   
        x = x.transpose(1, 2)         
        x = self.pool(x).squeeze(-1)  
        return self.classifier(x)

#Entrenamiento

def train():
    dataset = FERBinDataset("X_train.bin", "y_train.bin")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleViT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total_acc += (logits.argmax(1) == y).sum().item()

        avg_loss = total_loss / len(dataset)
        avg_acc = total_acc / len(dataset) * 100
        print(f"Época {epoch+1}: pérdida={avg_loss:.4f}, precisión={avg_acc:.2f}%")
    
    torch.save(model.state_dict(), "bitnet_vit.pth")
    print("Modelo entrenado guardado en: bitnet_vit.pth")

    torch.save(model.state_dict(), "bitnet_vit.pth")
    print("Modelo entrenado guardado en: bitnet_vit.pth")
    
    save_model_binaries(model)
    print("Pesos exportados en formato binario para C++")

def save_model_binaries(model, output_dir="model_weights"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, param in model.named_parameters():
        if 'weight' in name:

            if len(param.shape) == 4: 

                flat = param.data.cpu().numpy().reshape(param.shape[0], -1)
            else:
                flat = param.data.cpu().numpy().flatten()
            
            flat.tofile(f"{output_dir}/{name.replace('.', '_')}.bin")
        
        elif 'bias' in name:
            param.data.cpu().numpy().tofile(f"{output_dir}/{name.replace('.', '_')}.bin")
    
    print(f"Pesos exportados a {output_dir}")

if __name__ == "__main__":
    train()

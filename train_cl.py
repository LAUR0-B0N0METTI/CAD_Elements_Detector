import argparse
import os
import torch
import clip
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Configuração para salvar imagens sem precisar de interface gráfica (Essencial para nohup)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Importações locais
from data_utils_cl import load_dataset, NUM_CLASSES, CLASS_NAMES
from model_cl import CLIPClassifier, CLIPClassifierDataset

def parse_args():
    p = argparse.ArgumentParser(description="Treinamento CLIP CAD - Versão 2.0")
    p.add_argument('--version', type=str, default='2.0')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_epochs', type=int, default=60)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--early_stop', action='store_true')
    p.add_argument('--base_path', type=str, required=True)
    return p.parse_args()

def save_confusion_matrix(model, loader, device, save_path, epoch):
    """Gera e salva a matriz de confusão como imagem PNG"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Matriz de Confusão - Época {epoch}')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Organização de diretórios
    model_dir = os.path.expanduser(f"~/Documentos/01_Projetos/clip_cad_detector/model/v{args.version}")
    os.makedirs(model_dir, exist_ok=True)

    print(f"--- Iniciando Treinamento V{args.version} ---")
    print(f"Device: {device} | Batch Size: {args.batch_size} | LR: {args.lr}")

    # 1. Carregar Modelo CLIP e Wrapper
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    model = CLIPClassifier(clip_model, num_classes=NUM_CLASSES).to(device)
    
    # 2. Critério e Otimizador (Focado apenas na camada classifier)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 3. Carregar Dados
    t_paths, t_labels, v_paths, v_labels = load_dataset(args.base_path)
    
    train_loader = DataLoader(CLIPClassifierDataset(t_paths, t_labels, preprocess), 
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(CLIPClassifierDataset(v_paths, v_labels, preprocess), 
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_val_loss = float('inf')
    patience_counter = 0

    # Loop de Épocas
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Época {epoch}/{args.num_epochs}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Validação
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += lbls.size(0)
                correct += (predicted == lbls).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        print(f"\n[Fim da Época {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {accuracy:.2f}%")

        # Lógica de Matriz de Confusão e Checkpoints
        if epoch % args.save_every == 0 or epoch == 1:
            cm_path = os.path.join(model_dir, f"confusion_matrix_ep{epoch}.png")
            save_confusion_matrix(model, val_loader, device, cm_path, epoch)
            print(f"-> Matriz de Confusão salva em: {cm_path}")

        # Salvar melhor modelo (Best Model)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            patience_counter = 0
            print("-> Melhor modelo atualizado!")
        else:
            patience_counter += 1

        # Early Stopping
        if args.early_stop and patience_counter >= args.patience:
            print(f"--- Early Stopping ativado na época {epoch} ---")
            break

    print(f"Treinamento finalizado. Arquivos salvos em {model_dir}")

if __name__ == "__main__":
    train()

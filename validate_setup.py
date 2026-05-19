"""
Script de validação PRÉ-TREINO.
Execute este script ANTES de iniciar o treinamento para garantir que
o ambiente, o dataset e as dependências estão corretos.

    python validate_setup.py --base_path ~/Documentos/01_Projetos/clip_cad_detector/dataset
"""

import argparse, os, sys
import numpy as np

BASE = os.path.expanduser('~/Documentos/01_Projetos/clip_cad_detector/dataset')


def check(ok, msg_ok, msg_fail):
    if ok:
        print(f"  [OK] {msg_ok}")
    else:
        print(f"  [FAIL] {msg_fail}")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_path', default=BASE)
    args = p.parse_args()
    base = os.path.expanduser(args.base_path)

    print("\n" + "=" * 60)
    print("  VALIDAÇÃO DO AMBIENTE DE TREINAMENTO")
    print("=" * 60)

    # ── 1. Dependências ───────────────────────────────────────────────────────
    print("\n[1] Dependências Python")
    for pkg in ['clip', 'torch', 'numpy', 'sklearn', 'PIL', 'tqdm', 'matplotlib']:
        try:
            __import__(pkg if pkg != 'PIL' else 'PIL.Image')
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [FAIL] Pacote não encontrado: {pkg}")
            sys.exit(1)

    import torch
    print(f"\n  PyTorch  : {torch.__version__}")
    print(f"  CUDA     : {'disponível (' + torch.version.cuda + ')' if torch.cuda.is_available() else 'NÃO disponível – treinamento na CPU'}")

    # ── 2. Estrutura do dataset ───────────────────────────────────────────────
    print("\n[2] Estrutura do dataset")
    check(os.path.isdir(base), f"base_path existe: {base}", f"base_path não encontrado: {base}")

    expected_folders = [str(i) for i in range(1, 26)]

    for split in ('train', 'val', 'test'):
        sp = os.path.join(base, split)
        check(os.path.isdir(sp), f"pasta '{split}' existe", f"pasta '{split}' não encontrada: {sp}")
        folders = sorted([f for f in os.listdir(sp) if f.isdigit()], key=int)
        missing = [f for f in expected_folders if f not in folders]
        check(len(missing) == 0,
              f"'{split}': todas as 25 pastas presentes ({len(folders)})",
              f"'{split}': pastas faltando: {missing}")
        # Conta imagens
        total = sum(
            len([img for img in os.listdir(os.path.join(sp, f))
                 if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for f in folders
        )
        print(f"    {split}: {total} imagens totais")

    # ── 3. Carregamento do dataset ────────────────────────────────────────────
    print("\n[3] Carregamento via data_utils_cl")
    from data_utils_cl import load_dataset, NUM_CLASSES, CLASS_NAMES
    tp, tl, vp, vl = load_dataset(base)
    check(len(tp) > 0, f"train carregado: {len(tp)} imagens", "Nenhuma imagem de treino carregada")
    check(len(vp) > 0, f"val   carregado: {len(vp)} imagens", "Nenhuma imagem de val carregada")
    print(f"    Classes de treino presentes: {sorted(np.unique(tl).tolist())}")
    print(f"    Classes de val   presentes : {sorted(np.unique(vl).tolist())}")
    check(len(np.unique(tl)) == 25,
          "25 classes no treino (correto)", f"Esperado 25, encontrado {len(np.unique(tl))}")

    # ── 4. CLIP + Modelo ──────────────────────────────────────────────────────
    print("\n[4] CLIP + CLIPClassifier")
    import clip
    from model_cl import CLIPClassifier, CLIPClassifierDataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load('ViT-B/32', device=device)
    model = CLIPClassifier(clip_model, num_classes=NUM_CLASSES).to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check(trainable_params > 0, f"Parâmetros treináveis: {trainable_params:,}", "Nenhum parâmetro treinável!")
    print(f"    Parâmetros totais    : {total_params:,}")
    print(f"    Parâmetros treináveis: {trainable_params:,} (cabeça linear)")

    # ── 5. Forward pass de teste ──────────────────────────────────────────────
    print("\n[5] Forward pass de teste (1 batch)")
    from torch.utils.data import DataLoader
    ds = CLIPClassifierDataset(tp[:8], tl[:8], preprocess)
    dl = DataLoader(ds, batch_size=8)
    imgs, lbls = next(iter(dl))
    imgs = imgs.to(device)
    model.eval()
    with torch.no_grad():
        out = model(imgs)
    check(out.shape == (8, NUM_CLASSES),
          f"Saída correta: {out.shape}",
          f"Forma inesperada: {out.shape}")

    # ── 6. Classes ────────────────────────────────────────────────────────────
    print("\n[6] Lista de classes")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {i:2d}: {name}")

    print("\n" + "=" * 60)
    print("  TUDO OK — pode iniciar o treinamento!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

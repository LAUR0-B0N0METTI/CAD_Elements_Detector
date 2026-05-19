"""
normalize_dataset.py
────────────────────
Normaliza, balanceia e estrutura o dataset de elementos CAD.

Fluxo:
  1. Lê os PNGs originais em  DATA_SELECTION_DIR/<N>/
  2. Gera 3 rotações de cada original (90°, 180°, 270°) → pool de 4× os originais
  3. Preenche ciclicamente até atingir TRAIN_TARGET + VAL_TARGET + TEST_TARGET
  4. Embaralha todo o pool e distribui:
       • primeiros VAL_TARGET  → dataset/val/<N>/
       • próximos  TEST_TARGET → dataset/test/<N>/
       • restantes             → dataset/train/<N>/
  5. Renomeia cada arquivo com o padrão  <classe>_XXXX.png

Estrutura de pastas esperada antes de rodar:
  clip_cad_detector/
  └── data_selection/
      ├── 1/   (imagens originais selecionadas – single_door)
      ├── 2/   (double_door)
      …
      └── 25/  (stairs)

Estrutura criada/populada pelo script:
  clip_cad_detector/
  └── dataset/
      ├── train/  1/ … 25/  (TRAIN_TARGET = 8000 por classe)
      ├── val/    1/ … 25/  (VAL_TARGET   = 1200 por classe)
      └── test/   1/ … 25/  (TEST_TARGET  = 1200 por classe)
"""

import os
import random
import shutil
from pathlib import Path

from PIL import Image

# ── Configuração de caminhos ───────────────────────────────────────────────────

PROJECT_ROOT       = Path("/home/zeratull/Documentos/01_Projetos/clip_cad_detector")
DATA_SELECTION_DIR = PROJECT_ROOT / "data_selection"
DATASET_DIR        = PROJECT_ROOT / "dataset"

# ── Alvos de arquivos por split ───────────────────────────────────────────────

TRAIN_TARGET = 8000
VAL_TARGET   = 1200
TEST_TARGET  = 1200
TOTAL_TARGET = TRAIN_TARGET + VAL_TARGET + TEST_TARGET  # 10 400

SEED = 42

# ── Mapeamento pasta → nome de classe ─────────────────────────────────────────

CLASS_NAMES = {
    1:  "single_door",
    2:  "double_door",
    3:  "sliding_door",
    4:  "folding_door",
    5:  "window",
    6:  "bay_window",
    7:  "blind_window",
    8:  "opening_symbol",
    9:  "sofa",
    10: "bed",
    11: "chair",
    12: "table",
    13: "tv_cabinet",
    14: "wardrobe",
    15: "gas_stove",
    16: "sink",
    17: "refrigerator",
    18: "airconditioner",
    19: "bath",
    20: "bath_tub",
    21: "washing_machine",
    22: "squat_toilet",
    23: "urinal",
    24: "toilet",
    25: "stairs",
}

SPLITS = {
    "train": TRAIN_TARGET,
    "val":   VAL_TARGET,
    "test":  TEST_TARGET,
}

# ── Utilitários ───────────────────────────────────────────────────────────────

def get_png_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def rotate_image(src: Path, dest: Path, angle: int):
    """Salva uma cópia rotacionada de src em dest. angle ∈ {90, 180, 270}."""
    angle_map = {
        90:  Image.ROTATE_90,
        180: Image.ROTATE_180,
        270: Image.ROTATE_270,
    }
    with Image.open(src) as img:
        img.transpose(angle_map[angle]).save(dest)


def build_augmented_pool(src_folder: Path, work_folder: Path) -> list[Path]:
    """
    Cria um pool em work_folder com os originais + 3 rotações cada.
    Retorna lista de paths dos arquivos gerados.
    """
    originals = get_png_files(src_folder)
    if not originals:
        raise FileNotFoundError(f"Nenhum PNG encontrado em: {src_folder}")

    pool: list[Path] = []

    for i, orig in enumerate(originals):
        # Copia o original
        dest_orig = work_folder / f"orig_{i:05d}.png"
        shutil.copy2(orig, dest_orig)
        pool.append(dest_orig)

        # Três rotações
        for angle in (90, 180, 270):
            dest_rot = work_folder / f"rot{angle}_{i:05d}.png"
            rotate_image(orig, dest_rot, angle)
            pool.append(dest_rot)

    return pool


def fill_pool_to(pool: list[Path], work_folder: Path, target: int) -> list[Path]:
    """
    Copia arquivos do pool de forma cíclica em work_folder
    até o tamanho total do pool atingir `target`.
    """
    rng   = random.Random(SEED)
    extra = target - len(pool)
    if extra <= 0:
        return pool

    shuffled = pool[:]
    rng.shuffle(shuffled)

    new_files: list[Path] = []
    for i in range(extra):
        src  = shuffled[i % len(shuffled)]
        dest = work_folder / f"fill_{i:06d}.png"
        shutil.copy2(src, dest)
        new_files.append(dest)

    return pool + new_files


def copy_and_rename(files: list[Path], dest_folder: Path, class_name: str):
    """Copia files para dest_folder renomeando para <class_name>_XXXX.png."""
    ensure_dir(dest_folder)
    for i, src in enumerate(files, start=1):
        dest = dest_folder / f"{class_name}_{i:04d}.png"
        shutil.copy2(src, dest)


def clear_folder(folder: Path):
    """Remove todos os PNGs de uma pasta (não apaga subpastas)."""
    for f in folder.glob("*.png"):
        f.unlink()

# ── Processamento de uma classe ────────────────────────────────────────────────

def process_class(folder_num: int, class_name: str):
    print(f"\n{'─'*60}")
    print(f"  Pasta {folder_num:2d}  |  {class_name}")

    src_folder = DATA_SELECTION_DIR / str(folder_num)
    if not src_folder.exists():
        print(f"  ⚠  Pasta de origem não encontrada: {src_folder}")
        return False

    originals = get_png_files(src_folder)
    print(f"  Originais encontrados : {len(originals)}")
    if not originals:
        print(f"  ⚠  Nenhum PNG na pasta de origem. Pulando.")
        return False

    # Pasta de trabalho temporária
    work_folder = PROJECT_ROOT / ".tmp_work" / str(folder_num)
    if work_folder.exists():
        shutil.rmtree(work_folder)
    work_folder.mkdir(parents=True)

    try:
        # 1. Gera pool: originais + 3 rotações cada
        print(f"  Gerando rotações... ", end="", flush=True)
        pool = build_augmented_pool(src_folder, work_folder)
        print(f"{len(pool)} arquivos após rotações")

        # 2. Preenche até TOTAL_TARGET
        if len(pool) < TOTAL_TARGET:
            print(f"  Preenchendo até {TOTAL_TARGET}... ", end="", flush=True)
            pool = fill_pool_to(pool, work_folder, TOTAL_TARGET)
            print(f"{len(pool)} arquivos")
        else:
            pool = pool[:TOTAL_TARGET]

        # 3. Embaralha com semente fixa (reprodutibilidade)
        rng = random.Random(SEED)
        rng.shuffle(pool)

        # 4. Fatia para cada split
        val_files   = pool[:VAL_TARGET]
        test_files  = pool[VAL_TARGET: VAL_TARGET + TEST_TARGET]
        train_files = pool[VAL_TARGET + TEST_TARGET:]

        assert len(train_files) == TRAIN_TARGET, \
            f"Esperado {TRAIN_TARGET} para train, obtido {len(train_files)}"

        # 5. Copia + renomeia para os destinos
        for split_name, files in (("val", val_files),
                                   ("test", test_files),
                                   ("train", train_files)):
            dest = DATASET_DIR / split_name / str(folder_num)
            # Limpa o destino antes de popular
            if dest.exists():
                clear_folder(dest)
            print(f"  → {split_name:5s}: {len(files):5d} arquivos  →  {dest}")
            copy_and_rename(files, dest, class_name)

    finally:
        shutil.rmtree(work_folder)  # limpa temporários

    return True

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     NORMALIZAÇÃO E ESTRUTURAÇÃO DO DATASET CAD           ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Origem  : {str(DATA_SELECTION_DIR):<46}║")
    print(f"║  Destino : {str(DATASET_DIR):<46}║")
    print(f"║  train={TRAIN_TARGET}  val={VAL_TARGET}  test={TEST_TARGET}  seed={SEED:<19}║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Cria estrutura de pastas do dataset
    for split in SPLITS:
        for n in range(1, 26):
            ensure_dir(DATASET_DIR / split / str(n))

    errors:  list[str] = []
    success: list[int] = []

    for folder_num in range(1, 26):
        class_name = CLASS_NAMES[folder_num]
        ok = False
        try:
            ok = process_class(folder_num, class_name)
        except Exception as exc:
            msg = f"Pasta {folder_num} ({class_name}): {exc}"
            print(f"\n  ✘ ERRO: {msg}")
            errors.append(msg)
        if ok:
            success.append(folder_num)

    # ── Relatório final ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  RELATÓRIO FINAL")
    print(f"{'═'*60}")
    print(f"  {'Pasta':<4}  {'Classe':<22}  {'train':>7}  {'val':>6}  {'test':>6}  {'OK?':>4}")
    print(f"  {'─'*4}  {'─'*22}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*4}")

    all_ok = True
    for n in range(1, 26):
        cname = CLASS_NAMES[n]
        counts = {}
        for split in SPLITS:
            d = DATASET_DIR / split / str(n)
            counts[split] = len(list(d.glob("*.png"))) if d.exists() else 0

        ok = (counts["train"] == TRAIN_TARGET and
              counts["val"]   == VAL_TARGET   and
              counts["test"]  == TEST_TARGET)
        if not ok:
            all_ok = False

        mark = "✔" if ok else "✘"
        print(f"  {n:<4}  {cname:<22}  {counts['train']:>7}  "
              f"{counts['val']:>6}  {counts['test']:>6}  {mark:>4}")

    if errors:
        print(f"\n  Erros ({len(errors)}):")
        for e in errors:
            print(f"    • {e}")

    print(f"\n{'═'*60}")
    if all_ok:
        print(f"  ✔ Dataset estruturado com sucesso!")
        print(f"    train: 25 × {TRAIN_TARGET:,} = {25*TRAIN_TARGET:,} imagens")
        print(f"    val  : 25 × {VAL_TARGET:,}  = {25*VAL_TARGET:,} imagens")
        print(f"    test : 25 × {TEST_TARGET:,}  = {25*TEST_TARGET:,} imagens")
    else:
        print("  ⚠ Algumas pastas não atingiram as metas. Verifique os erros acima.")
    print(f"{'═'*60}\n")

    # Garante que pasta temporária foi removida
    tmp = PROJECT_ROOT / ".tmp_work"
    if tmp.exists():
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()

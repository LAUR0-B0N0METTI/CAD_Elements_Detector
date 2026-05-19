# CLIP CAD Detector

> **Classificador de elementos arquitetônicos em plantas CAD usando CLIP + Linear Probing**

Sistema de visão computacional que identifica e classifica automaticamente elementos arquitetônicos (portas, janelas, mobiliário, sanitários, escadas, etc.) em arquivos DXF e plantas SVG/XML da base FloorPlan, utilizando o modelo CLIP da OpenAI como backbone congelado com uma camada de classificação linear treinável.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura do Sistema](#arquitetura-do-sistema)
- [Classes Suportadas](#classes-suportadas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Descrição dos Módulos](#descrição-dos-módulos)
  - [extract_floorplan_elements.py](#extract_floorplan_elementspy)
  - [normalize_dataset.py](#normalize_datasetpy)
  - [data_utils_cl.py](#data_utils_clpy)
  - [model_cl.py](#model_clpy)
  - [train_cl.py](#train_clpy)
  - [validate_setup.py](#validate_setuppy)
  - [insert_classifier4C.py](#insert_classifier4cpy)
- [Pré-requisitos e Instalação](#pré-requisitos-e-instalação)
- [Pipeline Completo: Passo a Passo](#pipeline-completo-passo-a-passo)
- [Configuração e Parâmetros](#configuração-e-parâmetros)
- [Saída e Resultados](#saída-e-resultados)
- [Detalhes Técnicos](#detalhes-técnicos)

---

## Visão Geral

O projeto resolve um problema de classificação de imagens em um domínio altamente especializado: a identificação de blocos `INSERT` em desenhos técnicos CAD (formato DXF) e de elementos anotados em plantas arquitetônicas SVG. 

A abordagem escolhida é o **Linear Probing sobre o CLIP ViT-B/32**: o backbone visual do CLIP é mantido congelado (sem atualização de pesos), e apenas uma camada linear `512 → N_classes` é treinada do zero. Isso permite:

- Aproveitar as representações visuais ricas aprendidas pelo CLIP em larga escala.
- Treinar rapidamente, mesmo com hardware modesto (CPU ou GPU de entrada).
- Obter boa generalização com um dataset relativamente pequeno por classe.

O fluxo completo vai desde a **extração bruta de imagens** de plantas SVG anotadas, passa pela **normalização e balanceamento do dataset**, pelo **treinamento do classificador**, pela **validação de ambiente** e culmina na **inferência sobre arquivos DXF reais**.

---

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DE DADOS                            │
│                                                                     │
│  Plantas SVG/XML ──► extract_floorplan_elements.py ──► PNGs brutos  │
│                                                   (400×400 px)      │
│                              │                                      │
│                              ▼                                      │
│                   normalize_dataset.py                              │
│                   (rotações + balanceamento)                        │
│                              │                                      │
│              ┌───────────────┼───────────────┐                      │
│              ▼               ▼               ▼                      │
│          dataset/        dataset/        dataset/                   │
│           train/          val/            test/                     │
│          (8 000/cls)    (1 200/cls)    (1 200/cls)                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      PIPELINE DE MODELO                             │
│                                                                     │
│  data_utils_cl.py ──► DataLoader ──► CLIPClassifierDataset          │
│                                            │                        │
│                                            ▼                        │
│                              ┌─────────────────────────┐            │
│                              │      CLIPClassifier      │           │
│                              │  ┌────────────────────┐  │           │
│                              │  │  CLIP ViT-B/32     │  │           │
│                              │  │  (backbone frozen) │  │           │
│                              │  │  → 512-d features  │  │           │
│                              │  └────────┬───────────┘  │           │
│                              │           │ L2-norm       │           │
│                              │  ┌────────▼───────────┐  │           │
│                              │  │  Linear(512→26)    │  │           │
│                              │  │  (trainable head)  │  │           │
│                              │  └────────────────────┘  │           │
│                              └─────────────────────────┘            │
│                                            │                        │
│                              train_cl.py ──┘                        │
│                           (Adam + CrossEntropy)                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      PIPELINE DE INFERÊNCIA                         │
│                                                                     │
│  Arquivo .dxf ──► insert_classifier4C.py                           │
│                   ├── renderiza cada bloco INSERT como imagem       │
│                   ├── passa pelo CLIPClassifier treinado            │
│                   └── retorna classe + confiança por elemento       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Classes Suportadas

O modelo classifica elementos em **26 classes** (índices 0–25). As classes 0–24 correspondem a elementos arquitetônicos reais; a classe 25 (`unknown`) é reservada para inferência quando a confiança está abaixo do limiar ou o elemento não é reconhecido.

| Índice | Nome da Classe    | Pasta de Treino |
|-------:|-------------------|:--------------:|
| 0      | single door       | 1              |
| 1      | double door       | 2              |
| 2      | sliding door      | 3              |
| 3      | folding door      | 4              |
| 4      | window            | 5              |
| 5      | bay window        | 6              |
| 6      | blind window      | 7              |
| 7      | opening symbol    | 8              |
| 8      | sofa              | 9              |
| 9      | bed               | 10             |
| 10     | chair             | 11             |
| 11     | table             | 12             |
| 12     | TV cabinet        | 13             |
| 13     | wardrobe          | 14             |
| 14     | gas stove         | 15             |
| 15     | sink              | 16             |
| 16     | refrigerator      | 17             |
| 17     | air conditioner   | 18             |
| 18     | bath              | 19             |
| 19     | bath tub          | 20             |
| 20     | washing machine   | 21             |
| 21     | squat toilet      | 22             |
| 22     | urinal            | 23             |
| 23     | toilet            | 24             |
| 24     | stairs            | 25             |
| 25     | **unknown**       | *(inferência)* |

> **Nota:** A classe `unknown` (índice 25) não possui pasta de treino. Ela é atribuída automaticamente durante a inferência sempre que a confiança máxima do modelo fica abaixo do limiar `UNKNOWN_THRESHOLD = 0.40`.

---

## Estrutura do Projeto

```
clip_cad_detector/
│
├── data_selection/           # Imagens brutas selecionadas manualmente
│   ├── 1/                    # single_door
│   ├── 2/                    # double_door
│   │   ...
│   └── 25/                   # stairs
│
├── dataset/                  # Dataset estruturado e balanceado (gerado por normalize_dataset.py)
│   ├── train/
│   │   ├── 1/  ... 25/       # 8 000 imagens por classe
│   ├── val/
│   │   ├── 1/  ... 25/       # 1 200 imagens por classe
│   └── test/
│       ├── 1/  ... 25/       # 1 200 imagens por classe
│
├── model/
│   └── v2.0/
│       ├── best_model.pth              # Pesos do melhor modelo treinado
│       └── confusion_matrix_epN.png    # Matrizes de confusão por época
│
├── extract_floorplan_elements.py   # Extração de elementos de SVGs anotados
├── normalize_dataset.py            # Normalização, augmentation e estruturação
├── data_utils_cl.py                # Utilitários de dataset e mapeamento de classes
├── model_cl.py                     # Definição do modelo CLIPClassifier
├── train_cl.py                     # Loop de treinamento
├── validate_setup.py               # Validação pré-treino do ambiente
└── insert_classifier4C.py          # Inferência em arquivos DXF
```

---

## Descrição dos Módulos

### `extract_floorplan_elements.py`

**Responsabilidade:** Extrai elementos arquitetônicos individuais de arquivos SVG/XML da base FloorPlan (anotados com atributos `semantic-id` e `instance-id`) e os salva como imagens PNG normalizadas de 400×400 px.

#### Como funciona

O script realiza um pipeline de renderização em **dois passes** para cada instância de elemento encontrada:

**Passe 1 — Detecção de bounding box:**
Renderiza o elemento em escala reduzida (`PROBE_SCALE = 4`) sobre o viewport completo do SVG. Em seguida, analisa os pixels resultantes para identificar a região não-branca (fundo detectado por `BG_THRESHOLD = 238`), obtendo as coordenadas do bounding box em pixels e convertendo de volta para unidades SVG.

**Passe 2 — Renderização final normalizada:**
Com o bounding box conhecido, constrói um novo SVG com um `viewBox` ajustado (mais um padding proporcional de `PAD_FRACTION = 0.08`). Recalcula a espessura de stroke para que a linha apareça com exatamente `STROKE_PX_TARGET = 2.0 px` na saída final, independentemente da escala original do SVG. O resultado é centralizado em um canvas branco de 400×400 px, preservando a proporção original do elemento.

#### Parâmetros de configuração

| Constante         | Padrão | Descrição                                               |
|-------------------|--------|---------------------------------------------------------|
| `TARGET_PX`       | 400    | Lado do canvas de saída em pixels                       |
| `CANVAS_MARGIN_PX`| 15     | Margem interna do canvas (px)                           |
| `STROKE_PX_TARGET`| 2.0    | Espessura de stroke desejada na imagem final (px)       |
| `PROBE_SCALE`     | 4      | Escala da renderização do primeiro passe                |
| `BG_THRESHOLD`    | 238    | Limiar para detecção de pixels de fundo (0–255)         |
| `PAD_FRACTION`    | 0.08   | Padding ao redor do bbox como fração do maior lado      |
| `MAX_WORKERS`     | 4      | Número de threads paralelas para processamento          |

#### Mapeamento semântico

O script suporta **30 categorias** da base FloorPlan (incluindo elevator, escalator e revolving door, que não fazem parte do classificador final mas são extraídas para uso futuro):

```python
SEMANTIC_LABELS = {
    1:  ("01", "single_door"),
    2:  ("02", "double_door"),
    # ... até
    30: ("30", "escalator"),
}
```

Elementos com `instance-id == -1` são ignorados (representam estruturas de fundo, como paredes e eixos).

#### Processamento paralelo

O script usa `ThreadPoolExecutor` com `MAX_WORKERS` threads. Cada thread processa um arquivo SVG completo de forma independente, com os diretórios de saída pré-criados de forma thread-safe antes do início do processamento paralelo.

#### Uso

```bash
python3 extract_floorplan_elements.py
```

> Ajuste `DATASET_ROOT` no início do arquivo para apontar para o diretório raiz do dataset FloorPlan.

#### Saída

Para cada diretório pai de SVG processado, é criado um subdiretório `extraidos/` com 30 subpastas nomeadas `NN_nome_da_classe/`. Cada PNG é salvo com o nome `{stem_do_svg}_ins{instance_id:04d}.png`.

---

### `normalize_dataset.py`

**Responsabilidade:** Normaliza, balanceia e estrutura o dataset bruto (da pasta `data_selection/`) em splits `train/val/test` com quantidade fixa de imagens por classe, utilizando augmentation por rotação.

#### Estratégia de balanceamento

O script foi projetado para garantir que **todas as 25 classes** tenham exatamente o mesmo número de imagens em cada split, independentemente da quantidade de originais disponíveis:

1. **Augmentation por rotação:** Para cada imagem original, gera 3 cópias rotacionadas (90°, 180°, 270°), formando um pool inicial de 4× os originais.
2. **Preenchimento cíclico:** Se o pool ainda for menor que o alvo total (`TOTAL_TARGET = 10.400`), copia imagens do pool de forma cíclica e embaralhada até atingir o tamanho desejado.
3. **Distribuição:** O pool completo é embaralhado com semente fixa (`SEED = 42`) e dividido em:

| Split | Quantidade por classe | Total (25 classes) |
|-------|----------------------|-------------------|
| val   | 1.200                | 30.000            |
| test  | 1.200                | 30.000            |
| train | 8.000                | 200.000           |

4. **Renomeação padronizada:** Cada arquivo de destino recebe o nome `{class_name}_{N:04d}.png`, garantindo identificação inequívoca.

#### Reprodutibilidade

Todo o processo é determinístico graças ao `SEED = 42` utilizado tanto na geração do pool (`fill_pool_to`) quanto no embaralhamento final, garantindo que re-execuções produzam exatamente o mesmo dataset.

#### Uso

```bash
python3 normalize_dataset.py
```

> Ajuste `PROJECT_ROOT` no arquivo para apontar para o diretório raiz do projeto.

#### Relatório de saída

Ao final, o script exibe um relatório tabular verificando se cada classe atingiu as metas:

```
  Pasta  Classe                  train    val   test   OK?
  ────   ──────────────────────  ─────  ─────  ─────  ────
  1      single_door              8000   1200   1200     ✔
  2      double_door              8000   1200   1200     ✔
  ...
```

---

### `data_utils_cl.py`

**Responsabilidade:** Define o mapeamento canônico entre pastas numéricas do dataset e índices de classe, além das funções de carregamento dos dados de treino e validação.

#### Mapeamento de classes

A função `get_class_mapping()` retorna um dicionário `{str(pasta): class_idx}` onde a pasta `"1"` mapeia para o índice `0`, `"2"` para `1`, e assim por diante até `"25"` → `24`. Esse mapeamento é o contrato central que conecta a estrutura de diretórios ao espaço de classes do modelo.

```python
def get_class_mapping():
    return {str(i): i - 1 for i in range(1, 26)}
```

#### Carregamento do dataset

A função `load_dataset(base_path, seed=42)` carrega os splits `train` e `val`:

- **`load_folder_data(split, class_mapping, base_path)`** percorre o diretório do split em ordem numérica de pastas, coleta todos os caminhos de imagens (`.png`, `.jpg`, `.jpeg`) e cria os arrays de labels correspondentes.
- O split de treino é **embaralhado** com a semente fornecida antes de ser retornado.
- Retorna quatro arrays NumPy: `train_paths`, `train_labels`, `val_paths`, `val_labels`.

> **Importante:** O split `test` não é carregado por este módulo — ele deve ser avaliado separadamente para garantir a integridade da avaliação final.

#### `CLASS_NAMES`

Lista global com os 26 nomes de classe (índices 0–25). O índice 25 (`"unknown"`) não possui pasta de treino e serve exclusivamente como classe de rejeição durante a inferência.

---

### `model_cl.py`

**Responsabilidade:** Define a arquitetura do modelo e o dataset customizado compatível com o CLIP.

#### `CLIPClassifier`

```python
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=26):
        ...
        # Backbone CLIP completamente congelado
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # Cabeça linear treinável
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, images):
        with torch.no_grad():
            feats = self.clip_model.encode_image(images).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-norm
        return self.classifier(feats)
```

**Decisões de design:**

- **Backbone congelado (Linear Probing):** Todos os parâmetros do CLIP têm `requires_grad = False`. Isso reduz drasticamente o número de parâmetros treináveis (apenas 512 × 26 + 26 = **13.338 parâmetros**), permitindo treinar mais rápido, com menos dados e menor risco de overfitting.

- **Normalização L2:** As features extraídas pelo CLIP são normalizadas antes da camada linear. Isso é consistente com o espaço de embeddings para o qual o CLIP foi treinado (cosine similarity), tornando as features mais estáveis e a classificação linear mais eficaz.

- **`torch.no_grad()` no forward:** O cálculo de features é explicitamente executado sem gradientes, garantindo que nenhum gradiente acidental seja propagado pelo backbone.

#### `CLIPClassifierDataset`

Dataset PyTorch customizado que recebe arrays de caminhos e labels, abre as imagens com PIL (convertendo para RGB) e aplica a transformação do CLIP (`preprocess`).

---

### `train_cl.py`

**Responsabilidade:** Orquestra o treinamento completo do classificador, incluindo validação por época, early stopping, salvamento do melhor modelo e geração de matrizes de confusão.

#### Configuração via argumentos de linha de comando

| Argumento       | Padrão | Descrição                                          |
|-----------------|--------|----------------------------------------------------|
| `--base_path`   | *obrigatório* | Caminho para a raiz do dataset              |
| `--version`     | `2.0`  | Versão do experimento (define a pasta de saída)    |
| `--batch_size`  | 64     | Tamanho do batch de treinamento                    |
| `--lr`          | 1e-3   | Taxa de aprendizado do otimizador Adam             |
| `--num_epochs`  | 60     | Número máximo de épocas                            |
| `--save_every`  | 5      | Frequência (em épocas) para salvar matriz de confusão |
| `--patience`    | 8      | Épocas sem melhora antes do early stopping         |
| `--num_workers` | 6      | Workers para o DataLoader                          |
| `--early_stop`  | False  | Ativa o early stopping se fornecido               |

#### Loop de treinamento

Para cada época:

1. **Fase de treino:** Percorre todos os batches, calcula a `CrossEntropyLoss`, executa backpropagation e atualiza apenas os pesos da camada `classifier` via `Adam`.

2. **Fase de validação:** Avalia o modelo no conjunto de validação sem gradientes, calculando `val_loss` e `accuracy`.

3. **Salvamento do melhor modelo:** Se `avg_val_loss` for menor que o melhor registrado, salva o `state_dict` como `best_model.pth` e reseta o contador de paciência.

4. **Early stopping:** Se `--early_stop` estiver ativo e o contador de paciência atingir `--patience`, o treinamento é interrompido.

5. **Matriz de confusão:** A cada `save_every` épocas (e obrigatoriamente na época 1), gera e salva uma matriz de confusão completa (25×25) em formato PNG usando seaborn/matplotlib.

#### Exemplo de uso

```bash
python train_cl.py \
    --base_path ~/Documentos/01_Projetos/clip_cad_detector/dataset \
    --version 2.0 \
    --batch_size 64 \
    --lr 1e-3 \
    --num_epochs 60 \
    --save_every 5 \
    --patience 8 \
    --early_stop
```

> **Dica para execução em background (sem travar o terminal):**
> ```bash
> nohup python train_cl.py --base_path /caminho/dataset > train_log.txt 2>&1 &
> ```

#### Saída

```
~/Documentos/01_Projetos/clip_cad_detector/model/v2.0/
├── best_model.pth
├── confusion_matrix_ep1.png
├── confusion_matrix_ep5.png
├── confusion_matrix_ep10.png
└── ...
```

---

### `validate_setup.py`

**Responsabilidade:** Verificação pré-treino completa do ambiente, garantindo que todas as dependências, a estrutura do dataset e o pipeline de dados estão corretos antes de iniciar o treinamento.

#### Verificações realizadas

O script executa **6 blocos de validação** em sequência, abortando com `sys.exit(1)` na primeira falha encontrada:

**[1] Dependências Python**
Tenta importar cada pacote necessário (`clip`, `torch`, `numpy`, `sklearn`, `PIL`, `tqdm`, `matplotlib`) e reporta a versão do PyTorch e disponibilidade de CUDA.

**[2] Estrutura do dataset**
Verifica se os diretórios `train/`, `val/` e `test/` existem e se cada um contém as 25 pastas numéricas (1–25). Conta e reporta o total de imagens em cada split.

**[3] Carregamento via `data_utils_cl`**
Chama `load_dataset()` e verifica se os arrays retornados são não-vazios e se exatamente 25 classes estão presentes no treino.

**[4] CLIP + CLIPClassifier**
Instancia o modelo completo e reporta o total de parâmetros e quantos são treináveis (deve ser apenas a cabeça linear).

**[5] Forward pass de teste**
Roda um batch de 8 imagens reais pelo modelo e verifica que a forma de saída é `(8, 26)`.

**[6] Lista de classes**
Imprime a listagem completa de todas as 26 classes com seus índices para conferência visual.

#### Uso

```bash
python validate_setup.py --base_path ~/Documentos/01_Projetos/clip_cad_detector/dataset
```

#### Saída esperada (ambiente correto)

```
============================================================
  VALIDAÇÃO DO AMBIENTE DE TREINAMENTO
============================================================

[1] Dependências Python
  [OK] clip
  [OK] torch
  ...

[6] Lista de classes
     0: single door
     1: double door
  ...

============================================================
  TUDO OK — pode iniciar o treinamento!
============================================================
```

---

### `insert_classifier4C.py`

**Responsabilidade:** Inferência em arquivos DXF reais — renderiza cada bloco `INSERT` do modelo space como uma imagem e classifica utilizando o `CLIPClassifier` treinado.

#### Classe `CADElementClassifier`

A classe encapsula todo o pipeline de inferência:

**`__init__(model_path, num_classes, device)`**
Carrega o backbone CLIP e os pesos treinados, coloca o modelo em modo `eval()` e configura o dispositivo (CUDA ou CPU automaticamente).

**`classify_insert(doc, insert)`**
Classifica um único bloco INSERT:
1. Chama `_render_block()` para obter uma imagem PIL do bloco.
2. Aplica o `preprocess` do CLIP e passa pelo modelo.
3. Aplica `softmax` sobre os logits e obtém a classe de maior probabilidade.
4. Se a confiança máxima for menor que `UNKNOWN_THRESHOLD = 0.40`, retorna `"unknown"`.

**`_render_block(doc, insert, image_size, padding)`**
Renderiza um bloco DXF como imagem:
- Busca o bloco pelo nome no dicionário de blocos do documento.
- Converte entidades DXF em shapes geométricas (via módulo `shape`, com fallback se não disponível).
- Calcula escala e offset para caber no canvas com padding.
- Delega o desenho ao `draw_shapes()` do módulo `shape`.

**`process_doc(doc)`**
Itera sobre todos os `INSERT` do model space, classifica cada um e retorna:
- `results`: dicionário `{handle: {'class': str, 'confidence': float}}`.
- `counts`: contagem de elementos por classe.
- `excluded`: lista de handles classificados como `"unknown"`.

**`process_dxf_file(dxf_path)`**
Atalho que lê o arquivo DXF com `ezdxf` e chama `process_doc()`.

#### Limiar de confiança

```python
UNKNOWN_THRESHOLD = 0.40
```

Qualquer predição com probabilidade máxima abaixo de 40% é automaticamente reclassificada como `"unknown"`. Esse valor pode ser ajustado conforme a necessidade de precisão vs. recall da aplicação.

#### Dependências externas

- `ezdxf`: leitura de arquivos DXF.
- `clip`: modelo CLIP da OpenAI.
- `cv2` (OpenCV): conversão de formato de imagem.
- `shape` (módulo local, opcional): renderização de entidades DXF. Se não disponível, o script usa um fallback com canvas branco.

#### Exemplo de uso como API

```python
from insert_classifier4C import CADElementClassifier

classifier = CADElementClassifier(
    model_path="model/v2.0/best_model.pth",
    num_classes=26
)

results, counts, excluded = classifier.process_dxf_file("planta.dxf")

CADElementClassifier.print_summary(counts)

# Acessando resultado de um INSERT específico
for handle, info in results.items():
    print(f"  Handle {handle}: {info['class']} (conf: {info['confidence']:.2%})")
```

#### Exemplo de saída

```
=== RESUMO DE ELEMENTOS DETECTADOS ===
  single door         : 12
  window              : 8
  sofa                : 3
  bed                 : 2
  toilet              : 4
  unknown             : 1
  TOTAL               : 30
========================================
```

---

## Pré-requisitos e Instalação

### Requisitos de sistema

- Python 3.9+
- CUDA (opcional, mas recomendado para treinamento)
- `libcairo2` (para renderização SVG via CairoSVG)

### Instalação das dependências Python

```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install lxml cairosvg Pillow numpy
pip install scikit-learn matplotlib seaborn tqdm
pip install ezdxf opencv-python
```

> **Instalação do libcairo (Ubuntu/Debian):**
> ```bash
> sudo apt-get install libcairo2-dev
> ```

---

## Pipeline Completo: Passo a Passo

### Etapa 1 — Extração de imagens do dataset FloorPlan

```bash
# Ajuste DATASET_ROOT em extract_floorplan_elements.py antes de executar
python3 extract_floorplan_elements.py
```

As imagens serão salvas em subdiretórios `extraidos/` dentro de cada pasta de splits do dataset FloorPlan.

### Etapa 2 — Selecionar e organizar imagens brutas

Mova ou copie as imagens de cada classe para as pastas correspondentes em `data_selection/`:

```
data_selection/
├── 1/   ← imagens de single_door
├── 2/   ← imagens de double_door
...
└── 25/  ← imagens de stairs
```

### Etapa 3 — Normalizar e balancear o dataset

```bash
# Ajuste PROJECT_ROOT em normalize_dataset.py antes de executar
python3 normalize_dataset.py
```

Isso criará a estrutura completa em `dataset/train/`, `dataset/val/` e `dataset/test/`.

### Etapa 4 — Validar o ambiente de treinamento

```bash
python validate_setup.py \
    --base_path ~/Documentos/01_Projetos/clip_cad_detector/dataset
```

Confirme que todos os 6 blocos reportam `[OK]` antes de prosseguir.

### Etapa 5 — Treinar o modelo

```bash
python train_cl.py \
    --base_path ~/Documentos/01_Projetos/clip_cad_detector/dataset \
    --version 2.0 \
    --batch_size 64 \
    --lr 1e-3 \
    --num_epochs 60 \
    --save_every 5 \
    --patience 8 \
    --early_stop
```

O melhor modelo será salvo em `model/v2.0/best_model.pth`.

### Etapa 6 — Inferência em arquivo DXF

```python
from insert_classifier4C import CADElementClassifier

clf = CADElementClassifier("model/v2.0/best_model.pth")
results, counts, excluded = clf.process_dxf_file("sua_planta.dxf")
CADElementClassifier.print_summary(counts)
```

---

## Configuração e Parâmetros

### Principais constantes ajustáveis

| Arquivo                          | Constante              | Descrição                                    |
|----------------------------------|------------------------|----------------------------------------------|
| `extract_floorplan_elements.py`  | `TARGET_PX`            | Resolução das imagens extraídas (padrão: 400)|
| `extract_floorplan_elements.py`  | `STROKE_PX_TARGET`     | Espessura de stroke normalizada (padrão: 2.0)|
| `extract_floorplan_elements.py`  | `MAX_WORKERS`          | Threads de processamento paralelo            |
| `normalize_dataset.py`           | `TRAIN_TARGET`         | Imagens por classe no treino (padrão: 8000)  |
| `normalize_dataset.py`           | `VAL_TARGET`           | Imagens por classe no val (padrão: 1200)     |
| `normalize_dataset.py`           | `TEST_TARGET`          | Imagens por classe no test (padrão: 1200)    |
| `normalize_dataset.py`           | `SEED`                 | Semente para reprodutibilidade (padrão: 42)  |
| `insert_classifier4C.py`         | `UNKNOWN_THRESHOLD`    | Confiança mínima para aceitar classe (0.40)  |

---

## Saída e Resultados

### Modelo treinado

```
model/v2.0/
├── best_model.pth           # State dict do melhor checkpoint (menor val_loss)
├── confusion_matrix_ep1.png
├── confusion_matrix_ep5.png
└── confusion_matrix_ep10.png
```

### Formato do `best_model.pth`

O arquivo salvo é diretamente o `state_dict` do `CLIPClassifier`. Para carregar:

```python
import clip, torch
from model_cl import CLIPClassifier
from data_utils_cl import NUM_CLASSES

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
model = CLIPClassifier(clip_model, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("model/v2.0/best_model.pth", map_location=device))
model.eval()
```

> **Compatibilidade:** O `insert_classifier4C.py` aceita tanto o `state_dict` puro quanto dicionários com chave `'state_dict'` (formato de alguns frameworks de checkpoint), resolvendo automaticamente via `state.get('state_dict', state)`.

---

## Detalhes Técnicos

### Por que Linear Probing com CLIP?

O CLIP foi treinado com contrastive learning em centenas de milhões de pares imagem-texto, desenvolvendo representações visuais altamente transferíveis. Para um domínio como desenhos CAD (imagens essencialmente de linhas e formas geométricas), as features do CLIP capturam a estrutura e topologia dos elementos de forma surpreendentemente eficaz, mesmo sem ter sido treinado explicitamente nesse domínio.

O Linear Probing (congelar o backbone, treinar apenas a cabeça linear) é particularmente adequado aqui porque:

- **Eficiência:** Apenas ~13 mil parâmetros treináveis vs. os ~87 milhões do ViT-B/32 completo.
- **Velocidade:** O forward pass do backbone pode ser pré-computado e cacheado.
- **Generalização:** Menos parâmetros = menos overfitting com datasets de tamanho médio.
- **Reprodutibilidade:** Resultados muito mais estáveis entre diferentes runs.

### Normalização L2 das features

A normalização L2 antes da camada linear é essencial para manter a consistência com o espaço de embeddings do CLIP, onde a similaridade é medida por cosseno. Sem essa normalização, a escala dos embeddings pode variar significativamente entre imagens, prejudicando a classificação linear.

### Gerenciamento de memória no treinamento

O `torch.no_grad()` no método `forward` do `CLIPClassifier` garante que nenhum grafo computacional seja criado para o backbone durante o forward pass de treino, economizando memória de GPU substancialmente (mesmo que os gradientes do backbone não fossem calculados de qualquer forma por conta do `requires_grad = False`).

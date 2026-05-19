```python

doc_v2 = """# AI Powered CAD Element Detector & FloorPlan Cleaner (CLIP-CAD)

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B%2F32-000000.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Este repositório contém a arquitetura completa de um sistema inteligente para detecção automatizada de elementos arquitetônicos e limpeza de plantas baixa em formato CAD. Desenvolvido como um projeto de engenharia de ponta por **Lauro Bonometti**, o sistema utiliza aprendizado multimodal por meio de **Linear Probing sobre o CLIP (ViT-B/32)** para classificar e filtrar blocos estruturais e mobiliários a partir de arquivos vetoriais (`SVG` e `DXF`). A aplicação acelera drasticamente a preparação de plantas técnicas eliminando ruídos e elementos não essenciais de forma automatizada.

---

## 📑 Índice
1. [Contexto de Negócio e Problema](#1-contexto-de-negócio-e-problema)
2. [A Solução Proposta](#2-a-solução-proposta)
3. [Arquitetura Geral do Sistema](#3-arquitetura-geral-do-sistema)
4. [Mapeamento e Engenharia de Classes](#4-mapeamento-e-engenharia-de-classes)
5. [Pipeline de Dados (Data Engineering)](#5-pipeline-de-dados-data-engineering)
6. [Estrutura do Modelo e Treinamento](#6-estrutura-do-modelo-e-treinamento)
7. [Módulos do Sistema e Detalhes de Código](#7-módulos-do-sistema-e-detalhes-de-código)
8. [Como Executar o Projeto](#8-como-executar-o-projeto)
9. [Resultados e Pipeline de Inferência (`DXF`)](#9-resultados-e-pipeline-de-inferência-dxf)
10. [Autor](#10-autor)

---

## 1. Contexto de Negócio e Problema

Empresas de engenharia, arquitetura e instalação de sistemas prediais complexos (como redes de prevenção de incêndio ou automação) enfrentam um gargalo crônico na fase de aprovação de projetos técnicos. Para submissão em órgãos regulamentadores, vistorias e análises oficiais, as plantas baixas estruturais precisam estar perfeitamente limpas. Isso exige que contenham apenas elementos primários e estruturais (como paredes, portas, janelas, escadas, elevadores). 

No entanto, as plantas originais fornecidas pelos clientes costumam vir repletas de ruído geométrico, decorações, camadas de cotas, texturas complexas e blocos de mobiliários que poluem visualmente o layout.

**O Desafio:** Isolar, filtrar e remover manualmente centenas de blocos e geometrias aninhadas (`INSERTs` e linhas primitivas) em plantas industriais e edifícios comerciais de grande porte exige uma dedicação exaustiva de engenheiros qualificados. Esse processo manual consome um tempo valioso que atrasa o cronograma das obras e a entrega regulatória.

---

## 2. A Solução Proposta

Para automatizar a identificação de blocos e padronizar o processo de limpeza das geometrias, foi projetado um pipeline completo que utiliza Inteligência Artificial profunda. Na fase de P&D, foram testadas e avaliadas 4 arquiteturas proeminentes no estado da arte para reconhecimento visual e segmentação de layouts arquitetônicos:
1. **OpenCLIP** (Abordagem multimodal via representações contrastivas)
2. **SympointV2** (Detecção baseada em pontos de simetria)
3. **DPSS** (Deep Point-Set Segmentation)
4. **Mask2Former** (Segmentação universal por máscaras)

**O Modelo Escolhido:** O **OpenCLIP (com backbone ViT-B/32)** apresentou a melhor capacidade de generalização e abstração sobre representações esquemáticas e traços vetoriais brutos, demonstrando imunidade a variações severas de escala e exibindo o menor tempo de inferência por elemento isolado.

### Obtenção e Domínio dos Dados
A base de conhecimento foi construída a partir da consolidação estratégica e do ajuste de dois importantes datasets acadêmicos e técnicos do setor: o **FloorPlan** e o **ResPlan**. 

Desta união, foi extraído um pool robusto de **40.000 arquivos** de plantas e dados geométricos. Os dados foram inteiramente tratados, limpos de anomalias primitivas, segmentados e convertidos de formato inteiramente vetorial (`SVG`) para matricial (`PNG`) de alta definição para alimentar nativamente as camadas de encoders do CLIP.

---

## 3. Arquitetura Geral do Sistema

O fluxo lógico do ecossistema de dados divide-se nas seguintes frentes operacionais:

[ Datasets FloorPlan & ResPlan ] ──> [ Módulo Extrator SVG ] ──> [ Normalização/Augmentation ]
│
[ Inferência DXF (Plantas Brutas) ] <── [ Modelo CLIP Ajustado ] <────── [ Pipeline Treino ]

1. **Ingestão e Extração Vetorial (`extract_floorplan_elements.py`):** Varre arquivos SVG originais, isola elementos individuais por meio de bounding boxes dinâmicas e padroniza as geometrias.
2. **Processamento e Balanceamento (`normalize_dataset.py`):** Expande e balanceia o dataset aplicando rotações ortogonais controladas e segmenta os splits de forma estrita em **70% Treinamento, 15% Validação e 15% Teste**.
3. **Classificação e Filtragem (`insert_classifier4C.py`):** Realiza a inferência em ambiente real sobre arquivos CAD nativos (`DXF`), extraindo cada bloco (`INSERT`), identificando a sua classe e excluindo automaticamente itens classificados como mobiliários ou descartes geométricos.

---

## 4. Mapeamento e Engenharia de Classes

O modelo foi projetado e treinado para compreender com alto rigor granular **26 classes textuais e visuais** (25 categorias estruturadas provenientes do dataset limpo e 1 classe especializada para tratamento de incertezas em ambiente de produção).

| Índice | Classe Cadastrada | Origem (Pasta) | Descrição Tecnológica |
|---|---|---|---|
| **0** | single door | pasta 1 | Porta de folha simples |
| **1** | double door | pasta 2 | Porta de folha dupla |
| **2** | sliding door | pasta 3 | Porta de correr lateral |
| **3** | folding door | pasta 4 | Porta articulada/sanfonada |
| **4** | window | pasta 5 | Janela padrão |
| **5** | bay window | pasta 6 | Janela saliente (Bay window) |
| **6** | blind window | pasta 7 | Janela cega / Elemento opaco |
| **7** | opening symbol | pasta 8 | Símbolo de abertura/vão livre |
| **8** | sofa | pasta 9 | Sofá / Mobiliário de descanso |
| **9** | bed | pasta 10 | Cama (Estrutura interna) |
| **10** | chair | pasta 11 | Cadeira / Assento individual |
| **11** | table | pasta 12 | Mesa (Reunião / Jantar) |
| **12** | TV cabinet | pasta 13 | Rack / Painel de TV |
| **13** | wardrobe | pasta 14 | Armário embutido / Guarda-roupa |
| **14** | gas stove | pasta 15 | Fogão a gás / Cooktop |
| **15** | sink | pasta 16 | Pia / Cuba de cozinha ou banheiro |
| **16** | refrigerator | pasta 17 | Geladeira / Refrigerador |
| **17** | air conditioner | pasta 18 | Unidade de Ar Condicionado |
| **18** | toilet | pasta 19 | Vaso sanitário / Bacia |
| **19** | bathtub | pasta 20 | Banheira |
| **20** | shower | pasta 21 | Chuveiro / Box |
| **21** | washbasin | pasta 22 | Lavatório / Pia de coluna |
| **22** | urinal | pasta 23 | Mictório |
| **23** | elevator | pasta 24 | Cabine / Poço de Elevador |
| **24** | stairs | pasta 25 | Escadas (Lances / Caracol) |
| **25** | unknown | *N/A* | Classe de descarte (Confiança < `UNKNOWN_THRESHOLD`) |

---

## 5. Pipeline de Dados (Data Engineering)

### Passo 1: Extração de Elementos de Planta Baixa (`extract_floorplan_elements.py`)
Arquivos arquitetônicos nativos costumam apresentar problemas drásticos de variação de escala de traço. Para contornar esse problema, este script implementa uma técnica de **Renderização em Duas Passagens**:
1. **Primeira Passagem:** Faz o parsing do XML estruturado do SVG através da biblioteca `lxml`, analisa as coordenadas primitivas e calcula a bounding box real do elemento.
2. **Segunda Passagem:** Reescreve e redefine a propriedade `viewBox` do SVG de forma restrita, utilizando a biblioteca `cairosvg` para renderizar o elemento de maneira centralizada em uma matriz matricial quadrada de **400×400 pixels**.

* **Normalização de Traço (Stroke Normalization):** O script reescreve programaticamente os atributos `stroke-width` de todas as linhas e curvas. Isso garante que, na imagem final gerada, todas as linhas tenham uma espessura uniforme de **~2 pixels**, impedindo que o modelo sofra com vieses causados pelo peso visual ou grossura original do traço gráfico.

### Passo 2: Normalização e Distribuição de Splits (`normalize_dataset.py`)
Para expandir a robustez posicional da rede contra variações de orientação, cada imagem original passa por um processo de data augmentation estático que gera 3 rotações ortogonais completas (90°, 180° e 270°), gerando um pool expandido de dados. Esse conjunto é embaralhado de forma estocástica e distribuído ciclicamente até cumprir as metas exatas de volumes definidos pelas regras de governança (70% Treino, 15% Validação e 15% Teste).

---

## 6. Estrutura do Modelo e Treinamento

### Arquitetura de Redes (`model_cl.py`)
O sistema foi construído sobre a estratégia de **Linear Probing** aplicada ao modelo `ViT-B/32` do CLIP. Os pesos do codificador visual (Backbone) são inteiramente congelados (`requires_grad = False`), atuando como um extrator de características altamente robusto de 512 dimensões espacialmente estáveis.

Uma cabeça linear customizada (`nn.Linear(512, 26)`) é adicionada na saída para projetar o vetor de características diretamente no espaço probabilístico das nossas 26 classes. As características extraídas passam por uma normalização L2 (`F.normalize`), mantendo a integridade métrica da escala original do CLIP.

### Pipeline de Treinamento (`train_cl.py`)
O treinamento é executado por meio de um loop de otimização contínua com suporte a parada antecipada (`Early Stopping`), decaimento de taxa de aprendizado e geração automatizada de relatórios estatísticos de desempenho:
* **Matriz de Confusão:** Salva periodicamente em disco em formato `.png` usando `seaborn` e `sklearn`, permitindo o diagnóstico visual de intersecção ou ambiguidade de classes.
* **Mecanismo Anti-Crash (Headless Mode):** Configura explicitamente o backend `Agg` do `matplotlib`, permitindo a execução remota e robusta via sessões em background (`nohup`) sem dependência de servidores X11 ou interfaces de display.

---

## 7. Módulos do Sistema e Detalhes de Código

O ecossistema de software é composto por 7 scripts especialistas estruturados de forma modular:

1. **`data_utils_cl.py`**: Contém o mapa de classes global (`CLASS_NAMES`) e as funções de carregamento automatizado das imagens em memória (`load_folder_data`, `load_dataset`). Mapeia as pastas numéricas (`1-25`) para índices de tensores lógicos (`0-24`).
2. **`model_cl.py`**: Define a classe do dataset PyTorch (`CLIPClassifierDataset`) — responsável por forçar a conversão de imagens para o modo `RGB` exigido pelo CLIP — e a estrutura de rede do classificador linear (`CLIPClassifier`).
3. **`train_cl.py`**: Script de treinamento em lote. Aceita argumentos configuráveis por linha de comando (`--batch_size`, `--lr`, `--num_epochs`, `--patience`, `--base_path`) e implementa o loop de otimização utilizando a função de perda por entropia cruzada (`CrossEntropyLoss`) e Adam Optimizer.
4. **`validate_setup.py`**: Utilitário de verificação e sanidade pré-treino. Valida a presença de dependências cruciais do ecossistema de Machine Learning, checa a integridade das pastas de splits e realiza um **Forward Pass de Teste** com um batch dummy de 8 imagens para assegurar a consistência do ambiente de execução.
5. **`extract_floorplan_elements.py`**: Motor de engenharia de dados responsável pelo processamento paralelo multithread (`ThreadPoolExecutor`) dos arquivos SVG brutos, cálculo de viewBox dinâmica e normalização de traços de renderização.
6. **`normalize_dataset.py`**: Gerenciador do ciclo de vida dos dados. Realiza o balanceamento numérico perfeito entre as classes do dataset, injeta variações por rotação geométrica e consolida a estrutura final de diretórios requerida pelo treinamento.
7. **`insert_classifier4C.py`**: Componente crítico de inferência e produção. Carrega o modelo de melhor performance histórica (`best_model.pth`), realiza o parsing de arquivos CAD comerciais `.dxf` (via biblioteca `ezdxf`), isola as entidades dentro de blocos de inserção (`INSERT`), renderiza-as temporariamente e atribui uma classe e score de confiança. Geometrias abaixo do limiar de classificação (`UNKNOWN_THRESHOLD = 0.40`) são marcas automáticas como `unknown` para isolamento e exclusão do projeto final.

---

## 8. Como Executar o Projeto

### Instalação das Dependências Técnicas
Configure o seu ambiente Python executando o comando de instalação:

```bash
pip install torch torchvision clip-by-openai ezdxf lxml cairosvg Pillow numpy scikit-learn matplotlib seaborn tqdm

```

### Passo 1: Extração dos Dados a partir dos Datasets FloorPlan & ResPlan

Projete os arquivos brutos baixados dentro do diretório configurado e execute a extração vetorial:

```bash
python3 extract_floorplan_elements.py

```

### Passo 2: Balanceamento, Augmentation e Divisão dos Splits

Rode o utilitário de normalização para estruturar a árvore de diretórios de treino, teste e validação:

```bash
python3 normalize_dataset.py

```

### Passo 3: Validação de Sanidade do Ambiente

Antes de iniciar processamentos custosos, valide se o hardware e a estrutura de tensores estão perfeitamente operacionais:

```bash
python3 validate_setup.py --base_path /caminho/para/o/seu/dataset

```

### Passo 4: Executar o Pipeline de Treinamento

Inicie o treinamento otimizado passando as métricas ideais para o seu hardware:

```bash
nohup python3 train_cl.py --base_path /caminho/para/o/seu/dataset --batch_size 64 --lr 1e-3 --num_epochs 60 --early_stop > treino.log 2>&1 &

```

---

## 9. Resultados e Pipeline de Inferência (`DXF`)

A inferência opera diretamente sobre plantas arquitetônicas complexas em formato `.dxf`. O algoritmo lê o arquivo vetorial, detecta cada bloco e gera um sumário estatístico completo das entidades:

```python
from insert_classifier4C import CADElementClassifier

# Inicializa o classificador com os pesos gerados no treinamento
classifier = CADElementClassifier(model_path="models/best_model.pth")

# Executa o mapeamento estrutural e identificação de blocos na planta
results, counts, excluded = classifier.process_dxf_file("planta_baixa_bruta.dxf")

# Exibe o resumo estatístico no console
classifier.print_summary(counts)

```

### Exemplo de Saída no Terminal (Sumário Executivo)

```text
=== RESUMO DE ELEMENTOS DETECTADOS ===
  single door         : 42
  double door         : 12
  window              : 56
  sofa                : 8   <-- Elemento de descarte identificado
  chair               : 24  <-- Elemento de descarte identificado
  stairs              : 4
  unknown             : 15  <-- Filtrados pelo threshold de segurança
======================================
TOTAL DE ELEMENTOS PROCESSADOS: 161

```

A partir dos dicionários de handles gerados pelo classificador (`results` e `excluded`), o script invoca funções do `ezdxf` para expurgar instantaneamente todas as entidades classificadas como mobiliário ou blocos ruidosos. O tempo total gasto para realizar a filtragem complexa de uma planta baixa foi reduzido de **várias horas de desenho manual para menos de 2 minutos**.

---

## 10. Autor

* **Lauro Bonometti** — *Idealizador e Desenvolvedor do Projeto* — [GitHub](https://github.com/LAUR0-B0N0METTI) / [LinkedIn](https://www.linkedin.com/in/laurobonometti/)
"""

with open("README_v2.md", "w", encoding="utf-8") as f:
f.write(doc_v2)
print("README_v2.md gerado com sucesso!")

```

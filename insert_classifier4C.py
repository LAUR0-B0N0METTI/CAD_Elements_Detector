import os, clip, ezdxf, torch, cv2
from PIL import Image
from tqdm import tqdm
from model_cl import CLIPClassifier
from data_utils_cl import CLASS_NAMES, NUM_CLASSES

try:
    from shape import entity2shape, get_limits, draw_shapes
except ImportError:
    def entity2shape(e): return None
    def get_limits(s): return (0, 1, 0, 1)
    def draw_shapes(s, sz, o, sc):
        import numpy as np
        return (255 * np.ones((sz[1], sz[0], 3), dtype='uint8'))


class CADElementClassifier:
    """
    Classifica blocos INSERT de um arquivo DXF em 26 classes:
      0-24 : elementos da lista CAD
      25   : unknown  (confiança abaixo do limiar OU classe não reconhecida)
    """

    UNKNOWN_THRESHOLD = 0.40   # confiança mínima para aceitar classificação

    def __init__(self, model_path, num_classes=NUM_CLASSES, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        clip_model, preprocess = clip.load('ViT-B/32', device=self.device)
        self.preprocess   = preprocess
        self.model        = CLIPClassifier(clip_model, num_classes=num_classes).to(self.device)
        state             = torch.load(model_path, map_location=self.device)
        weights           = state.get('state_dict', state)
        self.model.load_state_dict(weights)
        self.model.eval()
        self.class_labels = CLASS_NAMES   # lista com 26 nomes

    # ── Classificação de um único INSERT ─────────────────────────────────────

    def classify_insert(self, doc, insert):
        img = self._render_block(doc, insert)
        if img is None:
            return "unknown", 0.0

        pil_img    = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs   = torch.softmax(outputs, dim=1)[0]
            max_p, idx = torch.max(probs, 0)

        confidence = max_p.item()
        if confidence < self.UNKNOWN_THRESHOLD:
            return "unknown", confidence
        return self.class_labels[idx.item()], confidence

    # ── Renderiza bloco DXF como imagem ──────────────────────────────────────

    def _render_block(self, doc, insert, image_size=(512, 512), padding=20):
        try:
            name       = insert.dxf.name
            block      = doc.blocks.get(name)
            shape_list = [s for s in [entity2shape(e) for e in block] if s]
            if not shape_list:
                return None
            min_x, max_x, min_y, max_y = get_limits(shape_list)
            dx, dy = max_x - min_x, max_y - min_y
            if dx == 0 or dy == 0:
                return None
            scale  = min((image_size[0] - 2 * padding) / dx,
                         (image_size[1] - 2 * padding) / dy)
            offset = (min_x - padding / scale, min_y - padding / scale)
            return draw_shapes(shape_list, image_size, offset, scale)
        except Exception:
            return None

    # ── Processa um documento DXF inteiro ────────────────────────────────────

    def process_doc(self, doc):
        """
        Retorna:
          results  : {handle: {'class': str, 'confidence': float}}
          counts   : {class_name: int}  – contagem por categoria
          excluded : [handle]           – handles com classe 'unknown'
        """
        msp     = doc.modelspace()
        results = {}
        counts  = {name: 0 for name in self.class_labels}
        excluded = []

        for insert in tqdm(msp.query("INSERT"), desc="Classificando INSERTs"):
            cl, conf = self.classify_insert(doc, insert)
            handle   = insert.dxf.handle
            results[handle] = {"class": cl, "confidence": round(conf, 4)}
            counts[cl] += 1
            if cl == "unknown":
                excluded.append(handle)

        return results, counts, excluded

    def process_dxf_file(self, dxf_path):
        return self.process_doc(ezdxf.readfile(dxf_path))

    # ── Sumário ───────────────────────────────────────────────────────────────

    @staticmethod
    def print_summary(counts):
        print("\n=== RESUMO DE ELEMENTOS DETECTADOS ===")
        total = 0
        for name, n in counts.items():
            if n > 0:
                print(f"  {name:<20}: {n}")
                total += n
        print(f"  {'TOTAL':<20}: {total}")
        print("=" * 40)

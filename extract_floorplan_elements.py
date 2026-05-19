"""
FloorPlan Dataset — Architectural Element Extractor  v2.0
=========================================================
Extracts individual architectural elements from FloorPlan SVG/XML files
and saves them as normalised 400×400 PNG images organised by category.

Key features
------------
• Two-pass rendering: first detects per-element bounding box, then
  re-renders with a tight viewBox so the element fills the canvas.
• Aspect-ratio preserved: tall or wide elements are letterboxed on a
  white 400×400 background — no distortion.
• Stroke normalisation: all strokes are rewritten to produce ~2 px
  thick lines in the final 400×400 output, regardless of the original
  SVG scale.
• Recursive dataset traversal: processes SVG files in every sub-folder
  including coco_vis/ sub-folders, but skips extraidos_CLIP_old/.
• Thread-safe parallel processing with a configurable worker pool.

Dependencies
------------
    pip install lxml cairosvg Pillow numpy

Usage
-----
    python3 extract_floorplan_elements.py

Adjust DATASET_ROOT and the tuning constants in the CONFIGURATION
section if needed.
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import io
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────────────
try:
    import cairosvg
    import numpy as np
    from lxml import etree
    from PIL import Image
except ImportError as _e:
    sys.exit(
        f"[ERROR] Missing dependency: {_e}\n"
        "Run: pip install lxml cairosvg Pillow numpy"
    )

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  edit these as needed
# ════════════════════════════════════════════════════════════════════════════

DATASET_ROOT = Path(
    "/home/zeratull/Documentos/01_Projetos/clip_cad_detector/dataset"
)

# Output folder created inside each split / coco_vis directory
EXTRAIDOS_FOLDER = "extraidos"

# Directories to skip entirely (case-insensitive name match)
SKIP_DIR_NAMES: set = {"extraidos_clip_old"}

# Output image settings
TARGET_PX        = 400    # final canvas side (pixels)
CANVAS_MARGIN_PX = 15     # whitespace border inside the 400x400 canvas
STROKE_PX_TARGET = 2.0    # desired stroke thickness in the final image (px)

# First-pass render scale (used only to detect bounding box)
PROBE_SCALE = 4            # 400 px for a 100-unit viewport; fast enough

# Background detection threshold (0-255, lower = stricter)
BG_THRESHOLD = 238         # pixels with ALL channels >= this are background

# Padding added around the detected bounding box (fraction of bbox max side)
PAD_FRACTION = 0.08

# Parallel workers (set to 1 to disable threading for debugging)
MAX_WORKERS = 4

# ════════════════════════════════════════════════════════════════════════════
# SEMANTIC-ID -> (folder_index_str, label) mapping
# ════════════════════════════════════════════════════════════════════════════

SEMANTIC_LABELS = {
    1:  ("01", "single_door"),
    2:  ("02", "double_door"),
    3:  ("03", "sliding_door"),
    4:  ("04", "folding_door"),
    5:  ("05", "revolving_door"),
    6:  ("06", "rolling_door"),
    7:  ("07", "window"),
    8:  ("08", "bay_window"),
    9:  ("09", "blind_window"),
    10: ("10", "opening_symbol"),
    11: ("11", "sofa"),
    12: ("12", "bed"),
    13: ("13", "chair"),
    14: ("14", "table"),
    15: ("15", "TV_cabinet"),
    16: ("16", "wardrobe"),
    17: ("17", "cabinet"),
    18: ("18", "gas_stove"),
    19: ("19", "sink"),
    20: ("20", "refrigerator"),
    21: ("21", "airconditioner"),
    22: ("22", "bath"),
    23: ("23", "bath_tub"),
    24: ("24", "washing_machine"),
    25: ("25", "squat_toilet"),
    26: ("26", "urinal"),
    27: ("27", "toilet"),
    28: ("28", "stairs"),
    29: ("29", "elevator"),
    30: ("30", "escalator"),
}

# ════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# SVG NAMESPACES
# ════════════════════════════════════════════════════════════════════════════

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"

# ════════════════════════════════════════════════════════════════════════════
# CORE RENDERING PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def parse_svg(path):
    """
    Parse an SVG/XML file.  Strips BOM / leading whitespace before parsing
    so lxml does not choke on files with an extra blank line before the
    XML declaration.

    Returns (root_element, viewport_width, viewport_height).
    """
    with open(path, "rb") as fh:
        raw = fh.read().lstrip()
    root = etree.fromstring(raw)
    vb = root.get("viewBox", "0 0 100 100").split()
    return root, float(vb[2]), float(vb[3])


def collect_instances(root):
    """
    Walk every element in the SVG tree.  Group path/shape elements by
    (semantic_id, instance_id), keeping only elements whose:
      - semantic-id is one of the 30 target categories
      - instance-id is NOT -1  (instance-id == -1 marks structural
        background objects such as walls and axes)

    Returns a dict mapping (semantic_id, instance_id) -> list of elements.
    """
    instances = defaultdict(list)

    for elem in root.iter():
        sem_raw = elem.get("semantic-id")
        iid_raw = elem.get("instance-id")
        if sem_raw is None or iid_raw is None:
            continue
        try:
            sem_id = int(sem_raw)
            ins_id = int(iid_raw)
        except ValueError:
            continue
        if ins_id == -1:
            continue
        if sem_id not in SEMANTIC_LABELS:
            continue
        instances[(sem_id, ins_id)].append(elem)

    return dict(instances)


def _build_svg_fragment(elements, vbx, vby, vbw, vbh, stroke_override=None):
    """
    Build a minimal, self-contained SVG string containing only the given
    elements on a white background.

    Dataset-specific attributes (semantic-id, instance-id) are stripped.
    If stroke_override is provided, every element's stroke-width is
    replaced with that value (in SVG user units for the given viewBox).
    """
    parts = [
        f'<svg viewBox="{vbx} {vby} {vbw} {vbh}" '
        f'xmlns="{SVG_NS}">',
        f'  <rect x="{vbx}" y="{vby}" width="{vbw}" height="{vbh}" '
        f'fill="white"/>',
    ]
    for el in elements:
        el_copy = etree.fromstring(etree.tostring(el))
        for attr in ("semantic-id", "instance-id"):
            el_copy.attrib.pop(attr, None)
        if stroke_override is not None:
            el_copy.set("stroke-width", f"{stroke_override:.6f}")
        parts.append("  " + etree.tostring(el_copy, encoding="unicode"))
    parts.append("</svg>")
    return "\n".join(parts)


def render_element_400(elements, vw, vh):
    """
    Two-pass render of a single element instance to a 400x400 PNG.

    Pass 1 - bounding-box detection
        Render the element at PROBE_SCALE on the full SVG viewport.
        Detect the axis-aligned bounding box of all non-background pixels
        and convert it back to SVG user units.

    Pass 2 - normalised high-quality render
        Build a new SVG with a tight viewBox around the detected bounding
        box (plus PAD_FRACTION padding).  Override all stroke-widths so
        that strokes appear STROKE_PX_TARGET pixels thick in the output.
        Render at the exact pixel dimensions required to fit the content
        inside (TARGET_PX - 2*CANVAS_MARGIN_PX)^2 while preserving the
        element's aspect ratio.  Place the result centred on a white
        TARGET_PX x TARGET_PX canvas.

    Returns a PIL Image, or None if the element is blank / too small.
    """
    # ── Pass 1: probe render ─────────────────────────────────────────────
    probe_svg = _build_svg_fragment(elements, 0.0, 0.0, vw, vh)
    try:
        probe_png = cairosvg.svg2png(
            bytestring=probe_svg.encode("utf-8"),
            scale=PROBE_SCALE,
        )
    except Exception as exc:
        log.debug("cairosvg probe error: %s", exc)
        return None

    probe_img = Image.open(io.BytesIO(probe_png)).convert("RGB")
    arr = np.asarray(probe_img)

    # Detect content: any pixel with at least one channel below threshold
    mask = np.any(arr < BG_THRESHOLD, axis=2)
    if not mask.any():
        return None  # completely invisible element

    row_idx = np.where(mask.any(axis=1))[0]
    col_idx = np.where(mask.any(axis=0))[0]
    r_min, r_max = int(row_idx[0]),  int(row_idx[-1])
    c_min, c_max = int(col_idx[0]),  int(col_idx[-1])

    # px -> SVG units  (probe canvas width = vw x PROBE_SCALE)
    ppu = probe_img.width / vw          # pixels per SVG unit
    svg_x1 = c_min / ppu
    svg_y1 = r_min / ppu
    svg_x2 = (c_max + 1) / ppu
    svg_y2 = (r_max + 1) / ppu

    bbox_w = svg_x2 - svg_x1
    bbox_h = svg_y2 - svg_y1

    if bbox_w < 0.05 or bbox_h < 0.05:
        return None  # degenerate / invisible

    # ── viewBox with padding ─────────────────────────────────────────────
    pad  = max(bbox_w, bbox_h) * PAD_FRACTION
    vbx  = svg_x1 - pad
    vby  = svg_y1 - pad
    vbw  = bbox_w + 2.0 * pad
    vbh  = bbox_h + 2.0 * pad

    # ── target render dimensions (aspect-ratio preserved) ────────────────
    content_px = TARGET_PX - 2 * CANVAS_MARGIN_PX   # e.g. 370 px
    aspect = vbw / vbh
    if aspect >= 1.0:                                 # wider than tall
        render_w = content_px
        render_h = max(1, int(round(content_px / aspect)))
    else:                                             # taller than wide
        render_h = content_px
        render_w = max(1, int(round(content_px * aspect)))

    # ── normalised stroke ────────────────────────────────────────────────
    # stroke_px = stroke_svg x (render_px / viewbox_svg_units)
    #  =>  stroke_svg = stroke_px x viewbox_svg_units / render_px
    scale_factor = max(render_w / vbw, render_h / vbh)
    stroke_svg   = STROKE_PX_TARGET / scale_factor

    # ── Pass 2: final render ─────────────────────────────────────────────
    final_svg = _build_svg_fragment(
        elements, vbx, vby, vbw, vbh,
        stroke_override=stroke_svg,
    )
    try:
        final_png = cairosvg.svg2png(
            bytestring=final_svg.encode("utf-8"),
            output_width=render_w,
            output_height=render_h,
        )
    except Exception as exc:
        log.debug("cairosvg final render error: %s", exc)
        return None

    content_img = Image.open(io.BytesIO(final_png)).convert("RGB")

    # ── centre on white 400x400 canvas ───────────────────────────────────
    canvas  = Image.new("RGB", (TARGET_PX, TARGET_PX), (255, 255, 255))
    paste_x = (TARGET_PX - render_w) // 2
    paste_y = (TARGET_PX - render_h) // 2
    canvas.paste(content_img, (paste_x, paste_y))
    return canvas


# ════════════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

def ensure_output_dirs(base_dir):
    """
    Create the extraidos/ folder and all 30 category sub-folders under
    base_dir.  Returns a mapping: semantic_id -> output Path.
    """
    extraidos = base_dir / EXTRAIDOS_FOLDER
    extraidos.mkdir(exist_ok=True)
    paths = {}
    for sem_id, (idx, label) in SEMANTIC_LABELS.items():
        folder = extraidos / f"{idx}_{label}"
        folder.mkdir(exist_ok=True)
        paths[sem_id] = folder
    return paths


# ════════════════════════════════════════════════════════════════════════════
# FILE-LEVEL PROCESSING
# ════════════════════════════════════════════════════════════════════════════

def process_svg_file(svg_path, output_dirs):
    """
    Parse one SVG file, extract every target instance, render each to a
    400x400 PNG, and save.  Returns a stats dict for this file.
    """
    stats = {"saved": 0, "blank": 0, "error": 0}
    stem = svg_path.stem   # e.g. "0000-0002"

    try:
        root, vw, vh = parse_svg(svg_path)
    except etree.XMLSyntaxError as exc:
        log.warning("  X XML error in %s: %s", svg_path.name, exc)
        stats["error"] += 1
        return stats

    instances = collect_instances(root)
    if not instances:
        return stats

    for (sem_id, ins_id), elements in instances.items():
        img = render_element_400(elements, vw, vh)
        if img is None:
            stats["blank"] += 1
            continue
        out_name = f"{stem}_ins{ins_id:04d}.png"
        out_path = output_dirs[sem_id] / out_name
        img.save(out_path, "PNG", optimize=True)
        stats["saved"] += 1

    return stats


# ════════════════════════════════════════════════════════════════════════════
# FILESYSTEM TRAVERSAL
# ════════════════════════════════════════════════════════════════════════════

def find_svg_files(dataset_root):
    """
    Recursively collect every .svg file under dataset_root, skipping any
    directory whose name (case-insensitive) is in SKIP_DIR_NAMES.
    """
    result = []
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        # Prune skip-listed directories IN-PLACE so os.walk never descends
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in SKIP_DIR_NAMES
        ]
        for fname in filenames:
            if fname.lower().endswith(".svg"):
                result.append(Path(dirpath) / fname)
    return sorted(result)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    if not DATASET_ROOT.exists():
        log.error("Dataset root not found: %s", DATASET_ROOT)
        sys.exit(1)

    div = "=" * 60
    log.info(div)
    log.info("FloorPlan Element Extractor  v2.0")
    log.info("  Dataset  : %s", DATASET_ROOT)
    log.info("  Output   : 400x400 px PNG, normalised strokes")
    log.info("  Workers  : %d", MAX_WORKERS)
    log.info(div)

    all_svgs = find_svg_files(DATASET_ROOT)
    log.info("Found %d SVG file(s)\n", len(all_svgs))
    if not all_svgs:
        log.warning("No SVG files found — verify DATASET_ROOT.")
        return

    # Pre-create output directories per unique parent folder
    dir_cache = {}
    for svg in all_svgs:
        parent = svg.parent
        if parent not in dir_cache:
            dir_cache[parent] = ensure_output_dirs(parent)
            log.info("[DIR] extraidos/ -> %s",
                     parent.relative_to(DATASET_ROOT))

    # Global counters
    total = {"processed": 0, "skipped": 0, "saved": 0,
             "blank": 0, "errors": 0}

    def _worker(svg_path):
        output_dirs = dir_cache[svg_path.parent]
        return svg_path, process_svg_file(svg_path, output_dirs)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_worker, p): p for p in all_svgs}
        for i, future in enumerate(as_completed(futures), start=1):
            svg_path = futures[future]
            try:
                _, stats = future.result()
            except Exception as exc:
                log.warning("[%d/%d] EXCEPTION %s: %s",
                            i, len(all_svgs), svg_path.name, exc)
                total["errors"] += 1
                continue

            total["processed"] += 1
            if stats["saved"] == 0:
                total["skipped"] += 1
            total["saved"]  += stats["saved"]
            total["blank"]  += stats["blank"]
            total["errors"] += stats["error"]

            if stats["saved"] > 0:
                log.info("[%d/%d] %s -> %d PNG(s)",
                         i, len(all_svgs),
                         svg_path.relative_to(DATASET_ROOT),
                         stats["saved"])
            elif i % 500 == 0:
                log.info("[%d/%d] ... (no targets in last batch)",
                         i, len(all_svgs))

    log.info("\n" + div)
    log.info("COMPLETE")
    log.info("  Processed  : %d SVG files", total["processed"])
    log.info("  No-target  : %d SVG files", total["skipped"])
    log.info("  Saved      : %d PNG images (400x400 px)", total["saved"])
    log.info("  Blank      : %d empty renders",  total["blank"])
    log.info("  Errors     : %d",                total["errors"])
    log.info(div)

    if total["saved"] == 0:
        log.warning(
            "No images saved.  Check that your SVG files contain "
            "semantic-id attributes (FloorPlan dataset format)."
        )


if __name__ == "__main__":
    main()

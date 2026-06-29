#!/usr/bin/env python3
"""Convert Transkribus/PRImA PAGE XML tables into training targets.

Produces, per TableRegion, both target representations we want to experiment with:

  1. detection targets  -- per-cell axis-aligned box in normalized [0,1000] ints
                           plus logical (row, col, rowspan, colspan); rendered as
                           the LocateAnything-style grounding string.
  2. structure targets  -- an HTML <table> (cells placed by row/col, spans honored,
                           empty cells kept) and an OTSL token sequence.

These feed all three planned experiments:
  - logical table VLM (HTML/OTSL target)            <- Exp 1
  - TIPS/DINOv3 + DETR head (detection boxes)       <- Exp 2
  - zero-shot baselines (HTML for TEDS comparison)

Stdlib only (xml.etree) so it runs anywhere. Handles BOTH PAGE flavors:
  - Transkribus: <TableCell row col rowSpan colSpan><Coords points=.../> ...
  - PRImA:       <TextRegion> in <TableRegion> with <Roles><TableCellRole
                 rowIndex columnIndex rowSpan colSpan/></Roles>

Empty cells (no <TextEquiv>) are first-class: structure is geometry-driven, so a
blank cell still yields a box + a grid slot. That is the whole point for
structure-present-but-text-absent handwritten tables.

Usage:
    python data/page_to_targets.py PAGE.xml [--image-root DIR] [--jsonl OUT.jsonl]
    python data/page_to_targets.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Namespace-agnostic XML helpers (PAGE uses 2013-07-15 / 2019-07-15 namespaces)
# --------------------------------------------------------------------------- #
def _local(tag: str) -> str:
    """Strip the {namespace} prefix from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find(elem, name):
    for c in elem:
        if _local(c.tag) == name:
            return c
    return None


def _findall(elem, name):
    return [c for c in elem if _local(c.tag) == name]


def _iter(elem, name):
    for c in elem.iter():
        if _local(c.tag) == name:
            yield c


def _parse_points(points_str: str) -> list[tuple[int, int]]:
    """'x1,y1 x2,y2 ...' -> [(x1,y1), ...]."""
    pts = []
    for tok in points_str.split():
        if "," in tok:
            x, y = tok.split(",")[:2]
            pts.append((int(round(float(x))), int(round(float(y)))))
    return pts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    row: int
    col: int
    rowspan: int
    colspan: int
    points: list[tuple[int, int]]
    text: str = ""

    @property
    def bbox(self):
        return _bbox(self.points)


@dataclass
class Table:
    cells: list[Cell] = field(default_factory=list)

    @property
    def n_rows(self):
        return max((c.row + c.rowspan for c in self.cells), default=0)

    @property
    def n_cols(self):
        return max((c.col + c.colspan for c in self.cells), default=0)


@dataclass
class Page:
    image: str
    width: int
    height: int
    tables: list[Table] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _cell_text(cell_elem) -> str:
    parts = []
    for te in _iter(cell_elem, "TextEquiv"):
        uni = _find(te, "Unicode")
        if uni is not None and uni.text:
            parts.append(uni.text.strip())
    return " ".join(p for p in parts if p)


def _coords_points(elem) -> list[tuple[int, int]]:
    coords = _find(elem, "Coords")
    if coords is not None and coords.get("points"):
        return _parse_points(coords.get("points"))
    return []


def _parse_table_region(region) -> Table:
    table = Table()

    # Transkribus flavor: explicit <TableCell row col rowSpan colSpan>
    tcells = _findall(region, "TableCell")
    if tcells:
        for tc in tcells:
            pts = _coords_points(tc)
            if not pts:
                continue
            table.cells.append(
                Cell(
                    row=int(tc.get("row", 0)),
                    col=int(tc.get("col", 0)),
                    rowspan=int(tc.get("rowSpan", 1) or 1),
                    colspan=int(tc.get("colSpan", 1) or 1),
                    points=pts,
                    text=_cell_text(tc),
                )
            )
        return table

    # PRImA flavor: <TextRegion> with <Roles><TableCellRole rowIndex .../>
    for tr in _findall(region, "TextRegion"):
        roles = _find(tr, "Roles")
        role = _find(roles, "TableCellRole") if roles is not None else None
        if role is None:
            continue
        pts = _coords_points(tr)
        if not pts:
            continue
        table.cells.append(
            Cell(
                row=int(role.get("rowIndex", 0)),
                col=int(role.get("columnIndex", 0)),
                rowspan=int(role.get("rowSpan", 1) or 1),
                colspan=int(role.get("colSpan", 1) or 1),
                points=pts,
                text=_cell_text(tr),
            )
        )
    return table


def parse_page(xml_path_or_string: str, is_string: bool = False) -> Page:
    root = ET.fromstring(xml_path_or_string) if is_string else ET.parse(xml_path_or_string).getroot()
    page_elem = next(_iter(root, "Page"), None)
    if page_elem is None:
        raise ValueError("No <Page> element found - is this PAGE XML?")
    page = Page(
        image=page_elem.get("imageFilename", ""),
        width=int(page_elem.get("imageWidth", 0)),
        height=int(page_elem.get("imageHeight", 0)),
    )
    for region in _iter(page_elem, "TableRegion"):
        table = _parse_table_region(region)
        if table.cells:
            page.tables.append(table)
    return page


# --------------------------------------------------------------------------- #
# Grid building (resolve spans into an occupancy map)
# --------------------------------------------------------------------------- #
def build_grid(table: Table):
    """Return a 2D list grid[r][c] = ('origin'|'cover'|None, cell)."""
    n_r, n_c = table.n_rows, table.n_cols
    grid = [[(None, None) for _ in range(n_c)] for _ in range(n_r)]
    for cell in table.cells:
        for dr in range(cell.rowspan):
            for dc in range(cell.colspan):
                r, c = cell.row + dr, cell.col + dc
                if 0 <= r < n_r and 0 <= c < n_c:
                    kind = "origin" if (dr == 0 and dc == 0) else "cover"
                    grid[r][c] = (kind, cell)
    return grid


# --------------------------------------------------------------------------- #
# Target renderers
# --------------------------------------------------------------------------- #
def to_html(table: Table) -> str:
    grid = build_grid(table)
    rows_html = []
    for r in range(table.n_rows):
        tds = []
        for c in range(table.n_cols):
            kind, cell = grid[r][c]
            if kind == "cover":
                continue
            if kind == "origin":
                attrs = ""
                if cell.rowspan > 1:
                    attrs += f' rowspan="{cell.rowspan}"'
                if cell.colspan > 1:
                    attrs += f' colspan="{cell.colspan}"'
                tds.append(f"<td{attrs}>{cell.text}</td>")
            else:  # genuine gap -> empty cell
                tds.append("<td></td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")
    return "<table>" + "".join(rows_html) + "</table>"


def to_otsl(table: Table) -> str:
    """Minimal OTSL: fcel/ecel (origin), lcel/ucel/xcel (span cover), nl (row end)."""
    grid = build_grid(table)
    toks = []
    for r in range(table.n_rows):
        for c in range(table.n_cols):
            kind, cell = grid[r][c]
            if kind == "origin":
                toks.append("fcel" if cell.text else "ecel")
            elif kind == "cover":
                same_row = cell.row == r
                same_col = cell.col == c
                if not same_row and not same_col:
                    toks.append("xcel")
                elif cell.row == r:
                    toks.append("lcel")  # horizontal span continuation
                else:
                    toks.append("ucel")  # vertical span continuation
            else:
                toks.append("ecel")
        toks.append("nl")
    return " ".join(toks)


def _norm(v: int, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(1000, round(v / size * 1000)))


def to_detection(table: Table, page: Page) -> tuple[str, list[dict]]:
    """LocateAnything-style grounding string + structured list (boxes in [0,1000])."""
    spans, structured = [], []
    for cell in sorted(table.cells, key=lambda x: (x.row, x.col)):
        x1, y1, x2, y2 = cell.bbox
        b = [_norm(x1, page.width), _norm(y1, page.height),
             _norm(x2, page.width), _norm(y2, page.height)]
        spans.append(
            f"<cell r={cell.row} c={cell.col} rs={cell.rowspan} cs={cell.colspan}>"
            f"[{b[0]},{b[1]},{b[2]},{b[3]}]</cell>"
        )
        structured.append({"row": cell.row, "col": cell.col,
                           "rowspan": cell.rowspan, "colspan": cell.colspan,
                           "box_1000": b, "bbox_px": [x1, y1, x2, y2],
                           "text": cell.text})
    return "".join(spans), structured


# --------------------------------------------------------------------------- #
# ShareGPT JSONL samples (one per task variant)
# --------------------------------------------------------------------------- #
def build_samples(page: Page) -> list[dict]:
    samples = []
    for ti, table in enumerate(page.tables):
        det_str, _ = to_detection(table, page)
        img = page.image
        samples.append({
            "task": "detection",
            "image": img,
            "conversations": [
                {"from": "human", "value": "<image-1>\nDetect every table cell and give its row, col, rowspan, colspan."},
                {"from": "gpt", "value": det_str},
            ],
        })
        samples.append({
            "task": "otsl",
            "image": img,
            "conversations": [
                {"from": "human", "value": "<image-1>\nOutput the table structure in OTSL."},
                {"from": "gpt", "value": to_otsl(table)},
            ],
        })
        samples.append({
            "task": "html",
            "image": img,
            "conversations": [
                {"from": "human", "value": "<image-1>\nReconstruct the table as HTML."},
                {"from": "gpt", "value": to_html(table)},
            ],
        })
    return samples


# --------------------------------------------------------------------------- #
# COCO detection targets (Exp B: RF-DETR / TableCenterNet-style training)
# --------------------------------------------------------------------------- #
def page_to_coco_entries(page: Page, image_id: int, ann_id_start: int, category_id: int = 1):
    """(image_dict, [annotation_dicts], next_ann_id) in COCO detection format.

    Each cell -> one annotation: axis-aligned bbox [x,y,w,h], a 4-point
    segmentation quad, and a TableCenterNet-style logic_axis
    [[start_col, end_col, start_row, end_row]] (0-indexed). Empty cells included.
    Feeds RF-DETR's COCO dataloader directly; regress logic_axis with an added
    logical-coordinate head (logic_embed = MLP(d, d, 4, 3)) alongside bbox_embed.
    """
    image = {"id": image_id, "file_name": page.image,
             "width": page.width, "height": page.height}
    anns, aid = [], ann_id_start
    for table in page.tables:
        for cell in table.cells:
            x1, y1, x2, y2 = cell.bbox
            w, h = x2 - x1, y2 - y1
            anns.append({
                "id": aid, "image_id": image_id, "category_id": category_id,
                "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
                "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
                "logic_axis": [[cell.col, cell.col + cell.colspan - 1,
                                cell.row, cell.row + cell.rowspan - 1]],
            })
            aid += 1
    return image, anns, aid


def build_coco(pages: list[Page], category_name: str = "cell") -> dict:
    images, annotations, aid = [], [], 1
    for i, page in enumerate(pages, start=1):
        img, anns, aid = page_to_coco_entries(page, i, aid)
        images.append(img)
        annotations.extend(anns)
    return {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": category_name}], "type": "instances"}


# --------------------------------------------------------------------------- #
# COCO TATR-style targets (Exp B, ZERO fork): row-band + column-band + spanning
# cell as separate classes. Detect them with STOCK RF-DETR, intersect row x col
# to recover the logical grid + spans. No custom head, no dataloader change.
# --------------------------------------------------------------------------- #
def _table_bbox(table: Table):
    bs = [c.bbox for c in table.cells]
    return (min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs))


def _coco_box(aid, image_id, cat, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    return {"id": aid, "image_id": image_id, "category_id": cat,
            "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
            "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]]}


def page_to_coco_tatr_entries(page: Page, image_id: int, ann_id_start: int):
    """Emit row-band (cat 1), column-band (cat 2), spanning-cell (cat 3) boxes."""
    image = {"id": image_id, "file_name": page.image,
             "width": page.width, "height": page.height}
    anns, aid = [], ann_id_start
    for table in page.tables:
        if not table.cells:
            continue
        tx1, ty1, tx2, ty2 = _table_bbox(table)
        for r in range(table.n_rows):
            cov = [c for c in table.cells if c.row <= r < c.row + c.rowspan]
            if cov:
                y1 = min(c.bbox[1] for c in cov); y2 = max(c.bbox[3] for c in cov)
                anns.append(_coco_box(aid, image_id, 1, tx1, y1, tx2, y2)); aid += 1
        for cc in range(table.n_cols):
            cov = [c for c in table.cells if c.col <= cc < c.col + c.colspan]
            if cov:
                x1 = min(c.bbox[0] for c in cov); x2 = max(c.bbox[2] for c in cov)
                anns.append(_coco_box(aid, image_id, 2, x1, ty1, x2, ty2)); aid += 1
        for c in table.cells:
            if c.rowspan > 1 or c.colspan > 1:
                x1, y1, x2, y2 = c.bbox
                anns.append(_coco_box(aid, image_id, 3, x1, y1, x2, y2)); aid += 1
    return image, anns, aid


def build_coco_tatr(pages: list[Page]) -> dict:
    images, annotations, aid = [], [], 1
    for i, page in enumerate(pages, start=1):
        img, anns, aid = page_to_coco_tatr_entries(page, i, aid)
        images.append(img)
        annotations.extend(anns)
    cats = [{"id": 1, "name": "table row"}, {"id": 2, "name": "table column"},
            {"id": 3, "name": "table spanning cell"}]
    return {"images": images, "annotations": annotations, "categories": cats, "type": "instances"}


# --------------------------------------------------------------------------- #
# Self-test (synthetic PAGE XML, both flavors) -- runnable with no external data
# --------------------------------------------------------------------------- #
_TRANSKRIBUS = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
 <Page imageFilename="p1.jpg" imageWidth="1000" imageHeight="500">
  <TableRegion id="t0">
   <Coords points="0,0 1000,0 1000,500 0,500"/>
   <TableCell id="c00" row="0" col="0" rowSpan="1" colSpan="2">
     <Coords points="0,0 500,0 500,100 0,100"/>
     <TextLine><TextEquiv><Unicode>Header</Unicode></TextEquiv></TextLine>
   </TableCell>
   <TableCell id="c02" row="0" col="2" rowSpan="1" colSpan="1">
     <Coords points="500,0 1000,0 1000,100 500,100"/>
   </TableCell>
   <TableCell id="c10" row="1" col="0" rowSpan="1" colSpan="1">
     <Coords points="0,100 250,100 250,200 0,200"/>
     <TextLine><TextEquiv><Unicode>Stockholm</Unicode></TextEquiv></TextLine>
   </TableCell>
   <TableCell id="c11" row="1" col="1" rowSpan="1" colSpan="1">
     <Coords points="250,100 500,100 500,200 250,200"/>
   </TableCell>
   <TableCell id="c12" row="1" col="2" rowSpan="1" colSpan="1">
     <Coords points="500,100 1000,100 1000,200 500,200"/>
   </TableCell>
  </TableRegion>
 </Page>
</PcGts>"""

_PRIMA = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
 <Page imageFilename="p2.jpg" imageWidth="800" imageHeight="400">
  <TableRegion id="t0">
   <Coords points="0,0 800,0 800,400 0,400"/>
   <TextRegion id="r0">
     <Coords points="0,0 400,0 400,100 0,100"/>
     <Roles><TableCellRole rowIndex="0" columnIndex="0" rowSpan="2" colSpan="1"/></Roles>
     <TextLine><TextEquiv><Unicode>Key</Unicode></TextEquiv></TextLine>
   </TextRegion>
   <TextRegion id="r1">
     <Coords points="400,0 800,0 800,100 400,100"/>
     <Roles><TableCellRole rowIndex="0" columnIndex="1" rowSpan="1" colSpan="1"/></Roles>
   </TextRegion>
   <TextRegion id="r2">
     <Coords points="400,100 800,100 800,200 400,200"/>
     <Roles><TableCellRole rowIndex="1" columnIndex="1" rowSpan="1" colSpan="1"/></Roles>
   </TextRegion>
  </TableRegion>
 </Page>
</PcGts>"""


def _self_test():
    for name, xml in [("Transkribus flavor", _TRANSKRIBUS), ("PRImA flavor", _PRIMA)]:
        print(f"\n========== {name} ==========")
        page = parse_page(xml, is_string=True)
        t = page.tables[0]
        print(f"image={page.image} {page.width}x{page.height}  cells={len(t.cells)}  grid={t.n_rows}x{t.n_cols}")
        print("HTML  :", to_html(t))
        print("OTSL  :", to_otsl(t))
        det, structured = to_detection(t, page)
        print("DETECT:", det)
        print("box[0]:", structured[0])

    # assertions
    p1 = parse_page(_TRANSKRIBUS, is_string=True)
    t1 = p1.tables[0]
    assert len(t1.cells) == 5, t1.cells
    assert (t1.n_rows, t1.n_cols) == (2, 3), (t1.n_rows, t1.n_cols)
    assert 'colspan="2"' in to_html(t1)              # span preserved
    assert to_otsl(t1).split(" nl ")[0] == "fcel lcel ecel"  # row0: origin, h-span cover, empty
    p2 = parse_page(_PRIMA, is_string=True)
    t2 = p2.tables[0]
    assert (t2.n_rows, t2.n_cols) == (2, 2), (t2.n_rows, t2.n_cols)
    assert 'rowspan="2"' in to_html(t2)              # PRImA vertical span
    assert "ucel" in to_otsl(t2)                     # vertical span cover token
    # empty-cell-without-text still produces a box and a grid slot
    det2, s2 = to_detection(t2, p2)
    assert any(c["text"] == "" for c in s2)
    # COCO emitter (RF-DETR / TableCenterNet target)
    coco = build_coco([p1, p2])
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 8, coco
    a0 = coco["annotations"][0]
    assert a0["bbox"] == [0, 0, 500, 100] and a0["logic_axis"] == [[0, 1, 0, 0]], a0
    assert len(a0["segmentation"][0]) == 8  # 4-point quad
    print("COCO :", len(coco["images"]), "images,", len(coco["annotations"]), "cell anns; ann[0].logic_axis", a0["logic_axis"])
    # COCO TATR emitter (row/col/spanning-cell, zero-fork RF-DETR)
    tatr = build_coco_tatr([p1, p2])
    cats = {c["category_id"] for c in tatr["annotations"]}
    n_rows = sum(c["category_id"] == 1 for c in tatr["annotations"])
    n_cols = sum(c["category_id"] == 2 for c in tatr["annotations"])
    n_span = sum(c["category_id"] == 3 for c in tatr["annotations"])
    # transkribus: 2 rows + 3 cols + 1 span(c00) ; prima: 2 rows + 2 cols + 1 span(r0)
    assert (n_rows, n_cols, n_span) == (4, 5, 2), (n_rows, n_cols, n_span)
    print("TATR :", n_rows, "row +", n_cols, "col +", n_span, "spanning-cell boxes (classes", sorted(cats), ")")
    print("\nAll self-tests passed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("page_xml", nargs="*", help="One or more Transkribus/PRImA PAGE XML files")
    ap.add_argument("--jsonl", help="Append ShareGPT samples (Exp A: LocateAnything) to this JSONL file")
    ap.add_argument("--coco", help="Write COCO detection json (Exp B: RF-DETR/TIPS) over all inputs")
    ap.add_argument("--coco-mode", choices=["cells", "tatr"], default="cells",
                    help="cells = one box/cell + logic_axis (needs logical head); "
                         "tatr = row/column/spanning-cell boxes (stock RF-DETR, zero fork)")
    ap.add_argument("--image-root", default="", help="Prefix prepended to image paths")
    ap.add_argument("--self-test", action="store_true", help="Run on synthetic PAGE XML and exit")
    args = ap.parse_args()

    if args.self_test or not args.page_xml:
        _self_test()
        return

    pages = []
    for path in args.page_xml:
        page = parse_page(path)
        if args.image_root and page.image:
            page.image = f"{args.image_root.rstrip('/')}/{page.image}"
        pages.append(page)

    if args.coco:
        coco = build_coco_tatr(pages) if args.coco_mode == "tatr" else build_coco(pages)
        with open(args.coco, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        print(f"# wrote {args.coco} ({args.coco_mode}): {len(coco['images'])} images, "
              f"{len(coco['annotations'])} anns", file=sys.stderr)
        return

    for page in pages:
        samples = build_samples(page)
        print(f"# {page.image}: {len(page.tables)} table(s), {len(samples)} sample(s)", file=sys.stderr)
        for s in samples:
            line = json.dumps(s, ensure_ascii=False)
            if args.jsonl:
                with open(args.jsonl, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            else:
                print(line)


if __name__ == "__main__":
    main()

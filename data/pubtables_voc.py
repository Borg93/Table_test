#!/usr/bin/env python3
"""Read PubTables / Table-Transformer (TATR) PASCAL VOC annotations -> our COCO.

This is the EXACT format inside kensho/PubTables-v2 (and PubTables-1M) cropped-table
structure data: one cropped-table image + one PASCAL VOC `.xml` per image, each
`<object>` carrying a `<name>` (class) and `<bndbox>` (xmin,ymin,xmax,ymax in px).

Class set (from microsoft/table-transformer `src/main.py` structure `class_map`):
    table, table column, table row, table column header,
    table projected row header, table spanning cell        (+ "no object" = bg)

We emit the same COCO json that `tips_detr/dataset.py` already consumes, so you can:
  * pretrain TipsDETR (Exp B) on PubTables-v2 (135k printed tables, CDLA-permissive)
  * then fine-tune on your handwritten Transkribus PAGE data (data/page_to_targets.py)
Both then share one label space. Stdlib only.

Usage:
    python data/pubtables_voc.py VOC_DIR [--images-dir DIR] --coco out.json
    python data/pubtables_voc.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# 0-indexed structure classes (TATR), excluding the background "no object".
TATR6 = [
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
]
NAME_TO_ID = {n: i for i, n in enumerate(TATR6)}


def read_pascal_voc(xml_path: str) -> tuple[list[list[float]], list[str], tuple[int, int], str]:
    """-> (bboxes[xmin,ymin,xmax,ymax], label names, (width,height), image filename).

    Mirrors table-transformer's `read_pascal_voc`: iterate <object>, read <name>
    and <bndbox>.
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    filename = root.findtext("filename", "") or (Path(xml_path).stem + ".jpg")

    bboxes, labels = [], []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(name)
    return bboxes, labels, (width, height), filename


def build_coco_from_voc(xml_paths: list[str], keep_table_box: bool = True) -> dict:
    """All VOC files -> one COCO json with the 6 TATR categories (ids 1..6)."""
    images, annotations, aid = [], [], 1
    for image_id, xp in enumerate(xml_paths, start=1):
        bboxes, labels, (w, h), filename = read_pascal_voc(xp)
        images.append({"id": image_id, "file_name": filename, "width": w, "height": h})
        for bbox, name in zip(bboxes, labels):
            if name not in NAME_TO_ID:
                continue
            if name == "table" and not keep_table_box:
                continue
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            annotations.append({
                "id": aid, "image_id": image_id,
                "category_id": NAME_TO_ID[name] + 1,            # COCO ids are 1-based
                "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0,
                "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
            })
            aid += 1
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(TATR6)]
    return {"images": images, "annotations": annotations,
            "categories": categories, "type": "instances"}


# --------------------------------------------------------------------------- #
_SAMPLE_VOC = """<?xml version="1.0"?>
<annotation>
  <filename>PMC0000001_table_0.jpg</filename>
  <size><width>600</width><height>400</height><depth>3</depth></size>
  <object><name>table</name>
    <bndbox><xmin>5</xmin><ymin>5</ymin><xmax>595</xmax><ymax>395</ymax></bndbox></object>
  <object><name>table row</name>
    <bndbox><xmin>5</xmin><ymin>5</ymin><xmax>595</xmax><ymax>55</ymax></bndbox></object>
  <object><name>table row</name>
    <bndbox><xmin>5</xmin><ymin>55</ymin><xmax>595</xmax><ymax>200</ymax></bndbox></object>
  <object><name>table column</name>
    <bndbox><xmin>5</xmin><ymin>5</ymin><xmax>300</xmax><ymax>395</ymax></bndbox></object>
  <object><name>table column</name>
    <bndbox><xmin>300</xmin><ymin>5</ymin><xmax>595</xmax><ymax>395</ymax></bndbox></object>
  <object><name>table column header</name>
    <bndbox><xmin>5</xmin><ymin>5</ymin><xmax>595</xmax><ymax>55</ymax></bndbox></object>
  <object><name>table spanning cell</name>
    <bndbox><xmin>5</xmin><ymin>5</ymin><xmax>595</xmax><ymax>55</ymax></bndbox></object>
</annotation>"""


def _self_test():
    tmp = Path(__file__).resolve().parent / "_voc_selftest.xml"
    tmp.write_text(_SAMPLE_VOC)
    try:
        bboxes, labels, size, fn = read_pascal_voc(str(tmp))
        print("file:", fn, "size:", size, "objects:", len(bboxes))
        print("labels:", labels)
        assert size == (600, 400) and fn == "PMC0000001_table_0.jpg"
        assert labels.count("table row") == 2 and labels.count("table column") == 2
        assert "table column header" in labels and "table spanning cell" in labels
        coco = build_coco_from_voc([str(tmp)])
        assert len(coco["images"]) == 1 and len(coco["annotations"]) == 7
        assert [c["name"] for c in coco["categories"]] == TATR6
        cid = {a["category_id"] for a in coco["annotations"]}
        assert cid == {1, 2, 3, 4, 6}, cid           # table, col, row, col-header, spanning
        a0 = coco["annotations"][0]
        assert a0["bbox"] == [5.0, 5.0, 590.0, 390.0]  # table: xywh from VOC xyxy
        coco2 = build_coco_from_voc([str(tmp)], keep_table_box=False)
        assert len(coco2["annotations"]) == 6          # 'table' box dropped
        print("COCO:", len(coco["annotations"]), "anns over", len(TATR6), "classes; ids", sorted(cid))
        print("\nAll self-tests passed.")
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("voc_dir", nargs="?", help="Directory of PASCAL VOC .xml files")
    ap.add_argument("--coco", help="Write merged COCO json here")
    ap.add_argument("--no-table-box", action="store_true",
                    help="Drop the whole-table 'table' box (keep only structure)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test or not args.voc_dir:
        _self_test()
        return

    xmls = sorted(str(p) for p in Path(args.voc_dir).glob("*.xml"))
    coco = build_coco_from_voc(xmls, keep_table_box=not args.no_table_box)
    out = args.coco or "pubtables_coco.json"
    Path(out).write_text(json.dumps(coco, ensure_ascii=False))
    print(f"# wrote {out}: {len(coco['images'])} images, {len(coco['annotations'])} anns, "
          f"{len(coco['categories'])} classes", file=sys.stderr)


if __name__ == "__main__":
    main()

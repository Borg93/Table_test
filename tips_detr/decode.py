#!/usr/bin/env python3
"""Decode detected boxes -> logical table grid -> HTML (the actual deliverable).

The model (or LocateAnything) outputs labeled boxes: `table row`, `table column`,
`table spanning cell`. This turns them into the coherent cell grid:

    rows (sorted top->bottom) x columns (sorted left->right) = cell at (r, c);
    a spanning-cell box that *overlaps* several (r, c) slots merges them (rowspan/colspan).

Robustness (hardened after adversarial review):
  - near-duplicate row/column bands (detector emits several boxes per physical line)
    are merged so each physical row/column is one grid line;
  - span membership uses box *overlap* (a fraction of the band), not band-center
    inclusion, so a tight or offset span box still recovers the right rowspan/colspan;
  - overlapping/nested spans can't corrupt the grid (a conflicting span is dropped);
  - a missing dimension (rows-but-no-cols, or vice versa) degrades gracefully.

Produces structure-only HTML (cells carry geometry + span, not text) so it can be
scored against ground truth with eval/eval_teds.py. Plain lists -> no torch/GPU needed.
"""
from __future__ import annotations

ROW, COL, SPAN = "table row", "table column", "table spanning cell"


def _cx(b):
    return (b[0] + b[2]) / 2.0


def _cy(b):
    return (b[1] + b[3]) / 2.0


def _ov(a0, a1, b0, b1):
    """1-D overlap length of [a0,a1] and [b0,b1]."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _merge_bands(bands, vertical, merge_frac=0.5):
    """Collapse near-duplicate bands (NMS-style) so each physical line is one band.

    `bands` must be pre-sorted along their axis. Two consecutive bands are merged
    (unioned) when their 1-D overlap exceeds `merge_frac` of the smaller extent.
    """
    lo, hi = (1, 3) if vertical else (0, 2)
    merged: list[tuple] = []
    for b in bands:
        if merged:
            p = merged[-1]
            ext = min(p[hi] - p[lo], b[hi] - b[lo])
            if ext > 0 and _ov(p[lo], p[hi], b[lo], b[hi]) >= merge_frac * ext:
                merged[-1] = (min(p[0], b[0]), min(p[1], b[1]),
                              max(p[2], b[2]), max(p[3], b[3]))
                continue
        merged.append(tuple(b))
    return merged


def decode_grid(labels: list[str], boxes: list[tuple], *, cover_frac: float = 0.3):
    """labels/boxes (xyxy) -> (n_rows, n_cols, origin, covered, cell_boxes).

    origin[(r,c)] = (rowspan, colspan) for a cell's top-left (1x1 cells are NOT in
    origin). covered = (r,c) slots hidden under a span. cell_boxes[(r,c)] = xyxy.
    """
    rows = _merge_bands(sorted((b for l, b in zip(labels, boxes) if l == ROW), key=_cy), True)
    cols = _merge_bands(sorted((b for l, b in zip(labels, boxes) if l == COL), key=_cx), False)
    spans = [b for l, b in zip(labels, boxes) if l == SPAN]
    R, C = len(rows), len(cols)

    cell_boxes = {}
    for r in range(R):
        for c in range(C):
            cell_boxes[(r, c)] = (cols[c][0], rows[r][1], cols[c][2], rows[r][3])

    origin: dict[tuple, tuple] = {}
    covered: set = set()
    # largest spans first -> deterministic, and a smaller nested span yields to it.
    for sp in sorted(spans, key=_area, reverse=True):
        rs = [r for r in range(R)
              if (rows[r][3] - rows[r][1]) > 0
              and _ov(rows[r][1], rows[r][3], sp[1], sp[3]) >= cover_frac * (rows[r][3] - rows[r][1])]
        cs = [c for c in range(C)
              if (cols[c][2] - cols[c][0]) > 0
              and _ov(cols[c][0], cols[c][2], sp[0], sp[2]) >= cover_frac * (cols[c][2] - cols[c][0])]
        if not rs or not cs:
            continue
        r0, c0 = min(rs), min(cs)
        rspan, cspan = max(rs) - r0 + 1, max(cs) - c0 + 1
        if rspan == 1 and cspan == 1:
            continue
        block = [(r, c) for r in range(r0, r0 + rspan) for c in range(c0, c0 + cspan)]
        # don't let an overlapping/nested span corrupt cells another span already owns
        if any((r, c) in covered or ((r, c) in origin and (r, c) != (r0, c0)) for r, c in block):
            continue
        origin[(r0, c0)] = (rspan, cspan)
        for r, c in block:
            if (r, c) != (r0, c0):
                covered.add((r, c))
    return R, C, origin, covered, cell_boxes


def grid_to_html(R: int, C: int, origin: dict, covered: set) -> str:
    """Structure-only HTML (empty cells); spans honored via rowspan/colspan.

    origin is checked before covered, so a span's anchor is always emitted.
    Degrades gracefully when one dimension is missing.
    """
    if R > 0 and C == 0:                      # rows recovered but no columns
        return "<table>" + "".join("<tr><td></td></tr>" for _ in range(R)) + "</table>"
    if R == 0 and C > 0:                       # columns recovered but no rows
        return "<table><tr>" + "".join("<td></td>" for _ in range(C)) + "</tr></table>"

    rows_html = []
    for r in range(R):
        tds = []
        for c in range(C):
            if (r, c) in origin:
                rspan, cspan = origin[(r, c)]
                attrs = (f' rowspan="{rspan}"' if rspan > 1 else "") + \
                        (f' colspan="{cspan}"' if cspan > 1 else "")
                tds.append(f"<td{attrs}></td>")
            elif (r, c) in covered:
                continue
            else:
                tds.append("<td></td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")
    return "<table>" + "".join(rows_html) + "</table>"


def detections_to_html(labels: list[str], boxes: list[tuple]) -> str:
    R, C, origin, covered, _ = decode_grid(labels, boxes)
    return grid_to_html(R, C, origin, covered)


def postprocess_to_html(result: dict, class_names: list[str], score_thresh: float = 0.5) -> str:
    """Bridge from model.post_process output (scores/labels/boxes tensors) -> HTML."""
    labels, boxes = [], []
    for s, l, b in zip(result["scores"].tolist(), result["labels"].tolist(),
                       result["boxes"].tolist()):
        if s < score_thresh:
            continue
        labels.append(class_names[int(l)])
        boxes.append(tuple(b))
    return detections_to_html(labels, boxes)


def _self_test():
    # 2 rows x 3 cols; header spans columns 0-1 in row 0.
    labels = [ROW, ROW, COL, COL, COL, SPAN]
    boxes = [(0, 0, 900, 100), (0, 100, 900, 200),
             (0, 0, 300, 200), (300, 0, 600, 200), (600, 0, 900, 200),
             (0, 0, 600, 100)]
    R, C, origin, covered, cells = decode_grid(labels, boxes)
    assert (R, C) == (2, 3), (R, C)
    assert origin == {(0, 0): (1, 2)}, origin
    assert covered == {(0, 1)}, covered
    html = grid_to_html(R, C, origin, covered)
    assert html == ('<table><tr><td colspan="2"></td><td></td></tr>'
                    '<tr><td></td><td></td><td></td></tr></table>'), html
    # order-invariance
    perm = list(reversed(range(len(labels))))
    assert detections_to_html([labels[i] for i in perm], [boxes[i] for i in perm]) == html

    # duplicate bands (detector NMS overlap) must collapse to one line
    dup = detections_to_html([ROW, ROW, COL, COL],
                             [(0, 0, 200, 100), (0, 2, 200, 102), (0, 0, 100, 100), (100, 0, 200, 100)])
    assert dup == "<table><tr><td></td><td></td></tr></table>", dup   # 1 row x 2 cols, not 2x2

    # rowspan recovered from an overlap-covering span box (the round-trip case)
    rs_html = detections_to_html(
        [ROW, ROW, COL, COL, SPAN],
        [(0, 0, 800, 100), (0, 100, 800, 200), (0, 0, 400, 200), (400, 0, 800, 200), (0, 0, 400, 200)])
    assert rs_html == ('<table><tr><td rowspan="2"></td><td></td></tr>'
                       '<tr><td></td></tr></table>'), rs_html

    # overlapping spans cannot corrupt the grid: a conflicting span is dropped,
    # every row still has full logical width.
    R2, C2, o2, cov2, _ = decode_grid(
        [ROW, COL, COL, COL, SPAN, SPAN],
        [(0, 0, 300, 100), (0, 0, 100, 100), (100, 0, 200, 100), (200, 0, 300, 100),
         (0, 0, 200, 100), (100, 0, 300, 100)])
    for r in range(R2):
        width = sum(o2.get((r, c), (1, 1))[1] for c in range(C2) if (r, c) not in cov2)
        assert width == C2, (r, width, C2, o2, cov2)

    # missing dimension degrades gracefully
    assert detections_to_html([COL, COL], [(0, 0, 100, 100), (100, 0, 200, 100)]) == \
        "<table><tr><td></td><td></td></tr></table>"
    assert detections_to_html([], []) == "<table></table>"
    print("All decode self-tests passed.")


if __name__ == "__main__":
    _self_test()

#!/usr/bin/env python3
"""Decode detected boxes -> logical table grid -> HTML (the actual deliverable).

The model (or LocateAnything) outputs labeled boxes: `table row`, `table column`,
`table spanning cell`. This turns them into the coherent cell grid:

    rows (sorted top->bottom) x columns (sorted left->right) = cell at (r, c);
    a spanning-cell box that covers several (r, c) slots merges them (rowspan/colspan).

Produces structure-only HTML (cells carry geometry + span, not text) so it can be
scored against ground truth with eval/eval_teds.py (structure-only TEDS). Works on
plain lists -> no torch/GPU needed, fully unit-testable.
"""
from __future__ import annotations

ROW, COL, SPAN = "table row", "table column", "table spanning cell"


def _cx(b):  # x center
    return (b[0] + b[2]) / 2.0


def _cy(b):  # y center
    return (b[1] + b[3]) / 2.0


def decode_grid(labels: list[str], boxes: list[tuple], *, tol: float = 0.0):
    """labels/boxes (xyxy) -> (n_rows, n_cols, origin, covered, cell_boxes).

    origin[(r,c)] = (rowspan, colspan) for the top-left of a merged region (or any
    1x1 cell). covered = set of (r,c) hidden under a span. cell_boxes[(r,c)] = xyxy.
    """
    rows = sorted((b for l, b in zip(labels, boxes) if l == ROW), key=_cy)
    cols = sorted((b for l, b in zip(labels, boxes) if l == COL), key=_cx)
    spans = [b for l, b in zip(labels, boxes) if l == SPAN]
    R, C = len(rows), len(cols)

    cell_boxes = {}
    for r in range(R):
        for c in range(C):
            cell_boxes[(r, c)] = (cols[c][0], rows[r][1], cols[c][2], rows[r][3])

    origin: dict[tuple, tuple] = {}
    covered: set = set()
    for sp in spans:
        rs = [r for r in range(R) if sp[1] - tol <= _cy(rows[r]) <= sp[3] + tol]
        cs = [c for c in range(C) if sp[0] - tol <= _cx(cols[c]) <= sp[2] + tol]
        if not rs or not cs:
            continue
        r0, c0 = min(rs), min(cs)
        rspan, cspan = max(rs) - r0 + 1, max(cs) - c0 + 1
        if rspan == 1 and cspan == 1:
            continue
        origin[(r0, c0)] = (rspan, cspan)
        for r in range(r0, r0 + rspan):
            for c in range(c0, c0 + cspan):
                if (r, c) != (r0, c0):
                    covered.add((r, c))
    return R, C, origin, covered, cell_boxes


def grid_to_html(R: int, C: int, origin: dict, covered: set) -> str:
    """Structure-only HTML (empty cells); spans honored via rowspan/colspan."""
    rows_html = []
    for r in range(R):
        tds = []
        for c in range(C):
            if (r, c) in covered:
                continue
            rspan, cspan = origin.get((r, c), (1, 1))
            attrs = (f' rowspan="{rspan}"' if rspan > 1 else "") + \
                    (f' colspan="{cspan}"' if cspan > 1 else "")
            tds.append(f"<td{attrs}></td>")
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
    assert cells[(1, 2)] == (600, 100, 900, 200), cells[(1, 2)]
    html = grid_to_html(R, C, origin, covered)
    print("HTML:", html)
    assert html == ('<table><tr><td colspan="2"></td><td></td></tr>'
                    '<tr><td></td><td></td><td></td></tr></table>'), html
    # unordered input still sorts into the right grid
    import random
    idx = list(range(len(labels)))
    perm = idx[::-1]
    html2 = detections_to_html([labels[i] for i in perm], [boxes[i] for i in perm])
    assert html2 == html, ("order-invariance failed", html2)
    print("All decode self-tests passed.")


if __name__ == "__main__":
    _self_test()

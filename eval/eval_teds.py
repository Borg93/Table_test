#!/usr/bin/env python3
"""TEDS (Tree-Edit-Distance-based Similarity) for table structure.

This is the metric that matches the real goal: it scores whether the predicted
*logical* table (cells, rows, cols, spans) matches the ground truth -- not box IoU.
Compare a model's predicted HTML (LocateAnything, or our own model) against
the GT HTML from ../data/page_to_targets.py.

  TEDS = 1 - EditDistance(pred_tree, gt_tree) / max(|pred_tree|, |gt_tree|)

Use --structure-only to ignore cell text (the right mode for mostly-empty
handwritten grids where you care about structure, not transcription).

Requires `pip install apted` (the canonical TEDS backend, same as PubTabNet).
"""
from __future__ import annotations

import argparse
import sys
from html.parser import HTMLParser


class _Node:
    __slots__ = ("tag", "text", "children")

    def __init__(self, tag, text=""):
        self.tag = tag
        self.text = text
        self.children = []


class _TableTreeBuilder(HTMLParser):
    """Build a tree of table/tr/td(+colspan/rowspan) from an HTML table string."""

    def __init__(self, structure_only: bool):
        super().__init__()
        self.structure_only = structure_only
        self.root = _Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        if tag in ("table", "tr", "td", "th"):
            label = tag
            if tag in ("td", "th"):
                d = dict(attrs)
                label = f"{tag} r{d.get('rowspan', '1')} c{d.get('colspan', '1')}"
            node = _Node(label)
            self.stack[-1].children.append(node)
            self.stack.append(node)

    def handle_endtag(self, tag):
        if tag in ("table", "tr", "td", "th") and len(self.stack) > 1:
            self.stack.pop()

    def handle_data(self, data):
        if not self.structure_only and self.stack[-1].tag.startswith(("td", "th")):
            self.stack[-1].text += data.strip()


def _to_apted_tree(node, structure_only: bool):
    """Render a node tree to apted's bracket format: {label{child}...}."""
    label = node.tag if structure_only or not node.text else f"{node.tag}:{node.text}"
    kids = "".join(_to_apted_tree(c, structure_only) for c in node.children)
    return f"{{{label}{kids}}}"


def _count(node):
    return 1 + sum(_count(c) for c in node.children)


def teds(pred_html: str, gt_html: str, structure_only: bool = True) -> float:
    try:
        from apted import APTED
        from apted.helpers import Tree
    except ImportError:
        sys.exit("TEDS needs apted: pip install apted")

    pb = _TableTreeBuilder(structure_only); pb.feed(pred_html)
    gb = _TableTreeBuilder(structure_only); gb.feed(gt_html)
    pred_s = _to_apted_tree(pb.root, structure_only)
    gt_s = _to_apted_tree(gb.root, structure_only)
    dist = APTED(Tree.from_text(pred_s), Tree.from_text(gt_s)).compute_edit_distance()
    n = max(_count(pb.root), _count(gb.root))
    return 1.0 - dist / n if n else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="Predicted HTML file")
    ap.add_argument("--gt", help="Ground-truth HTML file")
    ap.add_argument("--with-text", action="store_true", help="Include cell text (default: structure only)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        a = "<table><tr><td colspan='2'>H</td></tr><tr><td>a</td><td>b</td></tr></table>"
        print("TEDS(self)      =", round(teds(a, a), 4))
        b = "<table><tr><td>H</td><td></td></tr><tr><td>a</td><td>b</td></tr></table>"
        print("TEDS(span diff) =", round(teds(a, b), 4))
        return

    if not (args.pred and args.gt):
        ap.error("provide --pred and --gt (or --self-test)")
    score = teds(open(args.pred).read(), open(args.gt).read(), structure_only=not args.with_text)
    print(f"TEDS = {score:.4f}")


if __name__ == "__main__":
    main()

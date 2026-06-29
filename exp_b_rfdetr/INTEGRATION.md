# Exp B — RF-DETR + TIPS v2, staged integration

Grounded in the actual `rfdetr` source (v1.7.x). Goal: detect table structure on
handwritten pages with a pretrained-ViT DETR, optionally swapping the DINOv2
backbone for TIPS v2. Runs on the H100 box (`pip install rfdetr`); not in CI.

The work is staged so you get a result **before** any fork, and only fork when a
stage justifies it.

---

## B0 — stock RF-DETR, ZERO fork (do this first)

RF-DETR consumes Roboflow-COCO and one-class detection works out of the box.

### B0a — cell boxes
```bash
# one COCO json per split, Roboflow layout: <ds>/{train,valid,test}/_annotations.coco.json
python data/page_to_targets.py train/*.xml --coco dataset/train/_annotations.coco.json --coco-mode cells
```
> `--coco-mode cells` also writes `logic_axis`, which **stock** RF-DETR ignores
> (`ConvertCoco` reads only `boxes/labels/masks/keypoints`). Fine for B0 — you
> recover row/col by sorting the detected cell boxes (geometric reconstruction).

### B0b — TATR-style logical structure, still zero fork (recommended)
```bash
python data/page_to_targets.py train/*.xml --coco dataset/train/_annotations.coco.json --coco-mode tatr
```
Emits 3 classes — `table row`, `table column`, `table spanning cell`. Train stock
RF-DETR (`num_classes=3`), then **intersect** detected row-bands × column-bands to
get the cell grid with logical (row, col) indices; spanning-cell boxes fix merges.
This gives logical structure with **no custom head and no dataloader edit.**

### Train (either B0a/B0b)
```python
from rfdetr import RFDETRMedium          # DINOv2 ViT-S basis, patch 16, res 576
m = RFDETRMedium()                        # or RFDETRBase: patch 14, res 560
m.train(dataset_dir="dataset", epochs=100, batch_size=4,
        num_classes=3)                    # 1 for cells, 3 for tatr
```
Or the Lightning CLI: `rfdetr fit --config configs/rfdetr_base.yaml --model.model_config.init_args.num_classes 3 --model.train_config.init_args.dataset_dir dataset`.

**Resolution constraint** (`RFDETR*` docstrings): square `resolution` must be
divisible by `patch_size * num_windows` (Base: 14·4=56 → 560; Medium: 16·2=32 →
576). Your ~4000px pages: **tile** to a multiple of that, or raise `resolution`
(e.g. 1568 = 56·28) and accept the memory cost. Validate thin-column recall here.

---

## B1 — swap DINOv2 → TIPS v2 (small fork)

Both are patch-14 register ViTs, so the projector/transformer/heads are unchanged.
Three edits:

1. **`config.py`** — extend the encoder enum and add a config:
   ```python
   EncoderName = Literal[..., "tipsv2_windowed_large"]   # add ours
   class RFDETRTipsLargeConfig(RFDETRBaseConfig):
       encoder = "tipsv2_windowed_large"
       patch_size = 14
       num_windows = 4
       out_feature_indexes = [5, 11, 17, 23]   # ViT-L/24 taps
       hidden_dim = 256
       resolution = 1568                        # 56 * 28
       positional_encoding_size = 1568 // 14
       pretrain_weights = None                  # TIPS weights load inside the backbone
   ```
2. **`models/backbone/backbone.py`** — `Backbone.__init__` currently asserts
   `name_parts[0] == "dinov2"`. Branch on `tipsv2`:
   ```python
   if name.startswith("tipsv2"):
       self.encoder = TipsV2Backbone(size="L", out_feature_indexes=out_feature_indexes,
                                     freeze=freeze_encoder)
   else:
       self.encoder = DinoV2(...)   # unchanged
   self.projector = MultiScaleProjector(in_channels=self.encoder.num_channels,
                                        out_channels=hidden_dim,
                                        scale_factors=...)   # unchanged
   ```
3. **`TipsV2Backbone`** — mirror `DinoV2`'s contract: a `forward(pixel_values)`
   that returns a **list of feature maps** `[B, C, H/14, W/14]` at
   `out_feature_indexes`, and `.num_channels` (= embed_dim per tap). Use
   `models/tips_encoder.py` (the TIPS API you provided) for the actual ViT; add
   windowed attention for high res (port `dinov2_with_windowed_attn`, or run tiled
   first to defer it). Skeleton: `exp_b_rfdetr/tips_backbone.py`.

> Note the `num_channels`-adapter path in `inference.py` pokes
> `backbone[0].encoder.encoder.embeddings.patch_embeddings.projection` (HF-DINOv2
> internals). Keep `num_channels=3` for TIPS so that branch is skipped, or expose
> an equivalent patch-embed attribute.

After this, B0b (TATR, 3 classes) runs unchanged on the TIPS backbone.

---

## B2 — per-cell logical head (additive, only if B0b is not enough)

If row×column intersection can't resolve your spans/merges, regress logical
coordinates per query (à la TableCenterNet/LORE). Four edits:

1. **`models/heads/detection.py`** — add to `DetectionHead`:
   ```python
   self.logic_embed = MLP(hidden_dim, hidden_dim, 4, 3)     # (sc, ec, sr, er)
   # forward: return outputs_class, outputs_coord, self.logic_embed(hs)
   ```
2. **`models/lwdetr.py`** — thread `outputs_logic` into the output dict
   (`out["pred_logic"]`) and the aux-loss list.
3. **`datasets/coco.py` `ConvertCoco.__call__`** — parse our field:
   ```python
   if "logic_axis" in anno[0]:
       target["logic"] = torch.as_tensor(
           [a["logic_axis"][0] for a in anno], dtype=torch.float32)[keep]
   ```
   (use `--coco-mode cells`, which writes `logic_axis`.)
4. **`models/criterion.py`** — add an L1 term between matched
   `pred_logic` and `target["logic"]` (matcher unchanged — match on class+box).

---

## Licensing (all permissive)
RF-DETR base/Apache models = Apache-2.0 (avoid `rfdetr_plus` XL/2XL = PML 1.0).
TIPS v2 code Apache-2.0, **weights CC-BY-4.0** (attribute Google). Keep
`num_classes`/encoder overrides off the published RF-DETR checkpoints unless you
intend to train from scratch (see `_warn_pretrain_compatibility` in `config.py`).

## Recommended order
B0b (TATR, stock, zero fork) → if backbone is the bottleneck, B1 (TIPS) → if spans
unresolved, B2 (logic head). Stop at the first stage that meets the TEDS target.

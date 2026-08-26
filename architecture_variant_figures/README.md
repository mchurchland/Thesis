# Architecture Variant Figures

This folder contains scripts that build visual maps of architecture variants described in your methods text.

## Graph examples (requested style)

Run:

```bash
python architecture_variant_figures/make_weighted_graph_examples.py
```

Outputs to `architecture_variant_figures/graph_examples`:

- `model_grid_A_to_H.png` and `model_grid_I_to_O.png`: two large, portrait composite figures
  - grids are two columns by however many rows are needed
  - panels are lettered in the top-left corner for citation/reference in text
  - panels use letters only; model names are kept in the file index
  - the order follows the paper subsection order
  - `cel_sample` and `local_sign+sample` are omitted from the paper grids
  - the file index maps each letter to its corresponding architecture variant
- individual `*.png` files: one single-panel figure per architecture variant
  - edge colors use separate positive/negative scales derived from the C. elegans reference weights
  - negative edges use the negative color scale (solid edges)
- `00_file_index.png`: filename/variant index

## Trimmed topology figure and appendix controls

The sign-matched topology comparison follows the thesis feedback to keep the
main figure compact. It selects the original panels A and D, then relabels the
trimmed pair sequentially as A and B:

- A: sign-matched C. elegans connectome
- B: sign-matched connection shuffle

The former panel B (complex connections removed) is appended to the second
architecture appendix grid as panel P. The former panel C is omitted. This
produces `model_grid_I_to_P.png`; the same image is also written to the legacy
`model_grid_I_to_O.png` and `model_grid_I_to_Q.png` paths so existing thesis
references continue to resolve.

```bash
python architecture_variant_figures/make_weighted_graph_examples.py \
  --new-sign-matched-four-panel \
  --new-sign-matched-four-panel-out new_4to1_four_panel.png \
  --ce-ei ""
```

`new_4to1_four_panel.png` remains as a compatibility filename but now contains
the trimmed A/B comparison. An accurately named copy is written to
`new_4to1_two_panel.png`.

Implementation note:
- CE data loading uses `util.util.load_connectome`.
- Variant weight-matrix construction uses project-native code (`util.util.build_reservoir` plus CE shuffle helpers).

Optional args:

```bash
python architecture_variant_figures/make_weighted_graph_examples.py \
  --outdir architecture_variant_figures/graph_examples \
  --max-edges 0 \
  --layout-scale 1.35 \
  --ce-adj Connectome/ce_adj.npy \
  --ce-ei Connectome/ce_ei.npy
```

Notes:
- `--max-edges 0` draws all nonzero connections.
- Use `--skip-individual` to regenerate only the lettered composite figures and index.
- Use `--grid-titles` or `--no-grid-titles` to control whether the paper grids show panel titles.
- If `--max-edges` is positive, the script keeps the strongest edges and (by default) keeps all negative edges too.
- Use `--truncate-drops-negatives` if you want strict truncation.
- Use `--show-direction` to draw arrowheads (denser but explicit direction).
- Paper-size fonts are the default now; you can tune with:
  `--panel-title-fontsize`, `--suptitle-fontsize`, `--legend-fontsize`,
  `--cbar-title-fontsize`, `--cbar-label-fontsize`, `--cbar-tick-fontsize`,
  and `--index-fontsize`.

Show neuron-name node labels:

```bash
python architecture_variant_figures/make_weighted_graph_examples.py \
  --show-node-labels \
  --label-fontsize 5
```

## Previous map-style figures

If you still want the earlier property-map cards:

```bash
python architecture_variant_figures/make_architecture_figures.py
```

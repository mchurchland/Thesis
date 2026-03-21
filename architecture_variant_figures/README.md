# Architecture Variant Figures

This folder contains scripts that build visual maps of architecture variants described in your methods text.

## Graph examples (requested style)

Run:

```bash
python architecture_variant_figures/make_weighted_graph_examples.py
```

Outputs to `architecture_variant_figures/graph_examples`:

- `01_...png` through `14_...png`: one figure per architecture variant
  - left panel: biological C. elegans reference graph
  - right panel: architecture variant graph
  - both panels use the same Kamada-Kawai node layout for direct comparison
  - two separate scales are shown (right-side stacked colorbars):
    - positive edges: `0` to positive max
    - negative edges: most negative weight to `0`
  - colorbars are scaled from the left-panel C. elegans reference weights
  - negative edges use the negative color scale (solid edges)
- `00_file_index.png`: filename/variant index

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

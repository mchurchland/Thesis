# connectome_graph.py — 2025-10-02
# =====================================================================
# ConnectomeViewer colors nodes by instantaneous |V| with per-frame
# auto-contrast. Muscles (MD*/MV*) change color with |V| but never flash.
# No prints. Robust to either raw V or pre-normalized V in wc.V_vis.
# ---------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Optional, Sequence

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------
# Muscle lists (subset; extend as needed)
# ---------------------------------------------------------------------
muscleList = [
    # left, dorsal 07-23
    'MDL07','MDL08','MDL09','MDL10','MDL11','MDL12','MDL13','MDL14','MDL15','MDL16','MDL17','MDL18','MDL19','MDL20','MDL21','MDL22','MDL23',
    # left, ventral 07-23
    'MVL07','MVL08','MVL09','MVL10','MVL11','MVL12','MVL13','MVL14','MVL15','MVL16','MVL17','MVL18','MVL19','MVL20','MVL21','MVL22','MVL23',
    # right, dorsal 07-23
    'MDR07','MDR08','MDR09','MDR10','MDR11','MDR12','MDR13','MDR14','MDR15','MDR16','MDR17','MDR18','MDR19','MDR20','MDR21','MDR22','MDR23',
    # right, ventral 07-23
    'MVR07','MVR08','MVR09','MVR10','MVR11','MVR12','MVR13','MVR14','MVR15','MVR16','MVR17','MVR18','MVR19','MVR20','MVR21','MVR22','MVR23'
]

mLeft = [
    'MDL07','MDL08','MDL09','MDL10','MDL11','MDL12','MDL13','MDL14','MDL15','MDL16','MDL17','MDL18','MDL19','MDL20','MDL21','MDL22','MDL23',
    'MVL07','MVL08','MVL09','MVL10','MVL11','MVL12','MVL13','MVL14','MVL15','MVL16','MVL17','MVL18','MVL19','MVL20','MVL21','MVL22','MVL23'
]
mRight = [
    'MDR07','MDR08','MDR09','MDR10','MDR11','MDR12','MDR13','MDR14','MDR15','MDR16','MDR17','MDR18','MDR19','MDR20','MDR21','MDR22','MDR23',
    'MVR07','MVR08','MVR09','MVR10','MVR11','MVR12','MVR13','MVR14','MVR15','MVR16','MVR17','MVR18','MVR19','MVR20','MVR21','MVR22','MVR23'
]

__all__ = ["ConnectomeViewer"]


class ConnectomeViewer:
    """
    Lightweight live viewer for *C. elegans* connectome simulations.

    Expected wc attributes:
      - names: list[str] of node names (includes neurons and muscles)
      - W: np.ndarray (N,N) weights (only used for edge selection if _edge_* not present)
      - V: np.ndarray (N,) current potentials (fallback source)
      - V_vis: np.ndarray (N,) potentials to display (preferred)
      - spiked_vis: np.ndarray(bool, N) optional spike mask for pulsing
      - muscle_mask: np.ndarray(bool, N) marks muscle nodes (viewer also checks name)

    Optional fast-path attributes:
      - _edge_w: 1D array of nonzero edge weights
      - _edge_ptr: 2D array with rows [edge_index, i, j] mapping into W

    Optional group indices for the grouped layout:
      - touch_idx: np.ndarray[int]
      - food_idx: np.ndarray[int]
    """

    _ALLOWED_MUSCLE_PREFIX: Sequence[str] = ("MDL", "MDR", "MVL", "MVR")

    def __init__(
        self,
        wc,
        *,
        layout: str = "kamada_groups",
        spread: float = 1.6,
        pulse_size: float = 2.5,
        group_gap: float = 12.0,
        threshold: Optional[float] = None,  # used only for binary/energy “active” overlay
        max_edges: int = 6_000,
        node_size: int = 40,
        color_mode: str = "energy",  # 'binary' | 'heat' | 'energy'
        colormap: str = "plasma",
        vmax: Optional[float] = None,  # ignored; auto-scaled each frame
        inact_color: str = "#d3d3d3",
        act_color: str = "#ff5555",
    ):
        self.wc = wc
        self.node_size = node_size
        self.spread = spread
        self.pulse_size = pulse_size
        self.group_gap = group_gap
        self.color_mode = color_mode
        self.inact_col = inact_color
        self.act_col = act_color

        # Threshold map for “active” overlay only
        if threshold is not None:
            self.thr_map = np.full(wc.N, float(threshold))
        elif hasattr(wc, "_thr_map"):
            self.thr_map = wc._thr_map.astype(float)
        elif hasattr(wc, "threshold"):
            self.thr_map = np.full(wc.N, float(wc.threshold))
        else:
            self.thr_map = np.full(wc.N, 1.0)  # small default; only affects active overlay

        # Colormap and normalization domain; per-frame vmax set in step()
        if color_mode == "heat":
            self.cmap = cm.get_cmap(colormap)
        elif color_mode == "energy":
            self.cmap = cm.get_cmap("Blues_r")
        self.norm = mcolors.Normalize(0.0, 1.0)

        # Detect state provider mode (fallbacks kept for compatibility)
        if all(hasattr(wc, a) for a in ("post", "curcol")):
            self._state_mode = "double"
        elif all(hasattr(wc, a) for a in ("post", "t0")):
            self._state_mode = "double_t0"
        else:
            self._state_mode = "single"

        # Keep only allowed muscles; everything else shown as present in wc.names
        _MWHITELIST = set(muscleList)
        self._muscle_nodes = [n for n in wc.names if n in _MWHITELIST]

        self.G = self._build_graph(max_edges)
        self.name2idx = {n: i for i, n in enumerate(wc.names)}
        self.node_order = list(self.G.nodes())
        self.node_indices = np.array([self.name2idx[n] for n in self.node_order])
        self.muscle_mask_draw = np.array([n in self._muscle_nodes for n in self.node_order])

        self.pos = {n: p * self.spread for n, p in self._compute_layout(layout).items()}

        # Initial draw
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        base = self.inact_col if color_mode == "binary" else self.cmap(self.norm(0.0))
        self._draw_graph(base)
        self.ax.set_axis_off()
        self.fig.tight_layout()
        self._add_side_labels()

    # Public API
    def step(self):
        """Redraw node colors and sizes for current wc state."""
        # Choose voltage vector
        if hasattr(self.wc, "V_vis"):
            V_draw = np.asarray(self.wc.V_vis, dtype=float)[self.node_indices]
        else:
            if self._state_mode == "double":
                V_draw = np.asarray(self.wc.post[:, self.wc.curcol], dtype=float)[self.node_indices]
            elif self._state_mode == "double_t0":
                V_draw = np.asarray(self.wc.post[:, self.wc.t0], dtype=float)[self.node_indices]
            else:
                V_draw = np.asarray(self.wc.V, dtype=float)[self.node_indices]

        absV = np.abs(V_draw)

        # Per-frame auto-contrast (95th percentile)
        vmax = float(np.percentile(absV, 95)) + 1e-9
        self.norm.vmin = 0.0
        self.norm.vmax = vmax

        # Spike mask for pulsing; muscles never flash
        spk = getattr(self.wc, "spiked_vis", np.zeros_like(self.wc.V, dtype=bool))[self.node_indices]
        spk = spk.copy()
        spk[self.muscle_mask_draw] = False

        # Colors
        if self.color_mode == "binary":
            active = spk | (absV > self.thr_map[self.node_indices])
            active[self.muscle_mask_draw] = False
            colors = np.where(active, self.act_col, self.inact_col)
        elif self.color_mode == "heat":
            colors = self.cmap(self.norm(absV))
            colors[spk] = mcolors.to_rgba(self.act_col)
        else:  # 'energy'
            blues = self.cmap(self.norm(absV))
            active = spk | (absV > self.thr_map[self.node_indices])
            colors = np.where(active[:, None], mcolors.to_rgba(self.act_col), blues)

        self.node_coll.set_color(colors)

        # Sizes (pulse on spikes only)
        sizes = np.where(spk, self.node_size * self.pulse_size, self.node_size)
        self.node_coll.set_sizes(sizes)

        self.fig.canvas.draw_idle()

    update = step  # alias

    # Helpers
    def _evenly_space(self, nodes, x, dy, y_shift=0.0):
        if not nodes:
            return {}
        y0 = -(len(nodes) - 1) / 3.0 * dy + y_shift
        return {n: np.array([x, y0 + k * dy]) for k, n in enumerate(sorted(nodes))}

    def _is_allowed_muscle(self, name: str) -> bool:
        return name in self._muscle_nodes

    def _build_graph(self, max_edges: int) -> nx.DiGraph:
        keep = [n for i, n in enumerate(self.wc.names)
                if not (self._is_muscle_index(i) and not self._is_allowed_muscle(n))]
        keep_set = set(keep)

        G = nx.DiGraph()
        G.add_nodes_from(keep)

        if hasattr(self.wc, "_edge_w") and hasattr(self.wc, "_edge_ptr"):
            idx = np.argsort(np.abs(self.wc._edge_w))[::-1][:max_edges]
            for (_, i, j), w in zip(self.wc._edge_ptr[idx], self.wc._edge_w[idx]):
                s, d = self.wc.names[int(i)], self.wc.names[int(j)]
                if s in keep_set and d in keep_set:
                    G.add_edge(s, d, weight=float(w))
        else:
            flat = np.argsort(np.abs(self.wc.W).ravel())[::-1][:max_edges]
            src, dst = np.unravel_index(flat, self.wc.W.shape)
            for s, d in zip(src, dst):
                sn, dn = self.wc.names[int(s)], self.wc.names[int(d)]
                if sn in keep_set and dn in keep_set:
                    G.add_edge(sn, dn, weight=float(self.wc.W[int(s), int(d)]))
        return G

    def _draw_graph(self, init_color):
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, arrows=False, alpha=0.25, width=0.3)
        self.node_coll = nx.draw_networkx_nodes(
            self.G, self.pos, node_size=self.node_size, node_color=[init_color], ax=self.ax
        )

    def _compute_layout(self, layout: str):
        return self._compute_kamada_group_layout() if layout == "kamada_groups" else self._safe_standard_layout(layout)

    def _safe_standard_layout(self, name: str):
        if name != "kamada_kawai":
            return self._standard_layout(name)
        try:
            pos = self._standard_layout("kamada_kawai")
        except Exception:
            return self._standard_layout("spring")
        r = np.linalg.norm(np.vstack(list(pos.values())), axis=1)
        return pos if np.std(r) >= 1e-2 else self._standard_layout("spring")

    def _standard_layout(self, name: str):
        if name == "spring":
            return nx.spring_layout(self.G, seed=1)
        if name == "kamada_kawai":
            H = self.G.to_undirected()
            for _, _, d in H.edges(data=True):
                w = abs(d.get("weight", 1.0))
                d["length"] = 1.0 / max(w, 1e-9)
            return nx.kamada_kawai_layout(H, weight="length")
        raise ValueError(name)

    def _add_side_labels(self):
        left_nodes  = [n for n in self.pos if n in mLeft]
        right_nodes = [n for n in self.pos if n in mRight]

        def _label(nodes, text):
            if not nodes:
                return
            xs = [self.pos[n][0] for n in nodes]
            ys = [self.pos[n][1] for n in nodes]
            x = float(np.mean(xs))
            y = float(max(ys)) + 0.3
            self.ax.text(x, y, text, ha="center", va="bottom", fontsize=16, fontweight="bold")

        _label(left_nodes,  "L")
        _label(right_nodes, "R")

    def _compute_kamada_group_layout(self) -> Dict[str, np.ndarray]:
        pos = self._safe_standard_layout("kamada_kawai")
        n_by = lambda idx: self.wc.names[idx]

        g_touch = [n_by(i) for i in getattr(self.wc, "touch_idx", []) if n_by(i) in self.G]
        g_food  = [n_by(i) for i in getattr(self.wc, "food_idx", []) if n_by(i) in self.G]
        g_left  = [n for n in self.G if n in mLeft]
        g_right = [n for n in self.G if n in mRight]

        gap = self.group_gap
        x_center = 3.25 * gap
        offset = 0.5 * gap

        x_touch = -x_center - offset
        x_food  = -x_center + offset
        x_left  =  x_center - offset
        x_right =  x_center + offset

        dy_fixed = 0.04
        shift_up = 0.15

        pos.update(self._evenly_space(g_touch, x_touch, dy_fixed, y_shift=shift_up))
        pos.update(self._evenly_space(g_food,  x_food,  dy_fixed, y_shift=shift_up))
        pos.update(self._evenly_space(g_left,  x_left,  dy_fixed))
        pos.update(self._evenly_space(g_right, x_right, dy_fixed))

        g_center = [
            n for n in self.G
            if n not in g_touch + g_food + g_left + g_right and n != "MVULVA"
        ]
        if g_center:
            pts = np.vstack([pos[n] for n in g_center])
            cent = pts.mean(0)
            r = np.linalg.norm(pts - cent, axis=1)
            r_med = np.median(r)
            for n, rv in zip(g_center, r):
                scale = 6.0 if rv < r_med else 1.0
                pos[n] = cent + (pos[n] - cent) * scale

        return pos

    def _is_muscle_index(self, i: int) -> bool:
        try:
            return bool(self.wc.muscle_mask[i])
        except Exception:
            # fallback by name if mask missing
            name = self.wc.names[i]
            return any(name.startswith(pfx) for pfx in self._ALLOWED_MUSCLE_PREFIX)

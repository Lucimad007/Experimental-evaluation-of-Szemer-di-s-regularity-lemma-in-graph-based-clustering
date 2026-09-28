"""Clear concept diagrams for the defense deck. No text."""
from pathlib import Path

import subprocess

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parents[1] / "assets" / "figures"
BG = "#F4EFE4"
NAVY = "#1A334A"
TERR = "#A33B24"
SAGE = "#2C5E52"
SAND = "#C4A574"
INK = "#1B1816"
PALE_N = "#D5E0EA"
PALE_S = "#D5E4DE"
PALE_T = "#F0D7CF"
PALE_A = "#EFE4D0"


def panel(w=4.4, h=4.8):
    f, ax = plt.subplots(figsize=(w, h), dpi=150)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    f.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return f, ax


def label(ax, x, y, text, size=28, color=INK):
    hex_color = color.lstrip("#")
    slug = f"v2_{hex_color}_{size}_{text}"
    path = OUT / "_labels" / f"{abs(hash(slug))}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run(
            [
                "powershell",
                "-STA",
                "-File",
                str(Path(__file__).with_name("_fa_label.ps1")),
                "-Text",
                text,
                "-Out",
                str(path),
                "-Color",
                hex_color,
                "-Size",
                str(int(size * 6)),
            ],
            check=True,
        )
    image = plt.imread(path)
    ax.add_artist(
        AnnotationBbox(
            OffsetImage(image, zoom=0.28),
            (x, y),
            frameon=False,
            pad=0,
            zorder=6,
        )
    )


def fig():
    f, ax = plt.subplots(figsize=(13.2, 7.2), dpi=140)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 7.2)
    ax.set_aspect("equal")
    ax.axis("off")
    f.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return f, ax


def save(f, name):
    ax = f.axes[0]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    f.set_size_inches(x1 - x0, y1 - y0)
    ax.set_position([0, 0, 1, 1])
    ax.set_aspect("equal", adjustable="box")
    f.savefig(OUT / name, dpi=120, facecolor=BG)
    plt.close(f)


def arrow(ax, x1, y1, x2, y2, color=TERR, lw=3.2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=22,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def blob(ax, pts, color, pad=0.42):
    pts = np.asarray(pts)
    c = pts.mean(axis=0)
    ang = np.linspace(0, 2 * np.pi, 80)
    r = np.max(np.linalg.norm(pts - c, axis=1)) + pad
    ax.add_patch(Circle(c, r, facecolor=color, edgecolor="none", zorder=0, alpha=0.95))


def dots(ax, pts, color, s=280, z=3):
    pts = np.asarray(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=s, c=color, zorder=z, linewidths=0)


def vertices():
    f, ax = fig()
    row_colors = [NAVY, TERR, SAGE, "#8A5A2A", "#3E5F8A", "#6E4A62"]
    feat = ["#C9C1B4", "#C9C1B4", "#C9C1B4"]
    ys = np.linspace(6.2, 1.0, 6)
    for y, col in zip(ys, row_colors):
        ax.add_patch(
            FancyBboxPatch(
                (0.55, y - 0.38),
                4.15,
                0.76,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                facecolor="#FFFCF8",
                edgecolor="none",
                zorder=2,
            )
        )
        ax.add_patch(FancyBboxPatch((0.55, y - 0.38), 0.28, 0.76, boxstyle="round,pad=0,rounding_size=0.08", facecolor=col, edgecolor="none", zorder=3))
        for k, c in enumerate(feat):
            ax.scatter([1.7 + k * 0.85], [y], s=150, c=c, zorder=4, linewidths=0)
        ax.plot([4.7, 8.15], [y, y], color=col, lw=1.4, alpha=0.85, zorder=1)
        ax.scatter([8.7], [y], s=620, c=col, zorder=4, linewidths=0)
    save(f, "concept_vertices.png")


def heavy():
    f, ax = fig()
    rng = np.random.default_rng(4)
    centers = np.array([(1.15, 5.15), (3.15, 5.15), (1.15, 2.35), (3.15, 2.35)])
    cols = [NAVY, TERR, SAGE, SAND]
    pales = [PALE_N, PALE_T, PALE_S, PALE_A]
    groups = []
    for cx, cy in centers:
        ang = rng.uniform(0, 2 * np.pi, 6)
        rad = rng.uniform(0.12, 0.48, 6)
        groups.append(np.column_stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)]))
    allpts = np.vstack(groups)
    for i, a in enumerate(allpts):
        for b in allpts[i + 1 :]:
            if rng.random() < 0.28:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#C9C1B4", lw=0.45, zorder=1)
    dots(ax, allpts, NAVY, s=22, z=2)
    arrow(ax, 4.15, 3.75, 4.85, 3.75)
    for pts, pale, col in zip(groups, pales, cols):
        c = pts.mean(axis=0) + np.array([4.55, 0])
        ax.add_patch(Circle(c, 0.85, facecolor=pale, edgecolor="none", zorder=0))
        dots(ax, pts + np.array([4.55, 0]), col, s=22, z=3)
    arrow(ax, 9.55, 3.75, 10.25, 3.75)
    big = centers + np.array([9.7, 0])
    for i, j, w in (
        (0, 1, 8),
        (0, 2, 3),
        (1, 3, 5),
        (2, 3, 2),
    ):
        ax.plot([big[i, 0], big[j, 0]], [big[i, 1], big[j, 1]], color="#8A8175", lw=w, zorder=1, solid_capstyle="round")
    for p, col in zip(big, cols):
        ax.add_patch(Circle(p, 0.38, facecolor=col, zorder=3))
    save(f, "concept_heavy_graph.png")


def edges():
    f, ax = fig()
    # three pairs at increasing distance
    pairs = [
        ((2.2, 5.2), (3.5, 5.2), 14, TERR),
        ((6.0, 3.6), (8.2, 3.6), 6, "#C47A62"),
        ((2.4, 1.6), (10.6, 1.6), 1.6, "#C5CBB8"),
    ]
    for a, b, lw, col in pairs:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=lw, solid_capstyle="round", zorder=1)
    for a, b, *_ in pairs:
        dots(ax, [a, b], NAVY, s=700)
    save(f, "concept_edges.png")


def partition():
    f, ax = fig()
    rng = np.random.default_rng(3)
    centers = [(2.6, 5.2), (6.6, 5.2), (2.6, 2.0), (6.6, 2.0)]
    cols = [NAVY, TERR, SAGE, SAND]
    pales = [PALE_N, PALE_T, PALE_S, PALE_A]
    for (cx, cy), col, pale in zip(centers, cols, pales):
        ax.add_patch(Circle((cx, cy), 1.35, facecolor=pale, edgecolor="none", zorder=0))
        ang = rng.uniform(0, 2 * np.pi, 7)
        rad = rng.uniform(0.15, 0.85, 7)
        pts = np.column_stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)])
        dots(ax, pts, col, s=140)
    outliers = np.array([[10.6, 4.2], [11.4, 3.5], [10.5, 2.8]])
    dots(ax, outliers, "#B7B1A6", s=160)
    save(f, "concept_partition.png")


def _pair_groups():
    rng = np.random.default_rng(11)
    def cloud(cx, cy):
        ang = rng.uniform(0, 2 * np.pi, 8)
        rad = rng.uniform(0.25, 1.05, 8)
        return np.column_stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)])
    return cloud(2.8, 3.6), cloud(10.2, 3.6)


def eps_pair():
    f, ax = fig()
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    gold = "#C4A574"

    def column(x, y0, n, color):
        ys = np.linspace(y0, y0 + 6.4, n)
        return np.column_stack([np.full(n, x), ys])

    left_a, left_b = column(1.15, 1.2, 8, NAVY), column(4.55, 1.2, 8, NAVY)
    right_a, right_b = column(9.3, 1.2, 8, NAVY), column(12.7, 1.2, 8, NAVY)
    for a in left_a:
        for b in left_b:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=gold, lw=1.15, alpha=0.85, zorder=1)
    for a in right_a:
        for b in right_b:
            if a[1] > 4.4 and b[1] > 4.4:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=gold, lw=1.35, alpha=0.9, zorder=1)
    ax.add_patch(
        FancyBboxPatch(
            (8.85, 4.15),
            4.3,
            3.7,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="none",
            edgecolor=TERR,
            lw=2.2,
            zorder=2,
        )
    )
    ax.plot([8.05, 8.05], [0.7, 8.3], color=NAVY, lw=1.6, zorder=0)
    dots(ax, left_a, NAVY, s=420)
    dots(ax, left_b, NAVY, s=420)
    dots(ax, right_a, NAVY, s=420)
    dots(ax, right_b, NAVY, s=420)
    save(f, "concept_eps_pair.png")


def regular():
    f, ax = fig()
    L, R = _pair_groups()
    blob(ax, L, PALE_N, pad=0.55)
    blob(ax, R, PALE_S, pad=0.55)
    for a in L:
        for b in R:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=TERR, lw=0.7, alpha=0.55, zorder=1)
    dots(ax, L, NAVY, s=180)
    dots(ax, R, SAGE, s=180)
    save(f, "concept_regular.png")


def irregular():
    f, ax = fig()
    L, R = _pair_groups()
    blob(ax, L, PALE_N, pad=0.55)
    blob(ax, R, PALE_S, pad=0.55)
    for a in L:
        for b in R:
            if a[1] > 3.7 and b[1] > 3.7:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=TERR, lw=1.5, alpha=0.85, zorder=1)
    dots(ax, L, NAVY, s=180)
    dots(ax, R, SAGE, s=180)
    save(f, "concept_irregular.png")


def density():
    f, ax = fig()
    L = np.array([(1.3 + (i % 3) * 0.7, 5.0 - (i // 3) * 0.85) for i in range(9)])
    R = L + np.array([3.2, 0])
    blob(ax, L, PALE_N, pad=0.4)
    blob(ax, R, PALE_T, pad=0.4)
    rng = np.random.default_rng(1)
    for a in L:
        for b in R:
            if rng.random() < 0.45:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=TERR, lw=0.6, alpha=0.45, zorder=1)
    dots(ax, L, NAVY, s=90)
    dots(ax, R, TERR, s=90)
    arrow(ax, 6.55, 3.6, 7.7, 3.6)
    ax.add_patch(Circle((9.1, 3.6), 0.85, facecolor=NAVY, zorder=3))
    ax.add_patch(Circle((11.7, 3.6), 0.85, facecolor=TERR, zorder=3))
    ax.plot([9.95, 10.85], [3.6, 3.6], color=TERR, lw=18, solid_capstyle="butt", zorder=2)
    save(f, "concept_density.png")


def reduced():
    f, ax = fig()
    pales = [PALE_N, PALE_T, PALE_S, PALE_A]
    cols = [NAVY, TERR, SAGE, SAND]
    left_c = [(2.3, 6.0), (2.3, 4.35), (2.3, 2.7), (2.3, 1.05)]
    right_c = [(10.4, 6.0), (10.4, 4.35), (10.4, 2.7), (10.4, 1.05)]
    widths = {(0, 1): 11, (1, 2): 5, (2, 3): 2}
    for i, (c, pale, col) in enumerate(zip(left_c, pales, cols)):
        pts = np.array([[c[0] + dx, c[1] + dy] for dx in (-0.35, 0, 0.35) for dy in (-0.22, 0.22)])
        blob(ax, pts, pale, pad=0.28)
        dots(ax, pts, col, s=40)
        arrow(ax, c[0] + 0.85, c[1], right_c[i][0] - 0.7, right_c[i][1], lw=2.2)
        ax.add_patch(Circle(right_c[i], 0.48, facecolor=col, zorder=4))
    for (i, j), w in widths.items():
        y1, y2 = right_c[i][1] - 0.5, right_c[j][1] + 0.5
        ax.plot([10.4, 10.4], [y1, y2], color="#8A8175", lw=w, solid_capstyle="butt", zorder=2)
    arc_y = np.linspace(right_c[0][1], right_c[2][1], 40)
    arc_x = 10.95 + 0.85 * np.sin(np.linspace(0, np.pi, 40))
    ax.plot(arc_x, arc_y, color="#8A8175", lw=7, solid_capstyle="round", zorder=2)
    for c, col in zip(right_c, cols):
        ax.add_patch(Circle(c, 0.48, facecolor=col, zorder=4))
    save(f, "concept_reduced.png")


def _grid(ax, origin, mode):
    local = np.array(
        [[0, 1.15], [1.15, 1.15], [2.3, 1.15], [0, 0], [1.15, 0], [2.3, 0], [0, -1.15], [1.15, -1.15], [2.3, -1.15]]
    )
    cols = [TERR, TERR, TERR, SAGE, SAGE, SAGE, SAGE, SAGE, SAGE]
    pts = local + np.array(origin)
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#DDD6CB", lw=1.1, zorder=1)
    if mode == "plain":
        dots(ax, pts, NAVY, s=220)
    else:
        for p, c in zip(pts, cols):
            ax.scatter([p[0]], [p[1]], s=280, c=c, zorder=3, linewidths=0)
    return pts


def _same_graph(ax, ox, oy, colored):
    left = np.array([[0.0, 0.15], [0.55, 1.05], [0.35, -0.75]])
    right = np.array([[1.85, 0.35], [2.45, 1.0], [2.25, -0.7]])
    pts = np.vstack([left, right]) + np.array([ox, oy])
    for group in (pts[:3], pts[3:]):
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#C9BFB2", lw=1.6, zorder=1, solid_capstyle="round")
    for a in pts[:3]:
        for b in pts[3:]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#E3DBD0", lw=0.8, zorder=1)
    if colored:
        dots(ax, pts[:3], TERR, s=280)
        dots(ax, pts[3:], SAGE, s=280)
    else:
        dots(ax, pts, NAVY, s=280)
    return pts


def two_paths():
    f, ax = fig()
    _grid(ax, (2.2, 3.6), "color")
    ax.set_xlim(1.2, 5.5)
    ax.set_ylim(1.6, 5.6)
    save(f, "concept_path_direct.png")

    f, ax = fig()
    rng = np.random.default_rng(8)
    many = rng.uniform((1.15, 4.15), (5.55, 8.15), size=(28, 2))
    for i, a in enumerate(many):
        for b in many[i + 1 :]:
            if rng.random() < 0.16:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#DDD6CB", lw=0.7, zorder=1)
    dots(ax, many, NAVY, s=70)
    arrow(ax, 3.35, 3.7, 3.35, 2.85)
    small = np.array([(2.15, 1.7), (4.55, 1.7), (2.15, 0.55), (4.55, 0.55)])
    for i, j, w in ((0, 1, 5), (0, 2, 2.2), (1, 3, 3.4), (2, 3, 1.6)):
        ax.plot(
            [small[i, 0], small[j, 0]],
            [small[i, 1], small[j, 1]],
            color="#8A8175",
            lw=w,
            solid_capstyle="round",
            zorder=2,
        )
    dots(ax, small, NAVY, s=280)
    ax.set_xlim(0.55, 6.15)
    ax.set_ylim(0.05, 8.55)
    save(f, "concept_path_lemma.png")


def lift():
    f, ax = fig()
    supers = [(2.0, 5.4, NAVY), (4.2, 5.4, TERR), (2.0, 2.6, NAVY), (4.2, 2.6, SAGE)]
    for x, y, c in supers:
        ax.add_patch(Circle((x, y), 0.62, facecolor=c, zorder=3))
    arrow(ax, 5.3, 4.0, 6.6, 4.0)
    clouds = [
        (8.3, 5.3, NAVY),
        (11.0, 5.3, TERR),
        (8.3, 2.3, NAVY),
        (11.0, 2.3, SAGE),
    ]
    rng = np.random.default_rng(2)
    for x, y, c in clouds:
        pts = np.column_stack([rng.normal(x, 0.45, 18), rng.normal(y, 0.38, 18)])
        dots(ax, pts, c, s=36)
    save(f, "concept_lift.png")


def polish():
    f, ax = fig()
    rng = np.random.default_rng(5)

    def scene(shift, fix):
        blue = np.column_stack(
            [rng.normal(1.9 + shift, 0.38, 16), rng.normal(3.7, 0.38, 16)]
        )
        red = np.column_stack(
            [rng.normal(5.2 + shift, 0.38, 16), rng.normal(3.7, 0.38, 16)]
        )
        ax.add_patch(
            Circle((2.4 + shift, 3.7), 1.85, facecolor=PALE_N, edgecolor=NAVY, lw=1.6, zorder=0)
        )
        dots(ax, blue, NAVY, s=70)
        dots(ax, red, TERR, s=70)
        stray = np.array([3.85 + shift, 3.7])
        ax.scatter(
            [stray[0]],
            [stray[1]],
            s=460,
            c=TERR if fix else NAVY,
            zorder=5,
            linewidths=2.6,
            edgecolors="white",
        )

    scene(0, False)
    arrow(ax, 6.55, 3.7, 7.25, 3.7)
    scene(6.7, True)
    ax.set_xlim(0.4, 13.0)
    ax.set_ylim(1.6, 5.8)
    save(f, "concept_polish.png")


def four():
    f, ax = fig()
    # four panels, right-to-left reading: spectral, affinity, dominant, forest
    # drawn left-to-right as forest, dominant, affinity, spectral so RTL names match caption order
    # Caption lists spectral first (rightmost in RTL). Put spectral on the right.
    panels = [0.4, 3.6, 6.8, 10.0]
    for x in panels:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.6),
                2.8,
                6.0,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                facecolor="#FFFCF8",
                edgecolor="none",
            )
        )
    # panel 0 (left): forest — three small trees
    trees = [
        [(1.8, 5.4), (1.3, 4.5), (2.3, 4.5), (1.0, 3.6), (1.6, 3.6)],
        [(1.8, 2.8), (1.3, 1.9), (2.3, 1.9)],
    ]
    for tree, col in zip(trees, [NAVY, SAGE]):
        for i, a in enumerate(tree):
            if i:
                ax.plot([tree[0][0], a[0]], [tree[0][1], a[1]], color=col, lw=1.4, zorder=2)
        dots(ax, tree, col, s=80)
    # panel 1: one dense dominant clump, faint outliers
    rng = np.random.default_rng(7)
    core = np.column_stack([rng.normal(5.0, 0.28, 16), rng.normal(3.6, 0.28, 16)])
    dots(ax, core, SAGE, s=70)
    out = np.array([[4.0, 5.6], [6.0, 5.4], [3.9, 1.6], [6.1, 1.8], [5.0, 6.0]])
    dots(ax, out, "#D9D3C8", s=50)
    # panel 2: exemplars with satellites
    exemplars = [(8.2, 5.0, NAVY), (8.2, 2.4, TERR)]
    for ex, ey, col in exemplars:
        sats = [(ex - 0.7, ey + 0.45), (ex + 0.7, ey + 0.35), (ex - 0.55, ey - 0.45), (ex + 0.6, ey - 0.4)]
        for s in sats:
            ax.plot([ex, s[0]], [ey, s[1]], color=col, lw=1.1, ls=(0, (2, 2)), zorder=1)
            ax.scatter([s[0]], [s[1]], s=50, c=col, zorder=2, linewidths=0)
        ax.scatter([ex], [ey], s=280, c=col, zorder=3, linewidths=0)
    # panel 3 right: three separated arcs (spectral groups)
    for ang0, col, rad in ((20, NAVY, 0), (140, TERR, 0), (250, SAGE, 0)):
        angs = np.deg2rad(np.linspace(ang0, ang0 + 70, 7))
        pts = np.column_stack([11.4 + 0.95 * np.cos(angs), 3.6 + 0.95 * np.sin(angs)])
        dots(ax, pts, col, s=70)
    save(f, "concept_four_algos.png")


def certs():
    def pair(ax, left, right, edges, rings=()):
        blob(ax, left, PALE_N, pad=0.45)
        blob(ax, right, PALE_S, pad=0.45)
        for a, b, lw in edges:
            ax.plot([left[a, 0], right[b, 0]], [left[a, 1], right[b, 1]], color=TERR, lw=lw, zorder=1, solid_capstyle="round")
        dots(ax, left, NAVY, s=160)
        dots(ax, right, SAGE, s=160)
        for side, i in rings:
            p = left[i] if side == 0 else right[i]
            ax.scatter([p[0]], [p[1]], s=420, facecolors="none", edgecolors=TERR, linewidths=2.4, zorder=4)

    L = np.array([[1.2, 4.2], [2.0, 4.6], [1.4, 3.2], [2.2, 3.4], [1.7, 2.4]])
    R = L + np.array([3.4, 0])

    f, ax = fig()
    pair(ax, L, R, [(0, 2, 1.2), (3, 0, 1.2)])
    ax.set_xlim(0.2, 7.2)
    ax.set_ylim(1.4, 5.6)
    save(f, "concept_alon1.png")

    f, ax = fig()
    edges = [(i, j, 2.2) for i in range(3) for j in range(3)]
    pair(ax, L, R, edges)
    ax.set_xlim(0.2, 7.2)
    ax.set_ylim(1.4, 5.6)
    save(f, "concept_alon3.png")

    f, ax = fig()
    edges = [(i, j, 0.7) for i in range(5) for j in range(5) if (i + j) % 2 == 0]
    edges += [(0, 0, 3.2), (0, 1, 3.2), (1, 0, 3.2)]
    pair(ax, L, R, edges, rings=[(1, 0), (0, 0)])
    ax.set_xlim(0.2, 7.2)
    ax.set_ylim(1.4, 5.6)
    save(f, "concept_alon2.png")

    f, ax = fig()
    whole = np.array([[1.7, 4.55], [2.55, 4.7], [1.85, 3.7], [2.7, 3.15], [1.75, 2.45], [2.65, 2.2]])
    blob(ax, whole, PALE_N, pad=0.55)
    cert = whole[:3]
    rest = whole[3:]
    cc = cert.mean(axis=0)
    ax.add_patch(Circle(cc, 1.05, facecolor=PALE_T, edgecolor=TERR, lw=2.2, zorder=1))
    dots(ax, cert, TERR, s=220, z=3)
    dots(ax, rest, NAVY, s=220, z=3)
    label(ax, 1.15, 5.55, "یک گروه", size=22, color=NAVY)
    label(ax, 3.55, 4.85, "گواه", size=22, color=TERR)

    arrow(ax, 4.35, 3.45, 5.35, 3.45)

    top = np.array([[6.7, 5.15], [7.7, 5.25], [7.15, 4.35]])
    bot = np.array([[6.75, 2.55], [7.75, 2.35], [7.1, 1.55]])
    scrap = np.array([[10.15, 3.85], [10.85, 3.15]])
    blob(ax, top, PALE_T, pad=0.48)
    blob(ax, bot, PALE_N, pad=0.48)
    ax.add_patch(Circle(scrap.mean(axis=0), 0.85, facecolor="#E6E1D8", edgecolor="none", zorder=0))
    dots(ax, top, TERR, s=220)
    dots(ax, bot, NAVY, s=220)
    dots(ax, scrap, "#8A8175", s=160)
    label(ax, 8.55, 5.35, "گواه", size=22, color=TERR)
    label(ax, 8.55, 1.45, "بقیه", size=22, color=NAVY)
    label(ax, 10.5, 4.85, "استثنا", size=22, color="#6B645C")
    ax.set_xlim(0.2, 11.6)
    ax.set_ylim(0.55, 6.15)
    save(f, "concept_refine.png")


def fair():
    cols = [TERR, TERR, TERR, SAGE, SAGE, SAGE]
    spots = [(1.6, 4.6), (3.2, 4.6), (4.8, 4.6), (1.6, 2.4), (3.2, 2.4), (4.8, 2.4)]

    f, ax = fig()
    for (x, y), c in zip(spots, cols):
        ax.scatter([x], [y], s=900, c=c, zorder=3, linewidths=0)
    ax.set_xlim(0.4, 6.0)
    ax.set_ylim(1.0, 6.0)
    save(f, "concept_label_direct.png")

    f, ax = fig()
    ax.add_patch(FancyBboxPatch((0.45, 4.35), 2.5, 1.9, boxstyle="round,pad=0.02,rounding_size=0.18", facecolor=PALE_T, edgecolor=TERR, lw=2))
    ax.add_patch(FancyBboxPatch((3.15, 4.35), 2.5, 1.9, boxstyle="round,pad=0.02,rounding_size=0.18", facecolor=PALE_S, edgecolor=SAGE, lw=2))
    for x, c in ((0.9, TERR), (1.7, TERR), (2.5, TERR), (3.6, SAGE), (4.4, SAGE), (5.2, SAGE)):
        ax.scatter([x], [5.3], s=280, c=c, zorder=3, linewidths=0)
    arrow(ax, 3.05, 4.05, 3.05, 3.15)
    for x, c in ((0.9, TERR), (1.7, TERR), (2.5, TERR), (3.6, SAGE), (4.4, SAGE), (5.2, SAGE)):
        ax.scatter([x], [2.15], s=520, c=c, zorder=3, linewidths=0)
    ax.set_xlim(0.15, 5.95)
    ax.set_ylim(1.2, 6.6)
    save(f, "concept_label_copy.png")


def _panel(ax, x, y, w, h, face):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=face,
            edgecolor="none",
            zorder=0,
        )
    )


def similarity_matrix():
    f, ax = fig()
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 7.2)
    n = 10
    rng = np.random.default_rng(3)
    m = rng.uniform(0.08, 0.28, (n, n))
    m[:4, :4] = rng.uniform(0.62, 0.92, (4, 4))
    m[5:9, 5:9] = rng.uniform(0.55, 0.88, (4, 4))
    np.fill_diagonal(m, 1.0)
    m = (m + m.T) / 2
    cell = 0.5
    x0, y0 = 0.7, 0.55
    for i in range(n):
        for j in range(n):
            v = float(m[i, j])
            pale = np.array([0xF4, 0xEF, 0xE4]) / 255
            ink = np.array([0x1A, 0x33, 0x4A]) / 255
            rgb = pale * (1 - v) + ink * v
            ax.add_patch(
                Rectangle(
                    (x0 + j * cell, y0 + (n - 1 - i) * cell),
                    cell - 0.04,
                    cell - 0.04,
                    facecolor=rgb,
                    edgecolor="none",
                    zorder=2,
                )
            )
    save(f, "concept_similarity_matrix.png")


def why_cluster():
    f, ax = fig()
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 7.2)
    gray = np.array(
        [
            (1.15, 5.5),
            (1.7, 4.7),
            (0.85, 4.2),
            (1.55, 3.5),
            (0.7, 3.0),
            (1.9, 2.6),
            (1.2, 1.9),
        ]
    )
    dots(ax, gray, "#B7AFA3", s=220, z=3)
    arrow(ax, 2.55, 3.6, 3.35, 3.6)
    blob(ax, [(4.35, 5.15), (5.15, 5.35), (4.7, 4.45)], PALE_T, pad=0.35)
    blob(ax, [(4.4, 2.35), (5.2, 2.15), (4.75, 1.55)], PALE_S, pad=0.35)
    dots(ax, [(4.35, 5.15), (5.15, 5.35), (4.7, 4.45)], TERR, s=260, z=3)
    dots(ax, [(4.4, 2.35), (5.2, 2.15), (4.75, 1.55)], SAGE, s=260, z=3)
    save(f, "concept_why_cluster.png")


def _kind_cell(ax, x, y, w, h, face, edge, title, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            facecolor=face,
            edgecolor=edge,
            lw=2.4 if edge != "none" else 0,
            zorder=0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 0.78),
            w,
            0.78,
            boxstyle="round,pad=0.01,rounding_size=0.12",
            facecolor="#FFFCF8",
            edgecolor="none",
            zorder=1,
        )
    )
    return


def why_kinds():
    f, ax = fig()
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 7.2)
    _kind_cell(ax, 0.18, 3.72, 2.95, 3.28, PALE_N, "none", "تفکیکی", NAVY)
    _kind_cell(ax, 3.28, 3.72, 2.95, 3.28, PALE_A, "none", "سلسله‌مراتبی", "#8A5A2A")
    _kind_cell(ax, 0.18, 0.18, 2.95, 3.28, PALE_T, "none", "چگالی", TERR)
    _kind_cell(ax, 3.28, 0.18, 2.95, 3.28, "#F7E4DC", TERR, "گراف", TERR)
    # centers, points around them
    ax.scatter([1.15], [5.55], s=90, c=NAVY, marker="P", zorder=4)
    ax.scatter([2.15], [4.85], s=90, c=TERR, marker="P", zorder=4)
    dots(ax, [(0.85, 5.85), (1.45, 5.9), (0.95, 5.2), (1.5, 5.25)], NAVY, s=80, z=3)
    dots(ax, [(1.85, 5.15), (2.45, 5.15), (1.9, 4.5), (2.5, 4.55)], TERR, s=80, z=3)
    # dendrogram under the header
    tree = [(3.7, 4.55), (4.35, 4.55), (5.15, 4.55), (5.8, 4.55), (4.02, 5.15), (5.48, 5.15), (4.75, 5.75)]
    links = [(0, 4), (1, 4), (2, 5), (3, 5), (4, 6), (5, 6)]
    for i, j in links:
        ax.plot([tree[i][0], tree[j][0]], [tree[i][1], tree[j][1]], color="#8A5A2A", lw=2.2, zorder=2, solid_capstyle="round")
    dots(ax, tree[:4], "#8A5A2A", s=90, z=3)
    ax.scatter([tree[4][0], tree[5][0], tree[6][0]], [tree[4][1], tree[5][1], tree[6][1]], s=40, c="#8A5A2A", zorder=3)
    # dense clump versus stray points
    rng = np.random.default_rng(2)
    dots(ax, rng.normal((1.55, 1.85), 0.16, size=(12, 2)), TERR, s=55, z=3)
    dots(ax, [(0.55, 2.45), (0.7, 0.85), (2.55, 2.4), (2.7, 0.7)], "#C9C1B4", s=36, z=2)
    # neighbors joined by edges
    g = np.array([(3.85, 2.15), (4.55, 2.35), (5.25, 2.1), (4.15, 1.15), (5.05, 1.05)])
    ax.plot([3.85, 4.55, 5.25, 3.85], [2.15, 2.35, 2.1, 2.15], color=TERR, lw=5, zorder=2, solid_capstyle="round")
    ax.plot([4.15, 5.05], [1.15, 1.05], color=SAGE, lw=5, zorder=2, solid_capstyle="round")
    ax.plot([4.55, 5.05], [2.35, 1.05], color="#C9C1B4", lw=1.3, zorder=1)
    dots(ax, g[:3], TERR, s=160, z=3)
    dots(ax, g[3:], SAGE, s=160, z=3)
    save(f, "concept_why_kinds.png")


def why_graph():
    f, ax = fig()
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 7.2)
    a = np.array([(1.5, 5.3), (2.7, 5.6), (1.8, 4.2)])
    b = np.array([(4.2, 2.5), (5.3, 2.1), (4.6, 1.3)])
    for i in range(3):
        for j in range(i + 1, 3):
            ax.plot([a[i, 0], a[j, 0]], [a[i, 1], a[j, 1]], color=NAVY, lw=7, zorder=1, solid_capstyle="round")
            ax.plot([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], color=TERR, lw=7, zorder=1, solid_capstyle="round")
    ax.plot([2.7, 4.2], [5.6, 2.5], color="#C9C1B4", lw=1.4, zorder=1)
    dots(ax, a, NAVY, s=420, z=3)
    dots(ax, b, TERR, s=420, z=3)
    save(f, "concept_why_graph.png")


def why_lemma():
    f, ax = fig()
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 7.2)
    rng = np.random.default_rng(7)
    cloud = rng.uniform((0.45, 1.3), (2.55, 5.9), size=(16, 2))
    for i, p in enumerate(cloud):
        for q in cloud[i + 1 :]:
            if rng.random() < 0.35:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#C9C1B4", lw=0.6, zorder=1)
    dots(ax, cloud, NAVY, s=36, z=2)
    arrow(ax, 2.85, 3.6, 3.55, 3.6)
    big = np.array([(4.3, 5.3), (5.5, 5.1), (4.4, 2.0), (5.55, 2.3)])
    cols = [NAVY, TERR, SAGE, SAND]
    ax.plot([4.3, 5.5], [5.3, 5.1], color="#8A8175", lw=8, zorder=1, solid_capstyle="round")
    ax.plot([4.4, 5.55], [2.0, 2.3], color="#8A8175", lw=3.2, zorder=1, solid_capstyle="round")
    ax.plot([4.3, 4.4], [5.3, 2.0], color="#8A8175", lw=1.5, zorder=1, solid_capstyle="round")
    for p, c in zip(big, cols):
        ax.scatter([p[0]], [p[1]], s=700, c=c, zorder=3, linewidths=0)
    save(f, "concept_why_lemma.png")


if __name__ == "__main__":
    why_cluster()
    why_kinds()
    why_graph()
    why_lemma()
    vertices()
    heavy()
    edges()
    partition()
    regular()
    irregular()
    density()
    reduced()
    two_paths()
    lift()
    polish()
    certs()
    fair()
    four()
    print("ok")

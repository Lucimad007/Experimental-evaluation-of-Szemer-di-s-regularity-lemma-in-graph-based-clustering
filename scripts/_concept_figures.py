"""Clear concept diagrams for the defense deck. No text."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

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
    f.savefig(OUT / name, dpi=140, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
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


def two_paths():
    f, ax = fig()
    _grid(ax, (2.2, 3.6), "color")
    ax.set_xlim(1.2, 5.5)
    ax.set_ylim(1.6, 5.6)
    save(f, "concept_path_direct.png")

    f, ax = fig()
    _grid(ax, (2.2, 5.55), "plain")
    arrow(ax, 3.35, 4.05, 3.35, 3.45)
    ax.add_patch(Circle((2.35, 2.75), 0.42, facecolor=TERR, zorder=3))
    ax.add_patch(Circle((4.35, 2.75), 0.42, facecolor=SAGE, zorder=3))
    ax.plot([2.77, 3.93], [2.75, 2.75], color="#8A8175", lw=9, solid_capstyle="butt", zorder=2)
    arrow(ax, 3.35, 2.15, 3.35, 1.55)
    _grid(ax, (2.2, 0.05), "color")
    ax.set_xlim(1.2, 5.5)
    ax.set_ylim(-1.5, 7.15)
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
    whole = np.array([[1.5, 4.4], [2.3, 4.6], [1.6, 3.5], [2.4, 3.6], [1.8, 2.6], [2.5, 2.7]])
    blob(ax, whole, PALE_N, pad=0.5)
    dots(ax, whole[:3], TERR, s=180)
    dots(ax, whole[3:], NAVY, s=180)
    arrow(ax, 3.5, 3.6, 4.3, 3.6)
    top = whole[:3] + np.array([4.2, 0.7])
    bot = whole[3:] + np.array([4.2, -0.7])
    blob(ax, top, PALE_T, pad=0.4)
    blob(ax, bot, PALE_N, pad=0.4)
    dots(ax, top, TERR, s=160)
    dots(ax, bot, NAVY, s=160)
    dots(ax, np.array([[8.6, 4.8], [9.1, 4.2]]), "#B7B1A6", s=120)
    ax.set_xlim(0.4, 10.0)
    ax.set_ylim(1.2, 5.8)
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


if __name__ == "__main__":
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

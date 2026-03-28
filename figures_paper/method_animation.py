from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

from method_figure import (
    Arrow3D,
    OUTPUT_DIR,
    draw_basis,
    draw_plane,
    draw_sphere,
    nn_subspace_alignment,
    normalize_rows,
    resolve_save_path,
    set_equal_3d,
    simulate_modalities,
)


ANIMATION_DIR = OUTPUT_DIR / "animation"


def lerp(a, b, t):
    return (1.0 - t) * a + t * b


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def as_numpy(x):
    return x.detach().cpu().numpy()


def orthonormalize_columns(basis):
    q1 = basis[:, 0]
    q1 = q1 / q1.norm().clamp_min(1e-8)
    q2 = basis[:, 1] - torch.dot(basis[:, 1], q1) * q1
    q2 = q2 / q2.norm().clamp_min(1e-8)
    return torch.stack([q1, q2], dim=1)


def make_residual_target_basis(wx, residual_angle=0.18):
    normal_x = torch.linalg.cross(wx[:, 0], wx[:, 1])
    tilted = torch.stack([wx[:, 0], wx[:, 1] + residual_angle * normal_x], dim=1)
    return orthonormalize_columns(tilted)


def stage_progress(frame, start, length):
    if frame < start:
        return 0.0
    if frame >= start + length:
        return 1.0
    return smoothstep((frame - start) / max(length - 1, 1))


def shifted_progress(progress, lead=0.0, lag=0.0):
    adjusted = (progress + lead - lag) / max(1.0 + lead - lag, 1e-8)
    return float(np.clip(adjusted, 0.0, 1.0))


def draw_gap_arrow(ax, start, end, color="black", alpha=0.95, lw=2.0):
    start_np = as_numpy(start)
    end_np = as_numpy(end)
    arrow = Arrow3D(
        [start_np[0], end_np[0]],
        [start_np[1], end_np[1]],
        [start_np[2], end_np[2]],
        mutation_scale=16,
        lw=lw,
        arrowstyle="-|>",
        color=color,
        alpha=alpha,
    )
    ax.add_artist(arrow)


def draw_centroid(ax, point, color, label):
    pt = as_numpy(point)
    ax.scatter(*pt, s=72, color=color, marker="X", depthshade=False)
    ax.text(pt[0], pt[1], pt[2], label, color=color)


def build_animation_frames(X, Y, out):
    Xc_shifted = out["Xc"] + out["mu_X"]
    Yc_shifted = out["Yc"] + out["mu_Y"]
    Yrot_shifted = out["Yrot"] + out["mu_X"]

    return {
        "X_orig": X,
        "Y_orig": Y,
        "X_centered": Xc_shifted,
        "Y_centered": Yc_shifted,
        "X_rot": out["Xc"] + out["mu_X"],
        "Y_rot": Yrot_shifted,
        "Y_al": out["Yal"],
        "X_hat": normalize_rows(X),
        "Y_hat": out["Yhat"],
    }


def draw_cloud(ax, points, color, alpha=0.72, size=18):
    pts = as_numpy(points)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, alpha=alpha, color=color, depthshade=False)

def make_animation(
    X,
    Y,
    out,
    save_path="nn_subspace_alignment_animation.gif",
    fps=20,
    stage_frames=32,
):
    states = build_animation_frames(X, Y, out)
    total_frames = 5 * stage_frames

    fig = plt.figure(figsize=(7.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")

    titles = [
        r"$\mathrm{(1)\ Original\ modalities}$",
        r"$\mathrm{(2)\ Centering}$",
        r"$\mathrm{(3)\ Principal\ subspaces}$",
        r"$\mathrm{(4)\ Subspace\ alignment}$",
        r"$\mathrm{(5)\ Reconstruction\ and\ normalization}$",
    ]
    subtitles = [
        r"$X,\; Y \quad \mathrm{with\ modality\ gap}\; (\mu_Y \rightarrow \mu_X)$",
        r"$X_c = X - \mu_X,\; Y_c = Y - \mu_Y$",
        r"$W_X,\; W_Y \quad \mathrm{from\ centered\ embeddings}$",
        r"$Y_{\mathrm{rot}} = Y_c W_Y \Phi^\ast W_X^\top,\; \Phi^\ast = W_Y^\top W_X$",
        r"$Y_{\mathrm{al}} = Y_{\mathrm{rot}} + \mu_X,\; \hat{y}_{\mathrm{al}} = y_{\mathrm{al}}/\|y_{\mathrm{al}}\|_2$",
    ]

    mu_x = out["mu_X"]
    mu_y = out["mu_Y"]
    wx = out["WX"]
    wy = out["WY"]
    w_red_target = make_residual_target_basis(wx, residual_angle=0.18)

    def update(frame):
        ax.cla()
        draw_sphere(ax, alpha=0.05)
        set_equal_3d(ax, lim=1.15)

        p1 = stage_progress(frame, 0 * stage_frames, stage_frames)
        p2 = stage_progress(frame, 1 * stage_frames, stage_frames)
        p3 = stage_progress(frame, 2 * stage_frames, stage_frames)
        p4 = stage_progress(frame, 3 * stage_frames, stage_frames)
        p5 = stage_progress(frame, 4 * stage_frames, stage_frames)
        p4_plane = p4
        p4_points = shifted_progress(p4, lag=0.18)

        X_now = lerp(states["X_orig"], states["X_centered"], p1)
        Y_now = lerp(states["Y_orig"], states["Y_centered"], p1)
        X_now = lerp(X_now, states["X_rot"], p2)
        Y_now = lerp(Y_now, states["Y_centered"], p2)
        X_now = lerp(X_now, states["X_rot"], p3)
        Y_now = lerp(Y_now, states["Y_centered"], p3)
        Y_now = lerp(Y_now, states["Y_al"], p4_points)
        X_now = lerp(X_now, states["X_hat"], p5)
        Y_now = lerp(Y_now, states["Y_hat"], p5)

        draw_cloud(ax, X_now, "tab:blue")
        draw_cloud(ax, Y_now, "tab:red")

        if frame < stage_frames:
            title = titles[0]
            subtitle = subtitles[0]
            draw_centroid(ax, mu_x, "tab:blue", r"$\mu_X$")
            draw_centroid(ax, mu_y, "tab:red", r"$\mu_Y$")
            draw_gap_arrow(ax, mu_y, mu_x)
        elif frame < 2 * stage_frames:
            title = titles[1]
            subtitle = subtitles[1]
            draw_centroid(ax, mu_x, "tab:blue", r"$\mu_X$")
            draw_centroid(ax, mu_y, "tab:red", r"$\mu_Y$")
            draw_basis(ax, mu_x, wx, color="tab:blue", label=r"$W_X$", scale=0.42)
            draw_basis(ax, mu_y, wy, color="tab:red", label=r"$W_Y$", scale=0.42)
        elif frame < 3 * stage_frames:
            title = titles[2]
            subtitle = subtitles[2]
            draw_centroid(ax, mu_x, "tab:blue", r"$\mu_X$")
            draw_centroid(ax, mu_y, "tab:red", r"$\mu_Y$")
            draw_plane(ax, mu_x, wx, radius=0.58, color="tab:blue", alpha=0.12)
            draw_plane(ax, mu_y, wy, radius=0.58, color="tab:red", alpha=0.22)
            draw_basis(ax, mu_x, wx, color="tab:blue", label=r"$W_X$", scale=0.5)
            draw_basis(ax, mu_y, wy, color="tab:red", label=r"$W_Y$", scale=0.5)
        elif frame < 4 * stage_frames:
            title = titles[3]
            subtitle = subtitles[3]
            moving_center = lerp(mu_y, mu_x, p4_plane)
            moving_basis = orthonormalize_columns(lerp(wy, w_red_target, p4_plane))
            draw_centroid(ax, mu_x, "tab:blue", r"$\mu_X$")
            draw_centroid(ax, mu_y, "tab:red", r"$\mu_Y$")
            draw_plane(ax, mu_x, wx, radius=0.58, color="tab:blue", alpha=0.10)
            draw_plane(ax, moving_center, moving_basis, radius=0.58, color="tab:red", alpha=0.26)
            draw_basis(ax, mu_x, wx, color="tab:blue", label=r"$W_X$", scale=0.48)
            draw_basis(ax, moving_center, moving_basis, color="tab:red", label=r"$W_Y \rightarrow \widetilde{W}_Y$", scale=0.48)
        else:
            title = titles[4]
            subtitle = subtitles[4]
            draw_centroid(ax, mu_x, "tab:blue", r"$\mu_X$")
            draw_plane(ax, mu_x, wx, radius=0.58, color="tab:blue", alpha=0.10)
            draw_plane(ax, mu_x, w_red_target, radius=0.58, color="tab:red", alpha=0.24)
            draw_basis(ax, mu_x, wx, color="tab:blue", label=r"$W_X$", scale=0.48)
            draw_basis(ax, mu_x, w_red_target, color="tab:red", label=r"$\widetilde{W}_Y$", scale=0.48)

        ax.text2D(0.5, 0.97, title, transform=ax.transAxes, ha="center", va="top")
        ax.text2D(0.5, 0.92, subtitle, transform=ax.transAxes, ha="center", va="top")
        ax.text2D(
            0.5,
            0.04,
            r"$\mathrm{Blue}: X \qquad \mathrm{Red}: Y$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, blit=False)
    output_path = resolve_save_path(Path("animation") / save_path)
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    X, Y, labels = simulate_modalities(n=150, noise=0.04, seed=7, num_clusters=2)
    out = nn_subspace_alignment(X, Y, dsub=2)
    save_path = make_animation(X, Y, out)
    print(f"Saved animation to {save_path}")

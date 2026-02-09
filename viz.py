import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import time

try:
    import cmocean
except ImportError:
    cmocean = None

# 这些由主程序注入
WIDTH = HEIGHT = MAX_SPEED = NUM_SPECIES_A = COLLISION_DIST = None

def run_animation(sim, frames=600, interval=1, color_by="role_stamina", stride=1):
    """animation, speed box visualization

    Args:
        sim (_type_): a BoidsSim object
        frames (int, optional): _description_. Defaults to 600.
        interval (int, optional): _description_. Defaults to 1.
        color_by (str, optional): _description_. Defaults to "role_stamina".
        stride (int, optional): _description_. Defaults to 1.
        global: WIDTH, HEIGHT, MAX_SPEED, NUM_SPECIES_A
    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    # ---- helpers: role masks ----
    prey_mask = (sim.species_id < 1)
    pred_mask = ~prey_mask

    # ---- 1) prepare color values / cmap / norm / discrete labels ----
    is_discrete = False
    tick_locs = None
    tick_labels = None

    if color_by in ("role", "role_stamina"):
        is_discrete = True

        # v 是分类编号
        # role: prey=0, predator=1
        # role_stamina: prey=0, predator(stamina=False)=1, predator(stamina=True)=2
        v = np.zeros(sim.n, dtype=int)

        if color_by == "role":
            v[pred_mask] = 1
            n_cat = 2
            tick_locs = [0, 1]
            tick_labels = ["prey", "predator"]

        else:  # role_stamina
            # predator: stamina False -> 1, True -> 2
            # 注意：prey stamina 无意义，保持 0
            if not hasattr(sim, "stamina"):
                # 没有 stamina 就全部当 False
                v[pred_mask] = 1
            else:
                v[pred_mask] = np.where(sim.stamina[pred_mask], 2, 1)

            n_cat = 3
            tick_locs = [0, 1, 2]
            tick_labels = ["prey", "predator (stamina=0)", "predator (stamina=1)"]

        cmap = plt.get_cmap("tab10", n_cat)
        bounds = np.arange(-0.5, n_cat + 0.5, 1.0)
        norm = mpl.colors.BoundaryNorm(bounds, ncolors=n_cat)

    elif color_by == "species_id":
        is_discrete = True
        v = sim.species_id.astype(int)
        n_species = int(v.max()) + 1

        cmap = plt.get_cmap("tab10", n_species)
        bounds = np.arange(-0.5, n_species + 0.5, 1.0)
        norm = mpl.colors.BoundaryNorm(bounds, ncolors=n_species)

        tick_locs = list(range(n_species))
        tick_labels = [f"species {k}" for k in range(n_species)]

    else:
        # 连续字段：从 structured attrs 取
        # 对 None -> NaN（方便归一化与弱化显示）
        if color_by not in sim.attrs.dtype.names:
            raise ValueError(f"Unknown color_by='{color_by}'. "
                            f"Allowed: role, role_stamina, species_id, "
                            f"or any attrs field: {sim.attrs.dtype.names}")

        raw = sim.attrs[color_by]  # object array (可能含 None)
        v = np.array([np.nan if x is None else float(x) for x in raw], dtype=float)

        v_valid = v[np.isfinite(v)]
        if v_valid.size == 0:
            # 全是 None：退回 role_stamina
            return sim.run_animation(frames=frames, interval=interval, color_by="role_stamina", stride=stride)

        norm = Normalize(vmin=float(v_valid.min()), vmax=float(v_valid.max()))

        # 连续色带：海洋风格
        try:
            import cmocean
            cmap = cmocean.cm.deep
        except Exception:
            cmap = mpl.cm.viridis  # fallback

    # ---- 2) figure layout ----
    fig, (ax, ax_box) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [3.2, 1.0]}
    )

    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_title(f"Boids (N={sim.total}) | color_by={color_by} | prey={int(prey_mask.sum())}, predator={int(pred_mask.sum())}")

    # ---- initial colors (dead alpha=0) ----
    colors = cmap(norm(v))
    colors[:, 3] = sim.alive.astype(float)

    # 连续变量下：没有该属性（NaN）则弱化显示
    if not is_discrete:
        nan_mask = ~np.isfinite(v)
        colors[nan_mask, 3] *= 0.25

    quiver = ax.quiver(
        sim.positions[:, 0], sim.positions[:, 1],
        sim.velocities[:, 0], sim.velocities[:, 1],
        color=colors, angles="xy", scale_units="xy", scale=1,
        animated=True
    )
    
    # --- lifespan text (below main axis) ---
    lifespan_text = ax.text(
        0.5, 0.01, "",               # x center, slightly below axis
        transform=ax.transAxes,
        ha="center", va="top",
        clip_on = False
    )

    # ---- colorbar ----
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_by, rotation=90)

    if is_discrete and tick_locs is not None:
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(tick_labels)

    # ---- speed boxplot (live only) ----
    ax_box.set_title("Live speed boxplot")
    ax_box.set_xlim(0, 1)
    ax_box.set_ylim(0, MAX_SPEED)
    ax_box.set_xticks([])
    ax_box.set_ylabel("speed |v|")

    x_center = 0.5
    box_width = 0.55
    cap_width = 0.35

    box_rect = patches.Rectangle(
        (x_center - box_width / 2, 0.0), box_width, 0.0,
        fill=False, linewidth=1.5
    )
    ax_box.add_patch(box_rect)

    median_line = Line2D([x_center - box_width / 2, x_center + box_width / 2], [0.0, 0.0], linewidth=1.5)
    whisker_line = Line2D([x_center, x_center], [0.0, 0.0], linewidth=1.5)
    cap_low = Line2D([x_center - cap_width / 2, x_center + cap_width / 2], [0.0, 0.0], linewidth=1.5)
    cap_high = Line2D([x_center - cap_width / 2, x_center + cap_width / 2], [0.0, 0.0], linewidth=1.5)

    ax_box.add_line(median_line)
    ax_box.add_line(whisker_line)
    ax_box.add_line(cap_low)
    ax_box.add_line(cap_high)

    stats_text = ax_box.text(0.02, 0.98, "", transform=ax_box.transAxes, va="top")

    def _update_speed_box():
        alive_idx = np.flatnonzero(sim.alive)
        if alive_idx.size == 0:
            box_rect.set_y(0.0); box_rect.set_height(0.0)
            median_line.set_ydata([0.0, 0.0])
            whisker_line.set_ydata([0.0, 0.0])
            cap_low.set_ydata([0.0, 0.0])
            cap_high.set_ydata([0.0, 0.0])
            stats_text.set_text("alive=0")
            return

        speeds = np.linalg.norm(sim.velocities[alive_idx], axis=1)
        vmin = float(speeds.min())
        vmax = float(speeds.max())
        q1, med, q3 = np.percentile(speeds, [25, 50, 75])

        box_rect.set_y(float(q1))
        box_rect.set_height(float(q3 - q1))
        median_line.set_ydata([float(med), float(med)])
        whisker_line.set_ydata([vmin, vmax])
        cap_low.set_ydata([vmin, vmin])
        cap_high.set_ydata([vmax, vmax])

        # 额外显示 predator stamina 状态
        if hasattr(sim, "stamina"):
            pred_alive = np.flatnonzero(sim.alive & pred_mask)
            if pred_alive.size:
                s = bool(sim.stamina[pred_alive[0]])
                stats_text.set_text(f"alive={alive_idx.size}\nmax={vmax:.2f}\npred_stamina={int(s)}\nframe={sim.frame}\nduration={sim.duration[NUM_SPECIES_A]}")
            else:
                stats_text.set_text(f"alive={alive_idx.size}\nmax={vmax:.2f}\npred_stamina=NA\nframe={sim.frame}")
        else:
            stats_text.set_text(f"alive={alive_idx.size}\nmax={vmax:.2f}")

    # autosave on close
    if sim.autosave_path is not None:
        def _on_close(_evt):
            sim.save(sim.autosave_path)
            print(f"[autosave] saved to {sim.autosave_path}")
        fig.canvas.mpl_connect("close_event", _on_close)

    # ---- step targets for stride skipping ----
    step_targets = range(0, int(frames) + 1, int(stride))

    def animate(step_target):
        while sim.frame < step_target and not sim.all_dead():
            sim.step()
            time.sleep(0.1)

        # refresh v for dynamic colorings (role_stamina changes over time)
        nonlocal v, colors

        if color_by in ("role", "role_stamina"):
            # v depends on stamina potentially
            v = np.zeros(sim.n, dtype=int)
            if color_by == "role":
                v[pred_mask] = 1
            else:
                if hasattr(sim, "stamina"):
                    v[pred_mask] = np.where(sim.stamina[pred_mask], 2, 1)
                else:
                    v[pred_mask] = 1

        # update quiver
        quiver.set_offsets(sim.positions)
        quiver.set_UVC(sim.velocities[:, 0], sim.velocities[:, 1])

        colors = cmap(norm(v))
        colors[:, 3] = sim.alive.astype(float)

        if not is_discrete:
            nan_mask = ~np.isfinite(v)
            colors[nan_mask, 3] *= 0.25

        # ---- highlight locked prey in red ----
        if hasattr(sim, "locked"):
            locked_ids = sim.locked[sim.locked >= 0]   # 所有被锁定的 prey index

            if locked_ids.size > 0:
                # 去重（以防多个 predator 锁同一只）
                locked_ids = np.unique(locked_ids)

                # 设置为红色（RGBA）
                # R=1,G=0,B=0, alpha 保持原 alive
                colors[locked_ids, 0] = 1.0
                colors[locked_ids, 1] = 0.0
                colors[locked_ids, 2] = 0.0
                # alpha 已经由 alive 控制，无需再改

        quiver.set_color(colors)

        _update_speed_box()
        
        lifespan_text.set_text(f"t = {sim.frame}")

        return (quiver, box_rect, median_line, whisker_line, cap_low, cap_high, stats_text, lifespan_text)

    ani = FuncAnimation(fig, animate, frames=step_targets, interval=interval, blit=True)
    plt.show()
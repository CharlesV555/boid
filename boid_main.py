import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib as mpl
import cmocean
# -----------------------------
# Global settings
# -----------------------------
NUM_BOIDS = 120
WIDTH, HEIGHT = 100, 100
MAX_SPEED = 10

# 进化/初始化的参数范围（可按需调整）
# 可以一步一步固定着看
ATTR_NAMES = [
    "alignment_weight",
    "cohesion_weight",
    "separation_weight",
    "neighbor_radius",
    "separation_radius",
]
ATTR_RANGES = {
    "alignment_weight": (0.05, 0.05),
    "cohesion_weight": (0.01, 0.05),
    "separation_weight": (0.1, 0.1),
    "neighbor_radius": (15.0, 15.0),
    "separation_radius": (5.0, 5.0),
}

# 碰撞死亡阈值：距离 <= 1.0 撞死
COLLISION_DIST = 0.2

# 可视化：选择用哪个属性上色（五选一）
COLOR_BY = "cohesion_weight"   # 可以设置为 'alignment_weight', 'cohesion_weight', 'separation_weight', 'neighbor_radius', 'separation_radius'


def limit_speed(v, max_speed):
    speed = np.linalg.norm(v)
    if speed > max_speed and speed > 1e-12:
        return v / speed * max_speed
    return v


def random_attrs(n, rng):
    """随机初始化每只鸟的个体属性 attrs: (n,5)"""
    cols = []
    for name in ATTR_NAMES:
        lo, hi = ATTR_RANGES[name]
        cols.append(rng.uniform(lo, hi, size=n))
    return np.column_stack(cols)


class BoidsSim:
    def __init__(self, n=NUM_BOIDS, seed=None,
                 init_from=None,
                 autosave_path=None,
                 autosave_meta=None):
        """
        init_from:
          - None: 随机初始化 positions/velocities/attrs
          - dict: 来自 load_population() 的字典，直接用其中的状态
        autosave_path:
          - 关闭窗口时自动存档（npz）
        """
        self.n = n
        self.rng = np.random.default_rng(seed)

        if init_from is None:
            self.positions = self.rng.random((n, 2)) * np.array([WIDTH, HEIGHT], dtype=float)
            self.velocities = (self.rng.random((n, 2)) - 0.5) * MAX_SPEED
            self.attrs = random_attrs(n, self.rng)
            self.alive = np.ones(n, dtype=bool)
            self.lifespan = np.zeros(n, dtype=np.int32)
            self.meta = {"generation": 0}
        else:
            self.positions = init_from["positions"].copy()
            self.velocities = init_from["velocities"].copy()
            self.attrs = init_from["attrs"].copy()
            self.alive = init_from["alive"].copy()
            self.lifespan = init_from["lifespan"].copy()
            self.meta = dict(init_from.get("meta", {}))

        self.frame = 0
        self.autosave_path = autosave_path
        self.autosave_meta = autosave_meta if autosave_meta is not None else {}

    def step(self):
        """单步更新（只对 alive 的鸟计算邻居/速度/位置），然后做碰撞检测并更新死亡。"""
        new_vel = self.velocities.copy()

        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) == 0:
            self.frame += 1
            return

        # 速度更新：对每个活鸟 i，扫描其他活鸟作为邻居
        for i in alive_idx:
            pos_i = self.positions[i]
            vel_i = self.velocities[i]
            a_w, c_w, s_w, n_r, s_r = self.attrs[i]

            alignment = np.zeros(2, dtype=float)
            cohesion = np.zeros(2, dtype=float)
            separation = np.zeros(2, dtype=float)
            count = 0

            for j in alive_idx:
                if i == j:
                    continue
                offset = self.positions[j] - pos_i
                dist = np.linalg.norm(offset)
                if dist < n_r:
                    alignment += self.velocities[j]
                    cohesion += self.positions[j]
                    if dist < s_r:
                        # 避免 dist=0 的数值问题
                        if dist > 1e-12:
                            separation -= offset / dist
                    count += 1

            if count > 0:
                alignment = alignment / count - vel_i
                cohesion = cohesion / count - pos_i

            vel_i = vel_i + a_w * alignment + c_w * cohesion + s_w * separation
            vel_i = limit_speed(vel_i, MAX_SPEED)
            new_vel[i] = vel_i

        # 位置更新（死鸟不动）
        self.positions[self.alive] = (self.positions[self.alive] + new_vel[self.alive]) % np.array([WIDTH, HEIGHT])
        self.velocities[self.alive] = new_vel[self.alive]

        # 生存帧数累加
        self.lifespan[self.alive] += 1

        # 碰撞检测（距离 <= COLLISION_DIST 的两只活鸟都死亡）
        # self._apply_collisions()

        self.frame += 1

    def _apply_collisions(self):
        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) < 2:
            return

        d2_thr = COLLISION_DIST ** 2
        # O(K^2) 碰撞检测：K=活鸟数量，N=120 时可接受
        for ii in range(len(alive_idx)):
            i = alive_idx[ii]
            if not self.alive[i]:
                continue
            pi = self.positions[i]
            for jj in range(ii + 1, len(alive_idx)):
                j = alive_idx[jj]
                if not self.alive[j]:
                    continue
                pj = self.positions[j]
                dx, dy = (pj - pi)
                if (dx * dx + dy * dy) <= d2_thr:
                    self.alive[i] = False
                    self.alive[j] = False

                    # 可选：死后速度清零，避免显示箭头残留
                    self.velocities[i] = 0.0
                    self.velocities[j] = 0.0
    
    def alive_count(self):
        return int(self.alive.sum())

    def all_dead(self):
        return self.alive_count() == 0
    
    def run_animation(self, frames=600, interval=20, color_by=COLOR_BY, stride=1):
        """
        frames: 最大模拟步数（max_steps），注意现在语义是“模拟步上限”，不是“回调次数”
        stride: 跳帧步长；stride=10 表示每显示一帧，模拟推进 10 步
        """
        # 颜色映射基于 attrs 的某一列；死鸟 alpha=0
        col_idx = ATTR_NAMES.index(color_by)
        v = self.attrs[:, col_idx]
        norm = Normalize(vmin=float(v.min()), vmax=float(v.max()))
        cmap = cmocean.cm.deep

        # --- 1x2 子图布局（左：boids，右：速度箱型图）---
        fig, (ax, ax_box) = plt.subplots(
            1, 2, figsize=(11, 5),
            gridspec_kw={"width_ratios": [3.2, 1.0]}
        )

        # --- 左图：boids ---
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)

        # 颜色意义说明（你可以按需微调文字）
        meaning_map = {
            "alignment_weight": "Alignment weight (steer to match neighbors' velocity)",
            "cohesion_weight": "Cohesion weight (steer toward neighbors' center)",
            "separation_weight": "Separation weight (steer away to avoid crowding)",
            "neighbor_radius": "Neighbor radius (perception range)",
            "separation_radius": "Separation radius (repulsion range)",
        }
        meaning = meaning_map.get(color_by, color_by)

        ax.set_title(f"Boids | color_by={color_by}\n{meaning} | Collision<= {COLLISION_DIST} => death")

        colors = cmap(norm(v))
        colors[:, 3] = self.alive.astype(float)

        quiver = ax.quiver(
            self.positions[:, 0], self.positions[:, 1],
            self.velocities[:, 0], self.velocities[:, 1],
            color=colors, angles='xy', scale_units='xy', scale=1
        )

        # --- 新增：颜色标尺（colorbar）---
        # 用 ScalarMappable 挂上 norm+cmap，给 colorbar 用
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # Matplotlib 需要一个 array 占位

        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{color_by}", rotation=90)
        # 如需更明确，可把 meaning 放到 label：
        # cbar.set_label(f"{color_by} - {meaning}", rotation=90)
    
    
        # --- 右图：速度箱型图（仅活鸟）---
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

        median_line = Line2D(
            [x_center - box_width / 2, x_center + box_width / 2],
            [0.0, 0.0],
            linewidth=1.5
        )
        whisker_line = Line2D([x_center, x_center], [0.0, 0.0], linewidth=1.5)
        cap_low = Line2D([x_center - cap_width / 2, x_center + cap_width / 2], [0.0, 0.0], linewidth=1.5)
        cap_high = Line2D([x_center - cap_width / 2, x_center + cap_width / 2], [0.0, 0.0], linewidth=1.5)

        ax_box.add_line(median_line)
        ax_box.add_line(whisker_line)
        ax_box.add_line(cap_low)
        ax_box.add_line(cap_high)

        stats_text = ax_box.text(0.02, 0.98, "", transform=ax_box.transAxes, va="top")

        def _update_speed_box():
            alive_idx = np.where(self.alive)[0]
            if len(alive_idx) == 0:
                box_rect.set_y(0.0); box_rect.set_height(0.0)
                median_line.set_ydata([0.0, 0.0])
                whisker_line.set_ydata([0.0, 0.0])
                cap_low.set_ydata([0.0, 0.0])
                cap_high.set_ydata([0.0, 0.0])
                stats_text.set_text("alive=0")
                return

            speeds = np.linalg.norm(self.velocities[alive_idx], axis=1)
            vmin = float(speeds.min())
            vmax = float(speeds.max())
            q1, med, q3 = np.percentile(speeds, [25, 50, 75])

            box_rect.set_y(float(q1))
            box_rect.set_height(float(q3 - q1))
            median_line.set_ydata([float(med), float(med)])

            # whisker 用 min/max（如需 Tukey whisker 再改）
            whisker_line.set_ydata([vmin, vmax])
            cap_low.set_ydata([vmin, vmin])
            cap_high.set_ydata([vmax, vmax])

            stats_text.set_text(f"alive={len(alive_idx)}\nmax={vmax:.2f}")

        # 可选：窗口关闭自动存档
        if self.autosave_path is not None:
            def _on_close(_evt):
                self.save(self.autosave_path)
                print(f"[autosave] saved to {self.autosave_path}")
            fig.canvas.mpl_connect("close_event", _on_close)

        # 关键：把 FuncAnimation 的 frames 变成“要显示的模拟步编号”
        # 这样 stride 就是“跳帧查看”的控制旋钮
        step_targets = range(0, int(frames) + 1, int(stride))

        def animate(step_target):
            # 将模拟推进到指定步：lifespan 更新发生在 self.step() 内，因此与 step_target 同步
            while self.frame < step_target and not self.all_dead():
                self.step()

            # 更新 boids
            quiver.set_offsets(self.positions)
            quiver.set_UVC(self.velocities[:, 0], self.velocities[:, 1])

            colors = cmap(norm(v))
            colors[:, 3] = self.alive.astype(float)
            quiver.set_color(colors)

            # 更新箱型图
            _update_speed_box()

            return (quiver, box_rect, median_line, whisker_line, cap_low, cap_high, stats_text)

        ani = FuncAnimation(fig, animate, frames=step_targets, interval=interval, blit=True)
        plt.show()
   

# -----------------------------
# Example workflows
# -----------------------------
if __name__ == "__main__":
    # ====== 1) 跑第0代：随机初始化，关闭窗口后自动存档 ======
    sim0 = BoidsSim(
        n=NUM_BOIDS,
        seed=42,
        init_from=None,
        autosave_path="gen0.npz",
        autosave_meta={"generation": 0}
    )
    sim0.run_animation(frames=8000, interval=1, stride=1,color_by=COLOR_BY)
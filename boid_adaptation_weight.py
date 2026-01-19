import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib as mpl


# -----------------------------
# Global settings
# -----------------------------
NUM_BOIDS = 120
WIDTH, HEIGHT = 100, 100
MAX_SPEED = 10
MIN_SPEED = 0.01

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
    "cohesion_weight": (0.01, 0.01),
    "separation_weight": (0.1, 0.1),
    "neighbor_radius": (10.0, 10.0),
    "separation_radius": (3.0, 5.0),
}

# 碰撞死亡阈值：距离 <= 1.0 撞死
COLLISION_DIST = 0.2

# 可视化：选择用哪个属性上色（五选一）
COLOR_BY = "separation_radius"   # 可以设置为 'alignment_weight', 'cohesion_weight', 'separation_weight', 'neighbor_radius', 'separation_radius'


def limit_speed(min_speed, v, max_speed):
    speed = np.linalg.norm(v)
    if speed > max_speed and speed > 1e-12:
        return v / speed * max_speed
    elif speed < min_speed and speed > 1e-12:
        return v / speed * min_speed
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
        
        self.trace_boid = None # 记录速度分量构成
        self.trace = []   # list of dict，每步一条
        self.track_id = None
        self._last_dv_align = np.zeros(2, dtype=float)
        self._last_dv_coh   = np.zeros(2, dtype=float)
        self._last_dv_sep   = np.zeros(2, dtype=float)
        self._last_factor   = 1.0
        self._last_mean_dist = 1.0
    
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
            sep_count = 0
            mean_dist = 0

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
                            separation -= offset / dist # 方向，单位向量
                            mean_dist += dist
                            sep_count += 1
                    count += 1

            if count > 0:
                alignment = alignment / count - vel_i
                cohesion = cohesion / count - pos_i # 位置的均值减去自己位置，指向前进方向
            if  sep_count > 0:
                mean_dist /= sep_count
            else: mean_dist = 1
            
            ##########
            # 速度合成
            factor = (1.0 + np.log(s_r / (max(mean_dist, s_r))))
            
            dv_align = a_w * alignment
            dv_coh   = c_w * cohesion
            dv_sep   = s_w * separation * factor
            
            vel_before = vel_i.copy()
            vel_i = vel_i + dv_align + dv_coh + dv_sep            
            vel_i = limit_speed(MIN_SPEED, vel_i, MAX_SPEED)
            new_vel[i] = vel_i
            ##########
            
            # 记录速度分量，拿到每一步三项对速度变化的向量贡献。
            # 记录追踪鸟当步贡献
            if (self.track_id is not None) and (i == self.track_id):
                self._last_dv_align = dv_align.copy()
                self._last_dv_coh   = dv_coh.copy()
                self._last_dv_sep   = dv_sep.copy()
                self._last_factor   = float(factor)
                self._last_mean_dist = float(mean_dist)
                
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
    
    def set_trace_boid(self, boid_id: int):
        self.trace_boid = int(boid_id)
        self.trace.clear()

    def clear_trace(self):
        self.trace.clear()
        
  
    # def run_animation(self, frames=600, interval=20, color_by=COLOR_BY, stride=1):
        """
        frames: 最大模拟步数（max_steps），注意现在语义是“模拟步上限”，不是“回调次数”
        stride: 跳帧步长；stride=10 表示每显示一帧，模拟推进 10 步
        """
        # 颜色映射基于 attrs 的某一列；死鸟 alpha=0
        col_idx = ATTR_NAMES.index(color_by)
        v = self.attrs[:, col_idx]
        norm = Normalize(vmin=float(v.min()), vmax=float(v.max()))
        cmap = cm.inferno

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

    def run_animation(self, frames=600, interval=20, stride=1, track_id=0,
                    vector_gain=6.0, ripple_gain=8.0, ripple_sigma_deg=18.0):
        """
        - 不按 COLOR_BY 上色：所有活鸟灰色，追踪鸟红色
        - 同步跳帧：FuncAnimation 的 frames 是“目标模拟步编号”，stride 控制跳帧查看
        - 动态展示追踪鸟三项贡献：主图三根箭头 + 右侧极坐标波纹图
        """
        import matplotlib.patches as patches
        from matplotlib.lines import Line2D
        import matplotlib as mpl

        # 设置追踪鸟
        self.track_id = int(track_id)

        # --- 布局：主图 + 速度箱线图 + 极坐标波纹图 ---
        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 3, width_ratios=[3.2, 1.0, 1.2])
        ax = fig.add_subplot(gs[0, 0])
        ax_box = fig.add_subplot(gs[0, 1])
        ax_pol = fig.add_subplot(gs[0, 2], projection="polar")

        # -----------------------------
        # 主图：Boids（灰色）+ 追踪鸟（红色）+ 三箭头
        # -----------------------------
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.set_title("Boids | tracked boid in red | contribution vectors shown")

        # 灰色 RGBA（alpha随 alive）
        base_rgb = np.array([0.25, 0.25, 0.25, 1.0], dtype=float)
        colors = np.tile(base_rgb, (self.n, 1))
        colors[:, 3] = self.alive.astype(float)

        quiver = ax.quiver(
            self.positions[:, 0], self.positions[:, 1],
            self.velocities[:, 0], self.velocities[:, 1],
            color=colors, angles='xy', scale_units='xy', scale=1
        )

        # 追踪鸟红点
        track_scatter = ax.scatter([0], [0], c="red", s=60, zorder=5)
        track_label = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

        # 三项贡献箭头（在主图上，从追踪鸟位置发出）
        # 颜色：alignment(蓝)、cohesion(绿)、separation(橙)
        contrib_colors = ["tab:blue", "tab:green", "tab:orange"]
        contrib_quiver = ax.quiver(
            [0, 0, 0], [0, 0, 0],
            [0, 0, 0], [0, 0, 0],
            color=contrib_colors, angles='xy', scale_units='xy', scale=1, zorder=6
        )
        ax.legend(
            handles=[
                mpl.lines.Line2D([0], [0], color="tab:blue", lw=2, label="alignment Δv"),
                mpl.lines.Line2D([0], [0], color="tab:green", lw=2, label="cohesion Δv"),
                mpl.lines.Line2D([0], [0], color="tab:orange", lw=2, label="separation Δv"),
            ],
            loc="lower left"
        )

        # -----------------------------
        # 右侧1：活鸟速度箱线图（同你之前自绘）
        # -----------------------------
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

        box_text = ax_box.text(0.02, 0.98, "", transform=ax_box.transAxes, va="top")

        def _update_speed_box():
            alive_idx = np.where(self.alive)[0]
            if len(alive_idx) == 0:
                box_rect.set_y(0.0); box_rect.set_height(0.0)
                median_line.set_ydata([0.0, 0.0])
                whisker_line.set_ydata([0.0, 0.0])
                cap_low.set_ydata([0.0, 0.0])
                cap_high.set_ydata([0.0, 0.0])
                box_text.set_text("alive=0")
                return

            speeds = np.linalg.norm(self.velocities[alive_idx], axis=1)
            vmin = float(speeds.min())
            vmax = float(speeds.max())
            q1, med, q3 = np.percentile(speeds, [25, 50, 75])

            box_rect.set_y(float(q1))
            box_rect.set_height(float(q3 - q1))
            median_line.set_ydata([float(med), float(med)])

            whisker_line.set_ydata([vmin, vmax])
            cap_low.set_ydata([vmin, vmin])
            cap_high.set_ydata([vmax, vmax])

            box_text.set_text(f"alive={len(alive_idx)}\nmax={vmax:.2f}")

        # -----------------------------
        # 右侧2：极坐标“音响波纹图”
        # -----------------------------
        ax_pol.set_title("Δv ripple (tracked)")
        ax_pol.set_theta_zero_location("E")
        ax_pol.set_theta_direction(1)
        ax_pol.set_rticks([])  # 去掉径向刻度更像“波纹”
        ax_pol.grid(True, alpha=0.35)

        theta = np.linspace(0, 2*np.pi, 360, endpoint=False)
        sigma = np.deg2rad(float(ripple_sigma_deg))

        # 三条波纹曲线（每帧更新半径 r(theta)）
        lineA, = ax_pol.plot(theta, np.zeros_like(theta), color="tab:blue", lw=2, label="alignment")
        lineC, = ax_pol.plot(theta, np.zeros_like(theta), color="tab:green", lw=2, label="cohesion")
        lineS, = ax_pol.plot(theta, np.zeros_like(theta), color="tab:orange", lw=2, label="separation")
        pol_text = ax_pol.text(0.02, 1.02, "", transform=ax_pol.transAxes, va="bottom")

        ax_pol.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))

        def _ang_mag(vec):
            mag = float(np.linalg.norm(vec))
            ang = float(np.arctan2(vec[1], vec[0]))
            if ang < 0:
                ang += 2*np.pi
            return ang, mag

        def _ripple(ang, mag):
            # 角差 wrap 到 [-pi, pi]
            d = np.arctan2(np.sin(theta - ang), np.cos(theta - ang))
            d = np.abs(d)
            return (ripple_gain * mag) * np.exp(-(d*d) / (2*sigma*sigma))

        def _update_ripples():
            # 如果追踪鸟不存在或已死：清空
            tid = self.track_id
            if tid is None or tid < 0 or tid >= self.n or (not self.alive[tid]):
                lineA.set_ydata(np.zeros_like(theta))
                lineC.set_ydata(np.zeros_like(theta))
                lineS.set_ydata(np.zeros_like(theta))
                ax_pol.set_ylim(0, 1.0)
                pol_text.set_text("tracked dead / not available")
                return

            a_ang, a_mag = _ang_mag(self._last_dv_align)
            c_ang, c_mag = _ang_mag(self._last_dv_coh)
            s_ang, s_mag = _ang_mag(self._last_dv_sep)

            rA = _ripple(a_ang, a_mag)
            rC = _ripple(c_ang, c_mag)
            rS = _ripple(s_ang, s_mag)

            lineA.set_ydata(rA)
            lineC.set_ydata(rC)
            lineS.set_ydata(rS)

            rmax = float(max(rA.max(), rC.max(), rS.max(), 1e-3))
            ax_pol.set_ylim(0, rmax * 1.15)

            pol_text.set_text(
                f"|A|={a_mag:.3f} |C|={c_mag:.3f} |S|={s_mag:.3f}\n"
                f"factor={self._last_factor:.3f} mean_dist={self._last_mean_dist:.3f}"
            )

        # -----------------------------
        # 关闭窗口自动存档（保持你原逻辑）
        # -----------------------------
        if self.autosave_path is not None:
            def _on_close(_evt):
                self.save(self.autosave_path)
                print(f"[autosave] saved to {self.autosave_path}")
            fig.canvas.mpl_connect("close_event", _on_close)

        # 同步跳帧：frames 给“模拟步上限”，stride 控制显示步长
        step_targets = range(0, int(frames) + 1, int(stride))

        def animate(step_target):
            # 推进模拟到 step_target（lifespan 与 step 同步更新）
            while self.frame < step_target and not self.all_dead():
                self.step()

            # 更新主 quiver（灰色 + alive alpha）
            colors[:, 3] = self.alive.astype(float)
            quiver.set_offsets(self.positions)
            quiver.set_UVC(self.velocities[:, 0], self.velocities[:, 1])
            quiver.set_color(colors)

            # 更新追踪鸟红点 + 标签
            tid = self.track_id
            if tid is not None and 0 <= tid < self.n and self.alive[tid]:
                track_scatter.set_offsets(self.positions[tid])
                track_label.set_text(f"tracked={tid}  step={self.frame}  alive={int(self.alive.sum())}")
                track_scatter.set_alpha(1.0)
            else:
                track_scatter.set_alpha(0.0)
                track_label.set_text(f"tracked={tid}  step={self.frame}  alive={int(self.alive.sum())}")

            # 更新三项贡献箭头（从追踪鸟位置发出）
            if tid is not None and 0 <= tid < self.n and self.alive[tid]:
                x, y = self.positions[tid]
                X = np.array([x, x, x], dtype=float)
                Y = np.array([y, y, y], dtype=float)

                U = vector_gain * np.array([self._last_dv_align[0], self._last_dv_coh[0], self._last_dv_sep[0]])
                V = vector_gain * np.array([self._last_dv_align[1], self._last_dv_coh[1], self._last_dv_sep[1]])

                contrib_quiver.set_offsets(np.column_stack([X, Y]))
                contrib_quiver.set_UVC(U, V)
                contrib_quiver.set_alpha(1.0)
            else:
                contrib_quiver.set_alpha(0.0)

            # 更新速度箱线图 + 波纹图
            _update_speed_box()
            _update_ripples()

            # blit=True：返回所有会变化的 artist
            return (
                quiver, track_scatter, track_label, contrib_quiver,
                box_rect, median_line, whisker_line, cap_low, cap_high, box_text,
                lineA, lineC, lineS, pol_text
            )

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
    sim0.run_animation(track_id=0)
    sim0.run_animation(
        frames=800,
        interval=10,
        stride=5,                 # 跳帧：每显示一帧推进5步（lifespan 同步推进）
        track_id=0,
        vector_gain=8.0,          # 主图三箭头放大倍数（纯显示用）
        ripple_gain=12.0,         # 波纹半径放大倍数（纯显示用）
        ripple_sigma_deg=20.0     # 波纹“扩散角度”
    )
    # sim0.run_animation(frames=8000, interval=1, stride=1,color_by=COLOR_BY)
    # sim0.plot_trace_rose(n_bins=72)
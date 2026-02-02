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
NUM_SPECIES_A = 120
NUM_SPECIES_B = 1
NUM_PREDATOR = 1
WIDTH, HEIGHT = 100, 100
MAX_SPEED = 10
MIN_SPEED = 0.01

NUM_SPECIES = 3

ATTR_DTYPE = np.dtype([
    # --- intra-species ---
    ("alignment_weight",        object),
    ("cohesion_weight",         object),
    ("sep_w_same",              object),
    ("sep_r_same",              object),

    # --- inter-species ---
    ("sep_w_other",             object),
    ("sep_r_other",             object),
    ("cohesion_weight_other",   object),
    ("duration",                object),
])

ALIGNMENT_WEIGHT_RANGE = (0.05, 0.05)
COHESION_WEIGHT_RANGE = (0.01, 0.01)
SEPARATION_WEIGHT_RANGE = (0.1, 0.1)
NEIGHBOR_RADIUS_RANGE = (15.0, 15.0)
SEPARATION_RADIUS_RANGE = (3.0, 5.0)

SEPARATION_WEIGHT_OTHER_RANGE = (0.10, 0.40)   # 跨物种更强一些（示例）
SEPARATION_RADIUS_OTHER_RANGE = (15.0, 15.0)     # 跨物种作用半径（示例）
COHESION_WEIGHT_OTHER_RANGE = (0.001, 0.001) # 几乎没有聚集
DURATION_RANGE = (10, 10) # 耐力，以更新数为单位

# 碰撞死亡阈值：距离 <= 1.0 撞死
COLLISION_DIST = 0.2

# 可视化：选择用哪个属性上色（五选一）
COLOR_BY = "species_id"   # 可以设置为 'alignment_weight', 'cohesion_weight', 'separation_weight', 'neighbor_radius', 'separation_radius'


def limit_speed(v, max_speed):
    speed = np.linalg.norm(v)
    if speed > max_speed and speed > 1e-12:
        return v / speed * max_speed
    return v

def initialize_species_and_attributes(
    rng,
    p,          # 第一类对象数量
    p_n,        # 第二类对象数量
    p_species_A,  # shape = (NUM_SPECIES_A,)
    p_species_B,  # shape = (NUM_SPECIES_B,)
):
    """
    返回：
        species_id : (p+p_n,) int32
        attrs      : structured array[alignment_weight,
                                      cohesion_weight, 
                                      sep_w_same,
                                      sep_r_same,
                                      sep_w_other,
                                      sep_r_other,
                                      
                                      cohesion_weight_other,
                                      duration,
                                      ], length p+p_n
    """
    total = p + p_n

    species_id = np.empty(total, dtype=np.int32)

    # -----------------------------
    # 第一类对象：species A
    # -----------------------------
    species_id[:p] = rng.choice(
        NUM_SPECIES_A,
        size=p,
    ).astype(np.int32)

    # -----------------------------
    # 第二类对象：species B（整体偏移）
    # -----------------------------
    species_id[p:] = (
        rng.choice(
            NUM_SPECIES_B,
            size=p_n,
        ) + NUM_SPECIES_A
    ).astype(np.int32)

    # -----------------------------
    # 2) 初始化结构化 attrs
    # -----------------------------
    attrs = np.empty(total, dtype=ATTR_DTYPE)

    # 先全部设为 None（非常重要）
    for name in attrs.dtype.names:
        attrs[name] = None

    # -----------------------------
    # 3) 第一类对象（0 ~ p-1）
    #    从参数空间 A 采样
    # -----------------------------
    idx_A = slice(0, p)

    attrs["alignment_weight"][idx_A] = rng.uniform(
        ALIGNMENT_WEIGHT_RANGE[0],
        ALIGNMENT_WEIGHT_RANGE[1],
        p
    )

    attrs["cohesion_weight"][idx_A] = rng.uniform(
        COHESION_WEIGHT_RANGE[0],
        COHESION_WEIGHT_RANGE[1],
        p
    )

    attrs["sep_w_same"][idx_A] = rng.uniform(
        SEPARATION_WEIGHT_RANGE[0],
        SEPARATION_WEIGHT_RANGE[1],
        p
    )

    attrs["sep_r_same"][idx_A] = rng.uniform(
        SEPARATION_RADIUS_RANGE[0],
        SEPARATION_RADIUS_RANGE[1],
        p
    )

    # inter-species（若该类对象不具备，保持 None）
    attrs["sep_w_other"][idx_A] = rng.uniform(
        SEPARATION_WEIGHT_OTHER_RANGE[0],
        SEPARATION_WEIGHT_OTHER_RANGE[1],
        p
    )

    attrs["sep_r_other"][idx_A] = rng.uniform(
        SEPARATION_RADIUS_OTHER_RANGE[0],
        SEPARATION_RADIUS_OTHER_RANGE[1],
        p
    )

    # -----------------------------
    # 4) 第二类对象（p ~ p+p_n-1）
    #    从参数空间 B 采样
    # -----------------------------
    idx_B = slice(p, total)

    # intra-species：可能没有
    # -> 保持 None

    # inter-species 特有属性
    attrs["cohesion_weight_other"][idx_B] = rng.uniform(
        COHESION_WEIGHT_OTHER_RANGE[0],
        COHESION_WEIGHT_OTHER_RANGE[1],
        p_n
    )

    attrs["duration"][idx_B] = rng.uniform(
        DURATION_RANGE[0],
        DURATION_RANGE[1],
        p_n
    )

    # attrs["sep_w_other"][idx_B] = rng.uniform(
    #     SEPARATION_WEIGHT_OTHER_RANGE[0],
    #     SEPARATION_WEIGHT_OTHER_RANGE[1],
    #     p_n
    # )

    # attrs["sep_r_other"][idx_B] = rng.uniform(
    #     SEPARATION_RADIUS_OTHER_RANGE[0],
    #     SEPARATION_RADIUS_OTHER_RANGE[1],
    #     p_n
    # )

    return species_id, attrs


class BoidsSim:
    def __init__(self, n=NUM_SPECIES_A, n_p=NUM_SPECIES_B, seed=None,
                 init_from=None,
                 autosave_path=None,
                 autosave_meta=None):
        """
        initiate a new round of simulation, including boids_initiation, updating, animation.

        All birds are stored in one templet.

        init_from:
          - None: 随机初始化 positions/velocities/attrs
          - dict: 来自 load_population() 的字典，直接用其中的状态
        autosave_path:
          - 关闭窗口时自动存档（npz）
        """
        self.n = n # specie A
        self.predator_n = n_p # predator B
        self.rng = np.random.default_rng(seed)

        self.species_id, self.attrs = initialize_species_and_attributes(self.rng, self.n, self.predator_n, self.n, self.predator_n)
        self.positions = self.rng.random((n, 2)) * np.array([WIDTH, HEIGHT], dtype=float)
        self.velocities = (self.rng.random((n, 2)) - 0.5) * MAX_SPEED
        self.alive = np.ones(n, dtype=bool)
        self.lifespan = np.zeros(n, dtype=np.int32)
        self.meta = {"generation": 0}
        

        self.frame = 0
        self.autosave_path = autosave_path
        self.autosave_meta = autosave_meta if autosave_meta is not None else {}

    def step(self):
        """单步更新：species-aware 的 Boids + 捕食接口"""

        new_vel = self.velocities.copy()

        alive_idx = np.flatnonzero(self.alive)
        if alive_idx.size == 0:
            self.frame += 1
            return

        for i in alive_idx:
            pos_i = self.positions[i]
            vel_i = self.velocities[i]
            si = self.species_id[i]

            attr = self.attrs[i]

            # -----------------------------
            # 初始化力项
            # -----------------------------
            align = np.zeros(2)
            coh_same = np.zeros(2)
            sep_same = np.zeros(2)

            coh_other = np.zeros(2)
            sep_other = np.zeros(2)

            cnt_same = 0
            cnt_other = 0

            # -----------------------------
            # 扫描邻居
            # -----------------------------
            for j in alive_idx:
                if i == j:
                    continue

                offset = self.positions[j] - pos_i
                dist = np.linalg.norm(offset)
                if dist < 1e-12:
                    continue

                sj = self.species_id[j]

                # ===== 同类 =====
                if sj == si:
                    if attr["sep_r_same"] is not None and dist <= attr["sep_r_same"]:
                        sep_same -= offset / dist

                    # 对齐 & 聚集没有硬半径（可后续加）
                    align += self.velocities[j]
                    coh_same += self.positions[j]
                    cnt_same += 1

                # ===== 异类 =====
                else:
                    # 第一类：只做分离
                    if si < NUM_SPECIES_A:
                        if attr["sep_r_other"] is not None and dist <= attr["sep_r_other"]:
                            sep_other -= offset / dist
                            cnt_other += 1

                    # 第二类（捕食者）
                    else:
                        # 聚集到猎物（捕食接口）
                        coh_other += self.positions[j]
                        cnt_other += 1

            # -----------------------------
            # 汇总同类项
            # -----------------------------
            if cnt_same > 0:
                align = align / cnt_same - vel_i
                coh_same = coh_same / cnt_same - pos_i

            if cnt_other > 0:
                coh_other = coh_other / cnt_other - pos_i
                sep_other = sep_other / cnt_other

            # -----------------------------
            # 速度更新（按 species 分支）
            # -----------------------------
            if si < NUM_SPECIES_A:
                # ===== 第一类对象 =====
                vel_i = (
                    vel_i
                    + (attr["alignment_weight"] or 0.0) * align
                    + (attr["cohesion_weight"] or 0.0) * coh_same
                    + (attr["sep_w_same"] or 0.0) * sep_same
                    + (attr["sep_w_other"] or 0.0) * sep_other
                )

            else:
                # ===== 第二类对象（捕食者）=====
                stamina = attr["duration"] or 0.0

                vel_i = (
                    vel_i
                    + (attr["alignment_weight"] or 0.0) * align
                    + (attr["cohesion_weight"] or 0.0) * coh_same
                    + (attr["sep_w_same"] or 0.0) * sep_same
                    + (attr["cohesion_weight_other"] or 0.0) * coh_other * stamina
                )

            vel_i = limit_speed(vel_i, MAX_SPEED)
            new_vel[i] = vel_i

        # -----------------------------
        # 位置更新
        # -----------------------------
        self.positions[self.alive] = (
            self.positions[self.alive] + new_vel[self.alive]
        ) % np.array([WIDTH, HEIGHT])

        self.velocities[self.alive] = new_vel[self.alive]

        # -----------------------------
        # lifespan
        # -----------------------------
        self.lifespan[self.alive] += 1

        # 碰撞 / 捕食逻辑接口（暂不启用）
        # self._apply_collisions_or_predation()

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
        if color_by == "species_id":
            # 离散类别：species_id
            v = self.species_id.astype(int)

            # 物种数：若你有 NUM_SPECIES 可直接用；否则从数据推断
            n_species = int(v.max()) + 1

            # 离散色板（推荐 tab10 / Set2 这类分类色板）
            cmap = plt.get_cmap("tab10", n_species)  # 固定 n_species 个颜色

            # 边界归一化：每个整数落在自己的 bin 里
            bounds = np.arange(-0.5, n_species + 0.5, 1.0)
            norm = mpl.colors.BoundaryNorm(bounds, ncolors=n_species)

        else:
            col_idx = ATTR_NAMES.index(color_by)
            v = self.attrs[:, col_idx].astype(float)
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
        n=NUM_SPECIES_A,
        n_p=NUM_SPECIES_B,
        seed=42,
        init_from=None,
        autosave_path="gen0.npz",
        autosave_meta={"generation": 0}
    )
    sim0.run_animation(frames=8000, interval=1, stride=1,color_by=COLOR_BY)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib import cm

# -----------------------------
# Global settings
# -----------------------------
NUM_BOIDS = 120
WIDTH, HEIGHT = 100, 100
MAX_SPEED = 10

# 进化/初始化的参数范围（可按需调整）
ATTR_NAMES = [
    "alignment_weight",
    "cohesion_weight",
    "separation_weight",
    "neighbor_radius",
    "separation_radius",
]
ATTR_RANGES = {
    "alignment_weight": (0.01, 0.10),
    "cohesion_weight": (0.005, 0.05),
    "separation_weight": (0.1, 0.20),
    "neighbor_radius": (10.0, 15.0),
    "separation_radius": (3.0, 5.0),
}

# 碰撞死亡阈值：距离 <= 1.0 撞死
COLLISION_DIST = 0.2

# 可视化：选择用哪个属性上色（五选一）
COLOR_BY = "separation_radius"   # 可以设置为 'alignment_weight', 'cohesion_weight', 'separation_weight', 'neighbor_radius', 'separation_radius'


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


def save_population(path, positions, velocities, attrs, alive, lifespan, meta=None):
    """存档：npz 内含关键数组。"""
    meta = {} if meta is None else meta
    np.savez(
        path,
        positions=positions,
        velocities=velocities,
        attrs=attrs,
        alive=alive.astype(np.int8),
        lifespan=lifespan.astype(np.int32),
        meta=np.array([meta], dtype=object),
    )


def load_population(path):
    """读档：返回 dict。"""
    data = np.load(path, allow_pickle=True)
    meta = data["meta"][0] if "meta" in data else {}
    return {
        "positions": data["positions"].copy(),
        "velocities": data["velocities"].copy(),
        "attrs": data["attrs"].copy(),
        "alive": data["alive"].astype(bool).copy(),
        "lifespan": data["lifespan"].copy(),
        "meta": meta,
    }


def evolve_attrs(prev_attrs, fitness, n_out, rng,
                elite_frac=0.10, mutation_rate=0.25, sigma_frac=0.10):
    """
    由上一代 attrs 生成下一代 attrs。
    - fitness: (N,) 数值越大越好（这里可用 lifespan 或 alive + lifespan）
    - 精英保留 elite_frac
    - 其余通过加权抽样 + 简单“均值交叉” + 高斯变异
    """
    fitness = fitness.astype(float)
    fitness = np.maximum(fitness, 0.0)

    # 防止全 0：给一个微小基线
    probs = fitness + 1e-9
    probs = probs / probs.sum()

    n_elite = max(1, int(round(elite_frac * n_out)))
    elite_idx = np.argsort(fitness)[-n_elite:]
    elite = prev_attrs[elite_idx]

    children = []
    while len(children) < (n_out - n_elite):
        p1 = prev_attrs[rng.choice(len(prev_attrs), p=probs)]
        p2 = prev_attrs[rng.choice(len(prev_attrs), p=probs)]
        child = 0.5 * (p1 + p2)

        # mutation: 以一定概率对每个基因加噪声
        for k, name in enumerate(ATTR_NAMES):
            if rng.random() < mutation_rate:
                lo, hi = ATTR_RANGES[name]
                sigma = sigma_frac * (hi - lo)
                child[k] += rng.normal(0.0, sigma)
                child[k] = np.clip(child[k], lo, hi)

        children.append(child)

    new_attrs = np.vstack([elite, np.array(children)])
    rng.shuffle(new_attrs)
    return new_attrs


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
        self._apply_collisions()

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

    def save(self, path):
        meta = dict(self.meta)
        meta.update(self.autosave_meta)
        meta["last_frame"] = self.frame
        meta["alive_count"] = int(self.alive.sum())
        save_population(path, self.positions, self.velocities, self.attrs, self.alive, self.lifespan, meta=meta)

    def run_animation(self, frames=600, interval=20, color_by=COLOR_BY):
        # 颜色映射基于 attrs 的某一列；死鸟 alpha=0
        col_idx = ATTR_NAMES.index(color_by)
        v = self.attrs[:, col_idx]
        norm = Normalize(vmin=float(v.min()), vmax=float(v.max()))
        cmap = cm.viridis

        fig, ax = plt.subplots()
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.set_title(f"Boids (color_by={color_by}) | Collision<= {COLLISION_DIST} => death")

        colors = cmap(norm(v))
        colors[:, 3] = self.alive.astype(float)  # alpha: alive=1, dead=0

        quiver = ax.quiver(
            self.positions[:, 0], self.positions[:, 1],
            self.velocities[:, 0], self.velocities[:, 1],
            color=colors, angles='xy', scale_units='xy', scale=1
        )

        # 可选：窗口关闭自动存档
        if self.autosave_path is not None:
            def _on_close(_evt):
                self.save(self.autosave_path)
                print(f"[autosave] saved to {self.autosave_path}")
            fig.canvas.mpl_connect("close_event", _on_close)

        def animate(_frame):
            self.step()

            # 更新位置、速度
            quiver.set_offsets(self.positions)
            quiver.set_UVC(self.velocities[:, 0], self.velocities[:, 1])

            # 更新颜色（alpha 反映 alive）
            colors = cmap(norm(v))
            colors[:, 3] = self.alive.astype(float)
            quiver.set_color(colors)

            return (quiver,)

        ani = FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True)
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
    sim0.run_animation(frames=800, interval=5, color_by=COLOR_BY)

    # ====== 2) 基于 gen0.npz 进化生成 gen1_attrs，并跑第1代 ======
    # 说明：下面代码在你关闭第0代窗口后才会继续执行
    gen0 = load_population("gen0.npz")
    prev_attrs = gen0["attrs"]
    # fitness：这里用 lifespan（活得更久更好）；你也可以加 alive 权重
    fitness = gen0["lifespan"].astype(float) + 50.0 * gen0["alive"].astype(float)

    rng = np.random.default_rng(123)
    gen1_attrs = evolve_attrs(prev_attrs, fitness, n_out=NUM_BOIDS, rng=rng)

    # 第1代的初始位置/速度：通常重新随机更合理（避免继承“最终团块”）
    init1 = {
        "positions": rng.random((NUM_BOIDS, 2)) * np.array([WIDTH, HEIGHT], float),
        "velocities": (rng.random((NUM_BOIDS, 2)) - 0.5) * MAX_SPEED,
        "attrs": gen1_attrs,
        "alive": np.ones(NUM_BOIDS, dtype=bool),
        "lifespan": np.zeros(NUM_BOIDS, dtype=np.int32),
        "meta": {"generation": 1, "parent_file": "gen0.npz"},
    }

    sim1 = BoidsSim(
        n=NUM_BOIDS,
        seed=7,
        init_from=init1,
        autosave_path="gen1.npz",
        autosave_meta={"generation": 1}
    )
    sim1.run_animation(frames=800, interval=5, color_by=COLOR_BY)

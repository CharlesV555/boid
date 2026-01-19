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
MAX_SPEED = 30

# 撞死阈值：距离 <= 1.0
COLLISION_DIST = 1.0

# 属性定义：每只鸟 attrs[i] = [aw, cw, sw, nr, sr]
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
    "separation_weight": (0.05, 0.20),
    "neighbor_radius": (5.0, 10.0),
    "separation_radius": (2.0, 5.0),
}

# 渲染用：选择哪个属性着色
COLOR_BY = "neighbor_radius"


# -----------------------------
# Persistence
# -----------------------------
def save_population(path, positions, velocities, attrs, alive, lifespan, meta=None):
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


# -----------------------------
# Helpers
# -----------------------------
def limit_speed(v, max_speed):
    speed = np.linalg.norm(v)
    if speed > max_speed and speed > 1e-12:
        return v / speed * max_speed
    return v


def random_attrs(n, rng):
    cols = []
    for name in ATTR_NAMES:
        lo, hi = ATTR_RANGES[name]
        cols.append(rng.uniform(lo, hi, size=n))
    return np.column_stack(cols)


def evolve_attrs(prev_attrs, fitness, n_out, rng,
                elite_frac=0.10, mutation_rate=0.25, sigma_frac=0.10):
    """
    由上一轮 attrs 生成下一轮 attrs（简单 GA）：
    - elite 保留
    - 其余：按 fitness 抽样两亲本，均值交叉 + 变异
    """
    fitness = np.maximum(fitness.astype(float), 0.0)
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


# -----------------------------
# Core simulator
# -----------------------------
class BoidsSim:
    def __init__(self, n=NUM_BOIDS, seed=None, init_from=None,
                 autosave_path=None, autosave_meta=None):
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

        self.frame = 0  # 已完成的 step 数
        self.closed_by_user = False

        self.autosave_path = autosave_path
        self.autosave_meta = autosave_meta if autosave_meta is not None else {}

    def alive_count(self):
        return int(self.alive.sum())

    def all_dead(self):
        return self.alive_count() == 0

    def step(self):
        """执行 1 个模拟步：更新速度/位置 + 撞死检测。"""
        new_vel = self.velocities.copy()

        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) == 0:
            self.frame += 1
            return

        for i in alive_idx:
            pos_i = self.positions[i]
            vel_i = self.velocities[i]
            aw, cw, sw, nr, sr = self.attrs[i]

            alignment = np.zeros(2, dtype=float)
            cohesion = np.zeros(2, dtype=float)
            separation = np.zeros(2, dtype=float)
            count = 0

            for j in alive_idx:
                if i == j:
                    continue
                offset = self.positions[j] - pos_i
                dist = np.linalg.norm(offset)

                if dist < nr:
                    alignment += self.velocities[j]
                    cohesion += self.positions[j]
                    if dist < sr and dist > 1e-12:
                        separation -= offset / dist
                    count += 1

            if count > 0:
                alignment = alignment / count - vel_i
                cohesion = cohesion / count - pos_i

            vel_i = vel_i + aw * alignment + cw * cohesion + sw * separation
            vel_i = limit_speed(vel_i, MAX_SPEED)
            new_vel[i] = vel_i

        # 位置更新（死鸟不动）
        self.positions[self.alive] = (self.positions[self.alive] + new_vel[self.alive]) % np.array([WIDTH, HEIGHT])
        self.velocities[self.alive] = new_vel[self.alive]

        # 存活帧数累加
        self.lifespan[self.alive] += 1

        # 撞死
        self._apply_collisions()

        self.frame += 1

    def _apply_collisions(self):
        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) < 2:
            return

        d2_thr = COLLISION_DIST ** 2
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
                dx, dy = pj - pi
                if (dx * dx + dy * dy) <= d2_thr:
                    self.alive[i] = False
                    self.alive[j] = False
                    self.velocities[i] = 0.0
                    self.velocities[j] = 0.0

    def save(self, path):
        meta = dict(self.meta)
        meta.update(self.autosave_meta)
        meta["last_frame"] = int(self.frame)
        meta["alive_count"] = int(self.alive.sum())
        save_population(path, self.positions, self.velocities, self.attrs, self.alive, self.lifespan, meta=meta)

    # --------- run modes ---------
    def run_headless(self, max_steps):
        """
        不渲染：跑到 max_steps 或 全死。
        返回：reason, summary dict
        """
        while self.frame < max_steps and not self.all_dead():
            self.step()

        reason = "all_dead" if self.all_dead() else "max_steps"
        summary = {
            "reason": reason,
            "steps": int(self.frame),
            "alive_count": int(self.alive.sum()),
            "max_lifespan": int(self.lifespan.max()) if len(self.lifespan) else 0,
            "mean_lifespan": float(self.lifespan.mean()) if len(self.lifespan) else 0.0,
        }
        return reason, summary

    def run_animation(self, max_steps, interval_ms, color_by=COLOR_BY, display_mask=None):
        """
        渲染：窗口关闭 或 全死结束（全死会自动 close）。
        display_mask: bool array length=max_steps。True 的 step 才显示；False 的 step 只模拟不显示。
        返回：reason, summary dict
        """
        col_idx = ATTR_NAMES.index(color_by)
        v = self.attrs[:, col_idx]
        norm = Normalize(vmin=float(v.min()), vmax=float(v.max()))
        cmap = cm.viridis

        if display_mask is None:
            display_steps = np.arange(max_steps, dtype=int)
        else:
            display_mask = np.asarray(display_mask, dtype=bool)
            if display_mask.shape[0] != max_steps:
                raise ValueError("display_mask length must equal max_steps")
            display_steps = np.where(display_mask)[0].astype(int)
            if len(display_steps) == 0:
                # 一个都不显示=等价 headless
                return self.run_headless(max_steps=max_steps)

        fig, ax = plt.subplots()
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.set_title(f"Boids | color_by={color_by} | collision<={COLLISION_DIST}")

        colors = cmap(norm(v))
        colors[:, 3] = self.alive.astype(float)  # alpha alive/dead

        quiver = ax.quiver(
            self.positions[:, 0], self.positions[:, 1],
            self.velocities[:, 0], self.velocities[:, 1],
            color=colors, angles="xy", scale_units="xy", scale=1
        )

        # close event: 用户关窗
        def _on_close(_evt):
            self.closed_by_user = True
            if self.autosave_path is not None:
                self.save(self.autosave_path)

        fig.canvas.mpl_connect("close_event", _on_close)

        # 让“显示帧”对应到“目标 step”，并在两次显示之间补跑中间 step
        def animate(k):
            if self.closed_by_user:
                return (quiver,)

            target_step = int(display_steps[k])
            # 我们希望显示“完成 target_step+1 次 step 后”的状态（即运行到 frame==target_step+1）
            while self.frame <= target_step and self.frame < max_steps and not self.all_dead():
                self.step()

            quiver.set_offsets(self.positions)
            quiver.set_UVC(self.velocities[:, 0], self.velocities[:, 1])

            colors = cmap(norm(v))
            colors[:, 3] = self.alive.astype(float)
            quiver.set_color(colors)

            # 全死：自动关窗结束
            if self.all_dead():
                plt.close(fig)

            # 已经跑到 max_steps：自动关窗结束
            if self.frame >= max_steps:
                plt.close(fig)

            return (quiver,)

        ani = FuncAnimation(fig, animate, frames=len(display_steps), interval=interval_ms, blit=True)
        plt.show()

        # 退出 show 后给出原因
        if self.all_dead():
            reason = "all_dead"
        elif self.closed_by_user:
            reason = "window_closed"
        else:
            reason = "max_steps"

        summary = {
            "reason": reason,
            "steps": int(self.frame),
            "alive_count": int(self.alive.sum()),
            "max_lifespan": int(self.lifespan.max()) if len(self.lifespan) else 0,
            "mean_lifespan": float(self.lifespan.mean()) if len(self.lifespan) else 0.0,
        }
        return reason, summary


# -----------------------------
# Experiment loop (multi-generation)
# -----------------------------
def run_generations(
    num_generations,
    max_steps=800,
    interval_ms=5,
    color_by=COLOR_BY,
    render=False,
    render_mask=None,
    display_masks=None,
    seed_init=42,
    seed_evolve=123,
    save_template="gen{g}.npz",
):
    """
    render:
      - bool：全部渲染 or 全部不渲染
    render_mask:
      - 可选，bool array length=num_generations：逐轮控制是否渲染（优先于 render）
    display_masks:
      - None：渲染时显示全部步
      - bool array length=max_steps：所有渲染轮使用同一个“显示步数 mask”
      - list[bool array] length=num_generations：每轮一个 mask
    """
    rng_init = np.random.default_rng(seed_init)
    rng_evolve = np.random.default_rng(seed_evolve)

    prev_attrs = None
    prev_fitness = None

    for g in range(num_generations):
        # 决定本轮是否渲染
        if render_mask is not None:
            do_render = bool(np.asarray(render_mask, dtype=bool)[g])
        else:
            do_render = bool(render)

        # 本轮初始化
        if g == 0:
            attrs = random_attrs(NUM_BOIDS, rng_init)
        else:
            # 进化产生本轮 attrs
            attrs = evolve_attrs(prev_attrs, prev_fitness, n_out=NUM_BOIDS, rng=rng_evolve)

        init_state = {
            "positions": rng_init.random((NUM_BOIDS, 2)) * np.array([WIDTH, HEIGHT], float),
            "velocities": (rng_init.random((NUM_BOIDS, 2)) - 0.5) * MAX_SPEED,
            "attrs": attrs,
            "alive": np.ones(NUM_BOIDS, dtype=bool),
            "lifespan": np.zeros(NUM_BOIDS, dtype=np.int32),
            "meta": {"generation": g},
        }

        save_path = save_template.format(g=g)
        sim = BoidsSim(
            n=NUM_BOIDS,
            seed=int(rng_init.integers(0, 1_000_000_000)),
            init_from=init_state,
            autosave_path=save_path,
            autosave_meta={"generation": g},
        )

        # 取本轮 display_mask
        display_mask = None
        if display_masks is not None:
            if isinstance(display_masks, list):
                display_mask = display_masks[g]
            else:
                display_mask = display_masks  # 单个 mask 复用

        # 运行
        if do_render:
            reason, summary = sim.run_animation(
                max_steps=max_steps,
                interval_ms=interval_ms,
                color_by=color_by,
                display_mask=display_mask
            )
            # 若用户提前关窗，也算本轮结束
        else:
            reason, summary = sim.run_headless(max_steps=max_steps)

        # 确保存档（渲染模式下用户关窗会自动存，但这里再存一次也可接受）
        sim.save(save_path)

        # 输出每轮总结（不渲染要求的 max_lifespan / 全死结束在 reason 里）
        print(f"[gen {g}] reason={summary['reason']} steps={summary['steps']} "
              f"alive={summary['alive_count']} max_lifespan={summary['max_lifespan']:.0f}")

        # 为下一轮准备 fitness
        # 典型 fitness：lifespan + alive_bonus（如果没全死，鼓励存活）
        prev_attrs = sim.attrs.copy()
        prev_fitness = sim.lifespan.astype(float) + 50.0 * sim.alive.astype(float)

        # 若你希望“全死就停止整个实验”，可取消注释：
        # if sim.all_dead():
        #     print(f"All dead at generation {g}, stop experiment.")
        #     break


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # 例1：全部不渲染，跑 5 轮；每轮到 max_steps 或全死结束
    run_generations(
        num_generations=5,
        max_steps=800,
        render=False,
        save_template="gen{g}.npz"
    )

    # 例2：只渲染第0、2轮（render_mask 控制），并且渲染时只显示指定步数
    # 只显示：0~199 每 5 步显示一次
    mask = np.zeros(800, dtype=bool)
    mask[::5] = True

    run_generations(
        num_generations=3,
        max_steps=800,
        interval_ms=5,
        color_by="neighbor_radius",
        render_mask=[True, False, True],
        display_masks=mask,              # 所有渲染轮复用同一个显示 mask
        save_template="gen{g}.npz"
    )

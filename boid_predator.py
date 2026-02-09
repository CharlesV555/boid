import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib as mpl
import cmocean

import viz # visualization

# -----------------------------
# Global settings
# -----------------------------
NUM_SPECIES_A = 60
NUM_SPECIES_B = 1
NUM_PREDATOR = 1
WIDTH, HEIGHT = 100, 100
MAX_SPEED = 5
MAX_SPEED_predate = 6
MAX_SPEED_rest = 3
MIN_SPEED = 0.01

NUM_SPECIES = 2

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

# 注意，separation_radius_other>=separation_radius_same, 我认为先判断远近再识别物种，以捕食者的疏远范围为上界。
ALIGNMENT_WEIGHT_RANGE = (0.05, 0.05)
COHESION_WEIGHT_RANGE = (0.01, 0.015)
SEPARATION_WEIGHT_RANGE = (0.1, 0.12)
NEIGHBOR_RADIUS_RANGE = (13.0, 15.0)
SEPARATION_RADIUS_RANGE = (3.0, 5.0)

SEPARATION_WEIGHT_OTHER_RANGE = (0.5, 0.5)   # 跨物种更强一些（示例）
SEPARATION_RADIUS_OTHER_RANGE = (30.0, 30.0)     # 跨物种作用半径（示例）
COHESION_WEIGHT_OTHER_RANGE = (0.1, 0.1) # 捕食行为的靠近
DURATION_RANGE = (25.0, 27.0) # 耐力，以更新数为单位

# 碰撞死亡阈值：距离 <= 1.0 撞死
COLLISION_DIST = 3.0

# 可视化：选择用哪个属性上色（六选一）
COLOR_BY = "species_id"   # 可以设置为 'alignment_weight', 'cohesion_weight', 'separation_weight', 'neighbor_radius', 'separation_radius', role_stamina

# viz参数传递
viz.WIDTH = WIDTH
viz.HEIGHT = HEIGHT
viz.MAX_SPEED = MAX_SPEED
viz.NUM_SPECIES_A = NUM_SPECIES_A
viz.COLLISION_DIST = COLLISION_DIST


def limit_speed(v, max_speed):
    """
    limit speed in a reasonable range.
    """
    speed = np.linalg.norm(v)
    if speed > max_speed and speed > 1e-12:
        return v / speed * max_speed
    return v

def initialize_species_and_attributes(
    rng,
    p,          # 第一类对象数量
    p_n,        # 第二类对象数量
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

    # only 1 predator
    species_id = np.empty(total, dtype=np.int32)
    species_id[:p] = 0
    species_id[p:] = 1

    # initial attrs
    attrs = np.empty(total, dtype=ATTR_DTYPE)
    for name in attrs.dtype.names:
        attrs[name] = None

    # for boid, sampling from space A
    idx_A = slice(0, p)

    attrs["alignment_weight"][idx_A] = rng.uniform(ALIGNMENT_WEIGHT_RANGE[0],ALIGNMENT_WEIGHT_RANGE[1],p)
    attrs["cohesion_weight"][idx_A] = rng.uniform(COHESION_WEIGHT_RANGE[0],COHESION_WEIGHT_RANGE[1],p)
    attrs["sep_w_same"][idx_A] = rng.uniform(SEPARATION_WEIGHT_RANGE[0],SEPARATION_WEIGHT_RANGE[1],p)
    attrs["sep_r_same"][idx_A] = rng.uniform(SEPARATION_RADIUS_RANGE[0],SEPARATION_RADIUS_RANGE[1],p)

    # inter-species（若该类对象不具备，保持 None）
    attrs["sep_w_other"][idx_A] = rng.uniform(SEPARATION_WEIGHT_OTHER_RANGE[0],SEPARATION_WEIGHT_OTHER_RANGE[1],p)
    attrs["sep_r_other"][idx_A] = rng.uniform(SEPARATION_RADIUS_OTHER_RANGE[0],SEPARATION_RADIUS_OTHER_RANGE[1],p)

    # for predator, sampling from space B
    idx_B = slice(p, total)

    # intra-species：可能没有
    # -> 保持 None
    attrs["alignment_weight"][idx_B] = rng.uniform(ALIGNMENT_WEIGHT_RANGE[0],ALIGNMENT_WEIGHT_RANGE[1],p_n)
    attrs["cohesion_weight"][idx_B] = rng.uniform(COHESION_WEIGHT_RANGE[0],COHESION_WEIGHT_RANGE[1],p_n)
    attrs["sep_w_other"][idx_B] = rng.uniform(SEPARATION_WEIGHT_OTHER_RANGE[0],SEPARATION_WEIGHT_OTHER_RANGE[1],p_n)
    attrs["sep_r_other"][idx_B] = rng.uniform(SEPARATION_RADIUS_OTHER_RANGE[0],SEPARATION_RADIUS_OTHER_RANGE[1],p_n)

    # inter-species 特有属性
    attrs["cohesion_weight_other"][idx_B] = rng.uniform(COHESION_WEIGHT_OTHER_RANGE[0],COHESION_WEIGHT_OTHER_RANGE[1],p_n)
    attrs["duration"][idx_B] = rng.uniform(DURATION_RANGE[0],DURATION_RANGE[1],p_n)

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
        self.total = self.n + self.predator_n
        self.rng = np.random.default_rng(seed)

        # internal properties
        self.species_id, self.attrs = initialize_species_and_attributes(self.rng, self.n, self.predator_n)
        self.stamina = np.zeros(self.total, dtype=bool)   # predator 初始可捕食
        self.locked  = np.full(self.total, -1, dtype=np.int32)

        # active properties
        self.positions = self.rng.random((self.total, 2)) * np.array([WIDTH, HEIGHT], dtype=float)
        self.velocities = (self.rng.random((self.total, 2)) - 0.5) * MAX_SPEED
        self.alive = np.ones(self.total, dtype=bool)
        self.lifespan = np.zeros(self.total, dtype=np.int32)
        self.duration = self.attrs["duration"].copy() # 这里比较特殊，duration通过在attrs里面赋值，又参与到变化中，故出现两次
        # game properties
        self.meta = {"generation": 0}
        self.frame = 0 # step counter
        
        self.autosave_path = autosave_path
        self.autosave_meta = autosave_meta if autosave_meta is not None else {}

    def step(self):
        """单步更新：species-aware 的 Boids"""
        
        new_vel = self.velocities.copy()

        alive_idx = np.flatnonzero(self.alive)
        if alive_idx.size == 0:
            self.frame += 1
            return
        # for a certain bird i
        for i in alive_idx:
            pos_i = self.positions[i]
            vel_i = self.velocities[i]
            si = self.species_id[i]

            attr = self.attrs[i]
            locked = None # predator focus on a certain prey
            # initial records on velocity rates with different properties
            align = np.zeros(2)
            coh_same = np.zeros(2)
            sep_same = np.zeros(2)

            coh_other = np.zeros(2)
            sep_other = np.zeros(2)
            
            
            # count in order to calculate mean
            cnt_same = 0
            cnt_other = 0

                
            # def predation():
            #     """捕食关系：i捕食j
            #     当i锁定j时，在duration耗尽前都会跟踪J;
            #     当i与J距离小于碰撞距离时捕食成功，duration重置为0，与猎物脱钩；
            #     duration耗尽时捕食失败，与猎物脱钩。
                
            # for bird
            if si == 0:
                for j in alive_idx: 
                    # 跳过自己
                    if i == j:
                        continue
                    # 原来判断撞死的逻辑，未实现
                    offset = self.positions[j] - pos_i
                    dist = np.linalg.norm(offset) # 计算距离
                    if dist < 1e-12: 
                        continue
                    if attr["sep_r_other"] == None:
                        continue
                    if dist> attr["sep_r_other"]: # 超出范围不予理睬
                        continue
                    else:
                        # 识别是否同类
                        sj = self.species_id[j]
                        same = (sj == si)
                        
                        # ===== 同类 =====
                        if sj == si:
                            # 同类，判断是否在聚合与疏远范围内
                            if attr["sep_r_same"] is not None and dist <= attr["sep_r_same"]:
                                sep_same -= offset / dist

                            # 对齐 & 聚集没有硬半径（可后续加）
                            align += self.velocities[j]
                            coh_same += self.positions[j]
                            cnt_same += 1 # 计数

                        # ===== 异类：捕食者 =====
                        else:
                            if attr["sep_r_same"] is not None and dist <= attr["sep_r_same"]:
                                threat =self.calculate_threat(dist)
                                sep_other -= offset / threat
                                cnt_other += 1
                # -----------------------------
                # 扫描完所有鸟后更新数据
                # -----------------------------
                # 同时更新duration
                if cnt_same > 0:
                    align = align / cnt_same - vel_i
                    coh_same = coh_same / cnt_same - pos_i

                if cnt_other > 0:
                    coh_other = coh_other / cnt_other - pos_i
                    sep_other = sep_other / cnt_other

                # ===== 鸟群 =====
                vel_i = (
                    vel_i
                    + (attr["alignment_weight"] or 0.0) * align
                    + (attr["cohesion_weight"] or 0.0) * coh_same
                    + (attr["sep_w_same"] or 0.0) * sep_same
                    + (attr["sep_w_other"] or 0.0) * sep_other
                )

                vel_i = limit_speed(vel_i, MAX_SPEED)
                new_vel[i] = vel_i
            # for predator
            elif si == 1:
                _max_vel = 0
                record_prey_id = -1 # 仅在每轮扫描前重置
                def record_prey(_max_vel,prey):
                    "a simple compare function, to find prey with minimum distance"
                    # 计算相对速度最小的
                    v_offset = self.velocities[prey] - vel_i
                    _temp_vel = np.linalg.norm(v_offset)
                    if _temp_vel > _max_vel:
                        _max_vel = _temp_vel
                    
                    return _max_vel
                # -----------------------------
                # 扫描邻居
                # -----------------------------
                for j in alive_idx: 
                    # 跳过自己
                    if i == j:
                        continue
                    # 原来判断撞死的逻辑，未实现
                    offset = self.positions[j] - pos_i
                    dist = np.linalg.norm(offset) # 计算距离
                    if dist < 1e-12: 
                        continue
                    if attr["sep_r_other"] is None:
                        continue
                    if dist > attr["sep_r_other"]: # 超出范围不予理睬
                        continue
                    else:
                        # 识别是否同类
                        sj = self.species_id[j]
                        same = (sj == si)
                        
                        # ===== 同类 =====
                        if same:
                            # 同类，判断是否在聚合与疏远范围内
                            if attr["sep_r_same"] is not None and dist <= attr["sep_r_same"]:
                                sep_same -= offset / dist

                            # 对齐 & 聚集没有硬半径（可后续加）
                            align += self.velocities[j]
                            coh_same += self.positions[j]
                            cnt_same += 1 # 计数

                        # ===== 异类 =====
                        else:
                            align += self.velocities[j]
                            coh_same += self.positions[j]
                            cnt_same += 1 # 计数
                            # 对方是猎物：根据stamina选择追或者盘旋
                            d = self.duration[i]
                            if self.stamina[i]:# if true
                                
                                if d < 1.0:
                                    self.stamina[i] = False
                                    self.locked[i] = -1
                                    continue     
                                
                                # normal: purchase locked one                           
                                if j == self.locked[i]:
                                    coh_other += self.positions[j]
                                    cnt_other += 1
                            else: # if false
                                
                                if d > 20.0:
                                    # 聚集到猎物，计算coh_other
                                    record_prey_id = record_prey(_max_vel,j)
                                else:
                                    # 恢复中，盘旋
                                    if attr["sep_r_other"] is not None and dist <= attr["sep_r_other"]:
                                        sep_other -= offset / dist
                                        cnt_other += 1
                # -----------------------------
                # 汇总同类项
                # -----------------------------
                # 同时更新duration
                if cnt_same > 0: # 对捕食者来说，正常与其他猎物保持靠近
                    align = align / cnt_same - vel_i
                    coh_same = coh_same / cnt_same - pos_i

                if cnt_other > 0: # 猎物在范围内
                    coh_other = coh_other / cnt_other - pos_i
                    sep_other = sep_other / cnt_other
                else: # 休息区间或丢失猎物，或刚刚捕捉到猎物时进入
                    if record_prey_id == -1: # 没有合适的略无
                        self.locked[i] = -1 # lost target
                        self.stamina[i] = False
                    else:
                        self.locked[i] = record_prey_id # 聚焦到最近猎物上
                        record_prey_id = -1
                        self.stamina[i] = True
                    # find a best prey
                    
                # ===== predator =====
                # 体力变化
                a = self.stamina[i]
                if a:
                    self.duration[i] -= 0.2
                else:
                    print("in")
                    self.duration[i] = 0.1 + self.duration[i]
                    

                # 速度更新
                vel_i = (
                    vel_i
                    + (attr["alignment_weight"] or 0.0) * align
                    + (attr["cohesion_weight"] or 0.0) * coh_same
                    + (attr["sep_w_same"] or 0.0) * sep_same
                    + (attr["cohesion_weight_other"] or 0.0) * coh_other * self.stamina[i]
                )
                if self.stamina[i]:
                    vel_i = limit_speed(vel_i, MAX_SPEED_predate)
                else: vel_i = limit_speed(vel_i, MAX_SPEED_rest)
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
        
        # 碰撞 / 捕食逻辑接口
        locked = self._apply_predation()

        self.frame += 1
        
    def calculate_threat(self, dist):
        """in order to reduce the reduction of distance, use one with a more gentle slope

        Args:
            dist (_type_): 

        Returns:
            _type_: _description_
        """
        if dist > 1:
            return np.log(dist)
        return dist
    
        

    def _apply_predation(self):
        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) < 2:
            return

        d2_thr = COLLISION_DIST ** 2

        for i in alive_idx:
            # 只从 predator 出发检测
            # 先定下捕食者
            if self.species_id[i] < NUM_SPECIES_A:
                continue

            if not self.alive[i]:
                continue

            pi = self.positions[i]

            for j in alive_idx:
                if i == j:
                    continue

                # 只检查 prey
                if self.species_id[j] >= NUM_SPECIES_A:
                    continue

                if not self.alive[j]:
                    continue

                pj = self.positions[j]
                dx, dy = pj - pi

                if (dx * dx + dy * dy) <= d2_thr:
                    # predator 吃掉 prey
                    self.alive[j] = False
                    self.velocities[j] = 0.0
                    self.stamina[i] = False
                    return False
                    # 可选：predator 能量恢复接口
                    # if self.attrs["duration"][i] is not None:
                    #     self.attrs["duration"][i] += FEED_GAIN

                    # 一个 prey 只死一次
                    # 如果你希望 predator 一帧只能吃一只：
                    # break
        return

    
    def alive_count(self):
        return int(self.alive.sum())

    def all_dead(self):
        return self.alive_count() == 0
    

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
    viz.run_animation(sim0, frames=8000, interval=3, stride=1, color_by=COLOR_BY)

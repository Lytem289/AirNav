import cv2
import numpy as np
import json
import random
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict, defaultdict

# --- 配置参数 ---
# IMG_SIZE 根据 GPU 配置调整 (必须与训练 YAML 的 image_max_pixels 平方根一致, 避免无意义重采样):
#   - 单卡 4090 (24GB): IMG_SIZE = 336, image_max_pixels = 112896
#   - 双卡 4090 (48GB): IMG_SIZE = 448, image_max_pixels = 200704
IMG_SIZE = 336

# === GSD 策略 (PLAN §4.1 二次修订: 视觉反推 GSD + 围绕真值小幅抖动) ===
# 实测本地 DOTA v2.0: 98.8% train 图 meta 标 gsd:None。原"固定 GSD=0.5 + 大范围抖动
# [0.35, 0.7]" 策略在 DOTA 真实 GSD 跨度 (0.1-3 m/px) 下导致:
#   - 截太小: 真实 GSD=0.1 的高分辨率卫星图被当成 0.5, 物理视野低估 5×, 目标贴满画面;
#   - 截太大: 真实 GSD=2.0 的低分辨率图被当成 0.5, 物理视野高估 4×, 小目标几乎消失;
#   - 模糊:   crop_size 跨度 143-858 px 全部 resize 到 IMG_SIZE=336, 上/下采样比悬殊。
# 当前策略 (二次修订):
#   - 视觉反推 GSD: 用 DOTA OBB 标注 + 类别物理先验 (PRIOR_SIZE_M) 反推每张图真实 GSD,
#     ~85-90% 图可成功反推 (small-vehicle 覆盖率 63%+ + L 类大场地)。
#   - 训练端 ppm 抽样: 围绕反推真值 ±20% 抖动 (GSD_JITTER_RATIO_REAL), 保留数据增强但
#     不再依赖大范围抖动去"覆盖未知"。
#   - 反推失败兜底: 退化到旧的大范围抖动 [0.35, 0.7], 覆盖 fallback 图像的 GSD 不确定性。
#   - 评估端 / 默认: 固定 FIXED_PPM=2.0 (GSD=0.5), 保证 SR/SPL 复现性。
FIXED_GSD = 0.5      # m/pixel, 评估端基准 / 视觉反推失败时的兜底
FIXED_PPM = 1.0 / FIXED_GSD  # = 2.0 px/m
POTSDAM_GSD = 0.05   # ISPRS Potsdam ground sampling distance, meters per pixel.

POTSDAM_LABEL_COLORS_RGB = {
    'impervious-surface': (255, 255, 255),
    'building': (0, 0, 255),
    'low-vegetation': (0, 255, 255),
    'tree': (0, 255, 0),
    'car': (255, 255, 0),
    'clutter': (255, 0, 0),
}
POTSDAM_MIN_COMPONENT_AREA = {
    'car': 12,
    'building': 500,
    'tree': 180,
    'low-vegetation': 800,
    'impervious-surface': 2000,
}

# === 视觉反推 GSD 的类别物理先验 (PLAN §4.1 修订: DOTA 99% meta gsd:None) ===
# 思路: DOTA OBB 标注的最长边像素数 + 类别真实物理长边 = 真实 GSD (m/px)。
# 优势: 让每张图 (而非全局假设 GSD=0.5) 拥有接近真实的视野尺度,
#       彻底消除"有时截太小、有时截太大、resize 后模糊"的视觉退化。
# 类别选择原则:
#   - 高权重 (2.5-3.0): 全球工业标准, 物理尺寸方差极小 (小车 / 网球场 / 篮球场 / 田径场)
#   - 中权重 (1.0-2.0): 国际标准但有不同等级 (足球场 / 棒球场 / 大型车辆 / 环岛 / 停机坪)
#   - 低权重 (0.3-0.8): 物理尺寸方差较大, 但加权中位数仍可贡献信号 (船 / 飞机 / 油罐 / 直升机 / 泳池)
#   - 不纳入: bridge / harbor / airport / container-crane (跨数量级, 反推不可信)
# weight 反映"该类别贡献的 GSD 估计可信度", 加权中位数会偏向高权重类别。
# !!! 实测教训: 本地 419 张 DOTA, 初版 9 类只覆盖 26.3% (大量港口/工业图无任何先验类别),
#     扩充到 14 类后预期覆盖率提升到 50-70%。剩余仍 fallback 大范围抖动 [0.35, 0.7]。
PRIOR_SIZE_M = {
    # 高权重: 国际标准 + 全球一致
    'small-vehicle':      (4.5,   3.0),   # 轿车长 4-5m, 样本最密集
    'tennis-court':       (23.77, 3.0),   # ITF 国际标准
    'basketball-court':   (28.0,  2.5),   # FIBA 国际标准
    'ground-track-field': (118.0, 2.5),   # 400m 跑道外接矩形长边 ~115-120m
    # 中权重: 国际标准但有等级差异
    'large-vehicle':      (12.0,  2.0),   # 公交/卡车 10-14m
    'soccer-ball-field':  (105.0, 2.0),   # FIFA 长边 100-110m
    'baseball-diamond':   (27.4,  2.0),   # MLB 内场对角线
    'helipad':            (20.0,  1.5),   # H 标志直径
    'roundabout':         (35.0,  1.0),   # 环岛直径 20-60m, 中位 ~35m
    # 低权重: 方差较大, 仅作辅助信号
    'swimming-pool':      (25.0,  0.8),   # 短池 25m 主流 (家用更小, 奥运 50m 少数)
    'plane':              (35.0,  0.8),   # 民航 ~40m, 小型 ~20m
    'storage-tank':       (15.0,  0.5),   # 工业油罐直径 10-30m, 极低权
    'helicopter':         (15.0,  0.5),   # 民用直升机 ~12-18m
    'ship':               (40.0,  0.3),   # 港口船只 20-80m, 极低权但密集时有用
    # bridge / harbor / airport / container-crane 物理尺寸跨数量级 (50m-5km), 不纳入
}
GSD_JITTER_RATIO_REAL = 0.20    # 反推成功: 围绕真值 ±20% 抖动 (数据增强)
GSD_FALLBACK_JITTER = (0.35, 0.7)  # 反推失败: 退化到 PLAN §4.1 的大范围抖动

# === 多高度档位配置 (PLAN §4.1.1) ===
# 训练数据必须覆盖 25/50/75m 三档, 否则评估期高空档位会明显退化。
# 每条轨迹生成前先按 HEIGHT_BIN_RATIOS 采样高度档位, 再取对应 grid_size / fov。
HEIGHT_BINS = {
    25:  {'grid_size_m': 5.0,  'fov_m': 50.0},
    50:  {'grid_size_m': 10.0, 'fov_m': 100.0},
    75:  {'grid_size_m': 15.0, 'fov_m': 150.0},
}
HEIGHT_BIN_RATIOS = {25: 0.50, 50: 0.30, 75: 0.20}  # 25m 主导, 50/75m 验证多尺度泛化
MIN_SCENARIO_RATIO_FLOORS = {'E': 0.05}   # 至少保留一部分近终点 STOP 监督
MIN_HEIGHT_RATIO_FLOORS = {75: 0.10}      # 75m 低于该比例时泛化会明显退化

ACTION_GRID_SIZE_M = 5.0
TERMINAL_ACTION_GRID_SIZE_M = 2.5
TERMINAL_RADIUS_FACTOR = 2.5
TERMINAL_MAX_STEP = 1
LOCAL_SEARCH_ACTION_GRID_SIZE_M = TERMINAL_ACTION_GRID_SIZE_M
LOCAL_SEARCH_MAX_STEP = 2
RECOVERY_START_RATIO = 0.10
RECOVERY_YAW_STD = 8.0
RECOVERY_DIST_STD = 2.0
MIN_TRAJECTORY_MOVES = 2
DEFAULT_MAX_WORKERS = 16
LANDMARK_MAX_DISTANCE_M = 80.0
LANDMARK_NEARBY_DISTANCE_M = 80.0
MIN_START_DISTANCE_M = 20.0
DEFAULT_TRAJECTORIES_PER_IMAGE = None
DEFAULT_PROGRESS_INTERVAL = 10
IMAGE_CACHE_SIZE = 16
ASSISTANT_FORMATS = ("thought_action", "action_only")

# === Crop 像素范围约束 (与视觉反推 GSD 配合) ===
# 实测: DOTA 图 GSD 跨度 0.2-2.8 m/px, 极端高 GSD 图配低高度档位会产生 30-100 像素 crop,
# resize 到 IMG_SIZE=336 后必然模糊 (3-10× 上采样)。
# 因此对每张图按 real_gsd 过滤掉"会导致 crop_px 过小或过大"的高度档位。
CROP_PX_MIN = 200      # 上采样比 ≤ 336/200 = 1.68×, 视觉退化可接受
CROP_PX_MAX = 1200     # 下采样比 ≤ 1200/336 = 3.57×, 小目标仍可辨识
# Potsdam 的 GSD 固定为 0.05m/px; 若沿用 1200 上限, 25/50/75m 档位会分别对应
# 1000/2000/3000 px, 其中 50/75m 会被整体过滤掉, 无法形成真正的多高度分布。
CROP_PX_MAX_POTSDAM = 3200

MAX_STEP = 5            # 单步最大网格数 (无视高度, 始终是 ±5 网格的相对量)
MAX_TURN_DEG_PER_STEP = 25.0
FORWARD_BIAS_MIN_RATIO = 0.60
REAR_TURN_SLOWDOWN_RATIO = 0.50
NOISE_YAW_STD = 5.0     # 偏航角噪声标准差(度)
NOISE_DIST_STD = 2.0    # 纵向距离噪声标准差(米): 沿当前飞行方向的步长误差
NOISE_LATERAL_STD = 1.0 # 横向漂移噪声标准差(米): 与飞行方向正交的风偏 / IMU 漂移
                        # PLAN §4.3 的 noise_x, noise_y, 在图像坐标系下独立加到 dx_img / dy_img
                        # 默认 1.0m: 横向风偏一般小于纵向控制误差; 真机标定数据可重新校准
SEARCH_LANDMARK_MAX_STEPS = 5   # SEARCH_LANDMARK 最多允许盲搜步数, 超过则重采样

# STOP 阈值 (训练-评估完全一致, 见 PLAN §1.3 / §3.2)
STOP_EPS_MIN = 8.0     # 最小停止半径 (米), 覆盖视觉定位与控制离散误差
STOP_ALPHA = 0.75      # 网格倍数, max(8m, 0.75×grid_size)

# 调控制闭环时统一监督显式 STOP, 避免 (0,0) 稀释终止动作学习。
STOP_AS_ZERO_RATIO = 0.0

SCENARIO_KEYS = ('A', 'B', 'C', 'D', 'E')
NOMINAL_SCENARIO_RATIOS = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.15, 'E': 0.10}
SCENARIO_DEVIATION_TOL = 0.03
SCENARIO_COMPENSATION_CAP_RATIO = 0.12    # 单场景通过补偿额外吞掉的最大比例
DEFAULT_TRAIN_RATIO = 0.90
DEFAULT_SPLIT_SEED = 42


class _ImageLRUCache:
    """Thread-safe in-memory image cache for repeated per-pid reads."""

    def __init__(self, max_items):
        self.max_items = max(1, int(max_items))
        self._lock = threading.RLock()
        self._items = OrderedDict()

    def get(self, path, flags=cv2.IMREAD_COLOR):
        key = (path, int(flags))
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                self._items.move_to_end(key)
                return cached

        img = cv2.imread(path, flags)
        if img is None:
            return None

        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                self._items.move_to_end(key)
                return cached
            self._items[key] = img
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return img


_IMAGE_CACHE = _ImageLRUCache(IMAGE_CACHE_SIZE)
_IMAGE_SIZE_CACHE = {}
_IMAGE_SIZE_CACHE_LOCK = threading.RLock()
_ANNOTATION_CACHE = {}
_ANNOTATION_CACHE_LOCK = threading.RLock()


def read_cached_image(path, flags=cv2.IMREAD_COLOR):
    """Read an image through the shared process-local LRU cache."""
    return _IMAGE_CACHE.get(path, flags=flags)


def get_cached_image_shape(path):
    """Cache image shape separately so callers that only need H/W avoid re-decoding later."""
    with _IMAGE_SIZE_CACHE_LOCK:
        cached = _IMAGE_SIZE_CACHE.get(path)
        if cached is not None:
            return cached

    img = read_cached_image(path)
    if img is None:
        return None
    shape = img.shape[:2]

    with _IMAGE_SIZE_CACHE_LOCK:
        _IMAGE_SIZE_CACHE[path] = shape
    return shape

def compatible_height_bins(real_gsd, min_crop_px=CROP_PX_MIN, max_crop_px=CROP_PX_MAX,
                           dataset_name=None):
    """根据反推 GSD, 选出该图能稳定支持的高度档位 (crop_size 落在合理像素区间)。

    Args:
        real_gsd: 视觉反推 GSD (m/px), None 表示反推失败 (返回所有档位, 不做过滤).
        min_crop_px: crop_size 下限 (像素), 低于此值会过度上采样导致模糊.
        max_crop_px: crop_size 上限 (像素), 高于此值会过度下采样导致小目标消失.

    Returns:
        List[int]: 兼容的高度档位 (25/50/75 的子集).
                   极端 GSD 情况下若无任何档位兼容, 返回最接近合理区间的单一档位 (兜底).
    """
    if (dataset_name or '').lower() == 'potsdam':
        max_crop_px = max(max_crop_px, CROP_PX_MAX_POTSDAM)
    if real_gsd is None:
        return list(HEIGHT_BINS.keys())     # fallback: 全档位

    compat = []
    for h, cfg in HEIGHT_BINS.items():
        crop_px = cfg['fov_m'] / real_gsd
        if min_crop_px <= crop_px <= max_crop_px:
            compat.append(h)

    if not compat:
        # 极端 GSD: 选出 crop_px 离合理区间中点最近的档位作为兜底
        target = (min_crop_px + max_crop_px) / 2
        best = min(HEIGHT_BINS,
                   key=lambda h: abs(HEIGHT_BINS[h]['fov_m'] / real_gsd - target))
        compat = [best]
    return compat


def sample_height_bin(allowed_heights=None, preferred_height=None):
    """按 HEIGHT_BIN_RATIOS 采样高度档位, 返回 (height_m, grid_size_m, fov_m).

    Args:
        allowed_heights: 限定可采样的高度档位 (来自 compatible_height_bins),
                         None 表示从所有档位按 HEIGHT_BIN_RATIOS 采样.
        preferred_height: 若可兼容, 优先使用该高度档位。
    """
    if allowed_heights is None:
        heights = list(HEIGHT_BIN_RATIOS.keys())
    else:
        heights = list(allowed_heights)
    if preferred_height is not None and preferred_height in heights:
        h = preferred_height
        cfg = HEIGHT_BINS[h]
        return h, cfg['grid_size_m'], cfg['fov_m']
    weights = [HEIGHT_BIN_RATIOS[h] for h in heights]
    h = random.choices(heights, weights=weights, k=1)[0]
    cfg = HEIGHT_BINS[h]
    return h, cfg['grid_size_m'], cfg['fov_m']

# ============================================================
# 坐标系工具函数 (统一所有 yaw / 向量转换, 防止坑)
# 约定:
#   - Yaw=0 是正北, 顺时针为正 (东=90, 南=180, 西=270)
#   - 图像坐标系: x 向右, y 向下 (OpenCV 默认) -> 正北 = -y 方向
#   - 网格局部坐标系: x 向右(+5右), y 向前(+5前)
# ============================================================

def world_to_local(vec_x_img, vec_y_img, yaw_deg):
    """
    图像坐标系下的向量 -> 无人机自身坐标系 (右, 前)。
    返回 (local_x_right, local_y_forward)。
    !!! 全文唯一的坐标变换实现, 所有调用方必须复用此函数, 严禁 inline 重写 (历史 bug 来源)。
    """
    yaw_rad = math.radians(yaw_deg)
    local_x =  vec_x_img * math.cos(yaw_rad) + vec_y_img * math.sin(yaw_rad)
    local_y =  vec_x_img * math.sin(yaw_rad) - vec_y_img * math.cos(yaw_rad)
    return local_x, local_y

def yaw_step(yaw_deg, dist_pixel):
    """
    给定全局朝向, 计算在图像坐标系下的位移 (dx_img, dy_img)。
    Yaw=0 正北 = 图像 -y; Yaw=90 正东 = 图像 +x。
    """
    yaw_rad = math.radians(yaw_deg)
    dx_img =  dist_pixel * math.sin(yaw_rad)
    dy_img = -dist_pixel * math.cos(yaw_rad)
    return dx_img, dy_img


def yaw_towards_point(from_x, from_y, to_x, to_y):
    """Return heading degrees that points from one image-coordinate point to another."""
    return (math.degrees(math.atan2(to_x - from_x, from_y - to_y)) + 360.0) % 360.0


def stop_eps_m(grid_size_m):
    """统一 STOP 半径计算: 训练标签 / 评估 SR 共用此公式。"""
    return max(STOP_EPS_MIN, STOP_ALPHA * grid_size_m)


def object_distance_m(obj_a, obj_b, ppm):
    """Return Euclidean distance between two annotated objects in meters."""
    return math.hypot(obj_a['cx'] - obj_b['cx'], obj_a['cy'] - obj_b['cy']) / ppm


def max_landmark_distance_for_joint_terminal_view(fov_m, grid_size_m):
    """Maximum landmark-target distance that guarantees joint visibility near STOP.

    The UAV stops within stop_eps_m(grid_size_m) of the target. To guarantee that the
    final frame can contain both target and landmark regardless of the terminal yaw,
    we use a conservative sufficient condition:

        dist(target, landmark) + stop_eps <= fov_m / 2

    because the target is within stop_eps of the UAV, and any object whose Euclidean
    distance to the UAV is <= half-FOV is inside the square view under any yaw.
    """
    return max(0.0, (fov_m / 2.0) - stop_eps_m(grid_size_m))


def object_in_fov(obj, current_x, current_y, current_yaw, ppm, fov_m):
    """Check whether an annotated object is inside the current square FOV."""
    vec_x_px = obj['cx'] - current_x
    vec_y_px = obj['cy'] - current_y
    local_x_px, local_y_px = world_to_local(vec_x_px, vec_y_px, current_yaw)
    local_x_m = local_x_px / ppm
    local_y_m = local_y_px / ppm
    return max(abs(local_x_m), abs(local_y_m)) <= (fov_m / 2.0)


def spatial_word_from_local(local_x_px, local_y_px, grid_pixel):
    """Convert a local right/forward vector to a coarse Chinese spatial phrase."""
    return get_spatial_word(
        round(local_x_px / grid_pixel),
        round(local_y_px / grid_pixel),
    )

def normalize_action_to_max_step(grid_x, grid_y, max_step=MAX_STEP):
    """按比例缩放动作到合法动作空间, 保持方向不变。"""
    max_abs = max(abs(grid_x), abs(grid_y))
    if max_abs <= max_step:
        return int(round(grid_x)), int(round(grid_y))

    scale = max_step / max_abs
    scaled_x = int(round(grid_x * scale))
    scaled_y = int(round(grid_y * scale))
    scaled_x = max(-max_step, min(max_step, scaled_x))
    scaled_y = max(-max_step, min(max_step, scaled_y))
    return scaled_x, scaled_y

def quantize_local_action(local_x, local_y, grid_pixel, max_step=MAX_STEP):
    """将局部连续位移量化为网格动作, 超范围时按比例缩放。"""
    grid_x = local_x / grid_pixel
    grid_y = local_y / grid_pixel
    grid_x, grid_y = normalize_action_to_max_step(grid_x, grid_y, max_step=max_step)
    return int(round(grid_x)), int(round(grid_y))


def smooth_greedy_action(grid_x, grid_y,
                         max_turn_deg=MAX_TURN_DEG_PER_STEP,
                         forward_bias_min_ratio=FORWARD_BIAS_MIN_RATIO,
                         rear_turn_slowdown_ratio=REAR_TURN_SLOWDOWN_RATIO,
                         max_step=MAX_STEP):
    """Limit turn rate and bias motion forward to avoid tiny circling paths."""
    if grid_x == 0 and grid_y == 0:
        return 0, 0

    dist = math.hypot(grid_x, grid_y)
    if dist <= 1e-6:
        return 0, 0

    max_turn_deg = max(1.0, min(89.0, float(max_turn_deg)))
    forward_bias_min_ratio = max(0.0, min(0.95, float(forward_bias_min_ratio)))
    rear_turn_slowdown_ratio = max(0.1, min(1.0, float(rear_turn_slowdown_ratio)))

    desired_turn_deg = math.degrees(math.atan2(grid_x, grid_y))
    limited_turn_deg = max(-max_turn_deg, min(max_turn_deg, desired_turn_deg))
    limited_turn_rad = math.radians(limited_turn_deg)

    speed_scale = rear_turn_slowdown_ratio if abs(desired_turn_deg) > 90.0 else 1.0
    smoothed_dist = max(1.0, dist * speed_scale)

    raw_forward = smoothed_dist * math.cos(limited_turn_rad)
    raw_right = smoothed_dist * math.sin(limited_turn_rad)
    min_forward = max(1.0, smoothed_dist * forward_bias_min_ratio)
    forward = max(raw_forward, min_forward)
    right = raw_right

    smooth_x = int(round(right))
    smooth_y = int(round(forward))
    smooth_x, smooth_y = normalize_action_to_max_step(smooth_x, smooth_y, max_step=max_step)

    if smooth_x == 0 and smooth_y == 0:
        smooth_y = 1

    return smooth_x, smooth_y


def action_to_local_meters(grid_x, grid_y, action_grid_size_m):
    """Convert discrete action to local right/forward displacement in meters."""
    return grid_x * action_grid_size_m, grid_y * action_grid_size_m


def execute_grid_action(current_x, current_y, current_yaw, grid_x, grid_y, ppm,
                        action_grid_size_m, noise_yaw_std=0.0,
                        noise_dist_std=0.0, noise_lateral_std=0.0, rng=None):
    """Execute one discrete grid action with the same turn-then-move model used by eval."""
    if rng is None:
        rng = random

    local_delta_yaw_deg = math.degrees(math.atan2(grid_x, grid_y))
    next_yaw = (current_yaw + local_delta_yaw_deg
                + rng.gauss(0, noise_yaw_std) + 360.0) % 360.0

    move_dist_m = math.hypot(grid_x, grid_y) * action_grid_size_m
    noisy_dist_m = max(0.0, move_dist_m + rng.gauss(0, noise_dist_std))
    move_dist_px = noisy_dist_m * ppm
    dx_img, dy_img = yaw_step(next_yaw, move_dist_px)
    dx_img += rng.gauss(0, noise_lateral_std) * ppm
    dy_img += rng.gauss(0, noise_lateral_std) * ppm
    next_x = current_x + dx_img
    next_y = current_y + dy_img
    return next_x, next_y, next_yaw, move_dist_m


def candidate_action_set(max_step=MAX_STEP):
    """Enumerate legal discrete actions excluding explicit STOP."""
    candidates = []
    for gx in range(-max_step, max_step + 1):
        for gy in range(-max_step, max_step + 1):
            if gx == 0 and gy == 0:
                continue
            if max(abs(gx), abs(gy)) > max_step:
                continue
            candidates.append((gx, gy))
    candidates.sort(key=lambda a: (math.hypot(a[0], a[1]), abs(a[0]), -a[1]))
    return candidates


def search_best_grid_action(current_x, current_y, current_yaw, target, ppm,
                            action_grid_size_m, stop_radius_m,
                            candidate_max_step=MAX_STEP,
                            max_turn_deg=MAX_TURN_DEG_PER_STEP):
    """Search the best legal discrete action under the deployment execution model."""
    vec_x = target['cx'] - current_x
    vec_y = target['cy'] - current_y
    dist_now_m = math.hypot(vec_x, vec_y) / ppm
    if dist_now_m <= stop_radius_m:
        return None

    best = None
    for gx, gy in candidate_action_set(candidate_max_step):
        turn_deg = math.degrees(math.atan2(gx, gy))
        if abs(turn_deg) > max_turn_deg:
            continue

        next_x, next_y, next_yaw, move_dist_m = execute_grid_action(
            current_x, current_y, current_yaw, gx, gy, ppm,
            action_grid_size_m,
            noise_yaw_std=0.0, noise_dist_std=0.0, noise_lateral_std=0.0,
        )
        next_dist_m = math.hypot(target['cx'] - next_x, target['cy'] - next_y) / ppm
        progress_m = dist_now_m - next_dist_m
        if progress_m <= 0:
            continue

        next_local_x, next_local_y = world_to_local(
            target['cx'] - next_x,
            target['cy'] - next_y,
            next_yaw,
        )
        next_bearing_deg = math.degrees(math.atan2(next_local_x, next_local_y))

        overshoot_penalty = max(0.0, move_dist_m - dist_now_m) * 4.0
        orbit_penalty = abs(next_bearing_deg) * 0.03
        turn_penalty = abs(turn_deg) * 0.02
        action_penalty = math.hypot(gx, gy) * 0.15
        stop_bonus = 3.0 if next_dist_m <= stop_radius_m else 0.0

        cost = (next_dist_m + overshoot_penalty + orbit_penalty +
                turn_penalty + action_penalty - stop_bonus)
        item = {
            'grid_x': gx,
            'grid_y': gy,
            'next_x': next_x,
            'next_y': next_y,
            'next_yaw': next_yaw,
            'next_dist_m': next_dist_m,
            'move_dist_m': move_dist_m,
            'cost': cost,
        }
        if best is None or item['cost'] < best['cost']:
            best = item

    return best


def search_landmark_action(current_x, current_y, current_yaw, landmark, ppm,
                           action_grid_size_m, max_turn_deg=MAX_TURN_DEG_PER_STEP,
                           forward_bias_min_ratio=FORWARD_BIAS_MIN_RATIO,
                           rear_turn_slowdown_ratio=REAR_TURN_SLOWDOWN_RATIO,
                           candidate_max_step=MAX_STEP):
    """Plan a coarse search step toward a not-yet-visible landmark."""
    vec_x = landmark['cx'] - current_x
    vec_y = landmark['cy'] - current_y
    local_x, local_y = world_to_local(vec_x, vec_y, current_yaw)
    grid_pixel = action_grid_size_m * ppm
    raw_gx, raw_gy = quantize_local_action(
        local_x,
        local_y,
        grid_pixel,
        max_step=candidate_max_step,
    )

    if raw_gx == 0 and raw_gy == 0:
        if abs(local_x) >= abs(local_y):
            raw_gx = 1 if local_x >= 0 else -1
            raw_gy = 1
        else:
            raw_gx = 0
            raw_gy = 1 if local_y >= 0 else -1

    grid_x, grid_y = smooth_greedy_action(
        raw_gx,
        raw_gy,
        max_turn_deg=max_turn_deg,
        forward_bias_min_ratio=forward_bias_min_ratio,
        rear_turn_slowdown_ratio=rear_turn_slowdown_ratio,
        max_step=candidate_max_step,
    )
    if grid_x == 0 and grid_y == 0:
        return None

    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "search_word": spatial_word_from_local(local_x, local_y, grid_pixel),
    }


def forward_search_action(candidate_max_step=MAX_STEP):
    """Search forward when upstream navigation has already roughly aimed at the landmark."""
    return {
        "grid_x": 0,
        "grid_y": max(1, min(MAX_STEP, int(candidate_max_step))),
        "search_word": "前方",
    }


def local_scan_action(step, candidate_max_step=LOCAL_SEARCH_MAX_STEP):
    """Small local sweep used after GPS has brought the UAV near the landmark."""
    pattern = (
        (0, 2),
        (1, 2),
        (-1, 2),
        (2, 1),
        (-2, 1),
        (0, 2),
        (1, 1),
        (-1, 1),
    )
    gx, gy = pattern[int(step) % len(pattern)]
    gx, gy = normalize_action_to_max_step(gx, gy, max_step=candidate_max_step)
    return {
        "grid_x": gx,
        "grid_y": max(1, gy),
        "search_word": get_spatial_word(gx, max(1, gy)),
    }


def relation_guided_search_action(landmark_local_x_px, landmark_local_y_px,
                                  relation_word, relation_dist_m, current_yaw,
                                  ppm, action_grid_size_m,
                                  step, candidate_max_step=LOCAL_SEARCH_MAX_STEP,
                                  max_turn_deg=MAX_TURN_DEG_PER_STEP):
    """Search from the observed landmark toward the instruction relation, without target coordinates."""
    if landmark_local_x_px is None or landmark_local_y_px is None:
        return local_scan_action(step, candidate_max_step=candidate_max_step)

    relation_offset = relation_word_to_local_offset(
        relation_word, relation_dist_m, current_yaw, ppm
    )
    if relation_offset is None:
        return local_scan_action(step, candidate_max_step=candidate_max_step)

    desired_x = landmark_local_x_px + relation_offset[0]
    desired_y = landmark_local_y_px + relation_offset[1]
    grid_pixel = action_grid_size_m * ppm
    raw_gx, raw_gy = quantize_local_action(
        desired_x,
        desired_y,
        grid_pixel,
        max_step=candidate_max_step,
    )
    grid_x, grid_y = smooth_greedy_action(
        raw_gx,
        raw_gy,
        max_turn_deg=max_turn_deg,
        forward_bias_min_ratio=0.35,
        rear_turn_slowdown_ratio=0.35,
        max_step=candidate_max_step,
    )
    if grid_x == 0 and grid_y == 0:
        return local_scan_action(step, candidate_max_step=candidate_max_step)
    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "search_word": spatial_word_from_local(desired_x, desired_y, grid_pixel),
    }


def perturb_local_search_start(start_x, start_y, start_yaw, landmark, target,
                               ppm, fov_m, img_shape, max_attempts=8):
    """Inject small closed-loop drift so trajectories include recovery states."""
    if landmark is None or random.random() >= RECOVERY_START_RATIO:
        return start_x, start_y, start_yaw, False

    H, W = img_shape[:2]
    crop_half_px = (fov_m * ppm * 1.414) / 2 + 4
    for _ in range(max_attempts):
        yaw = (start_yaw + random.gauss(0.0, RECOVERY_YAW_STD) + 360.0) % 360.0
        dist_m = max(0.0, random.gauss(RECOVERY_DIST_STD, RECOVERY_DIST_STD * 0.35))
        dx, dy = yaw_step(yaw, dist_m * ppm)
        x = start_x + dx
        y = start_y + dy
        if not (crop_half_px < x < W - crop_half_px and crop_half_px < y < H - crop_half_px):
            continue
        landmark_visible = object_in_fov(landmark, x, y, yaw, ppm, fov_m)
        target_visible = object_in_fov(target, x, y, yaw, ppm, fov_m)
        if landmark_visible or target_visible:
            return x, y, yaw, True
    return start_x, start_y, start_yaw, False


def landmark_reachable_within_search_steps(start_x, start_y, start_yaw, landmark,
                                           img_shape, ppm, fov_m,
                                           action_grid_size_m=ACTION_GRID_SIZE_M,
                                           max_turn_deg=MAX_TURN_DEG_PER_STEP,
                                           forward_bias_min_ratio=FORWARD_BIAS_MIN_RATIO,
                                           rear_turn_slowdown_ratio=REAR_TURN_SLOWDOWN_RATIO,
                                           max_steps=SEARCH_LANDMARK_MAX_STEPS):
    """Simulate SEARCH_LANDMARK without noise; require the landmark to enter FOV quickly."""
    if landmark is None:
        return False

    current_x, current_y, current_yaw = start_x, start_y, start_yaw
    extend_size = int(fov_m * ppm * 1.414) + 2
    for _ in range(max(0, int(max_steps)) + 1):
        if object_in_fov(landmark, current_x, current_y, current_yaw, ppm, fov_m):
            return True

        action = search_landmark_action(
            current_x,
            current_y,
            current_yaw,
            landmark,
            ppm=ppm,
            action_grid_size_m=action_grid_size_m,
            max_turn_deg=max_turn_deg,
            forward_bias_min_ratio=forward_bias_min_ratio,
            rear_turn_slowdown_ratio=rear_turn_slowdown_ratio,
            candidate_max_step=MAX_STEP,
        )
        if action is None:
            return False

        next_x, next_y, next_yaw, _ = execute_grid_action(
            current_x,
            current_y,
            current_yaw,
            action["grid_x"],
            action["grid_y"],
            ppm,
            action_grid_size_m,
            noise_yaw_std=0.0,
            noise_dist_std=0.0,
            noise_lateral_std=0.0,
        )
        if not can_crop_without_padding(img_shape, next_x, next_y, extend_size):
            return False
        current_x, current_y, current_yaw = next_x, next_y, next_yaw

    return False

def can_crop_without_padding(img_shape, cx, cy, size):
    """判断指定中心裁剪是否会触发黑边补齐。"""
    h, w = img_shape[:2]
    half = size / 2.0
    return half <= cx <= (w - half) and half <= cy <= (h - half)

def cleanup_generated_images(trajectory):
    """删除已写出的轨迹图片, 避免丢弃轨迹时遗留脏文件。"""
    image_paths = set()
    for step in trajectory:
        image_paths.add(step.get("image_hist"))
        image_paths.add(step.get("image_cur"))

    for path in image_paths:
        if path and os.path.exists(path):
            os.remove(path)

def load_dota_gsd_meta(meta_path):
    """
    可选: 读取 DOTA meta 文件记录原始 gsd 值, 仅作 trajectory 元信息 (不参与裁剪计算)。
    本地实测 98.8% 标 gsd:None, 当前策略不依赖 meta gsd (训练抖动 + 评估固定, 见 PLAN §4.1),
    此函数返回结果仅用于数据质量分析。
    返回 (gsd_or_None, is_numeric)。
    """
    if not meta_path or not os.path.exists(meta_path):
        return None, False
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[:5]:
                if line.strip().lower().startswith('gsd:'):
                    raw = line.split(':', 1)[1].strip()
                    if raw.lower() == 'none':
                        return None, False
                    try:
                        return float(raw), True
                    except ValueError:
                        return None, False
    except IOError:
        pass
    return None, False


def save_original_trajectory_visualization(img, trajectory_points, heading_samples,
                                           target, landmark, output_path,
                                           pid='UNK', traj_id=0):
    """Save an overlay of the trajectory on the original image for debugging."""
    if img is None or not trajectory_points:
        return None

    canvas = img.copy()
    h, w = canvas.shape[:2]
    base = max(1, min(h, w) // 1200)
    line_thickness = max(2, 2 * base)
    point_radius = max(4, 4 * base)
    arrow_len = max(16, 18 * base)
    font_scale = 0.5 + 0.15 * base
    text_thickness = max(1, base)

    if len(trajectory_points) >= 2:
        poly = np.array(
            [[int(round(x)), int(round(y))] for x, y in trajectory_points],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, (0, 255, 0),
                      thickness=line_thickness, lineType=cv2.LINE_AA)

    for idx, (x, y) in enumerate(trajectory_points):
        pt = (int(round(x)), int(round(y)))
        if idx == 0:
            color = (0, 255, 255)
            radius = point_radius + 1
        elif idx == len(trajectory_points) - 1:
            color = (0, 0, 255)
            radius = point_radius + 1
        else:
            color = (0, 200, 0)
            radius = point_radius
        cv2.circle(canvas, pt, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, str(idx), (pt[0] + 4, pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    text_thickness, cv2.LINE_AA)

    for x, y, yaw_deg in heading_samples:
        start = (int(round(x)), int(round(y)))
        dx, dy = yaw_step(yaw_deg, arrow_len)
        end = (int(round(x + dx)), int(round(y + dy)))
        cv2.arrowedLine(canvas, start, end, (255, 255, 0),
                        thickness=max(1, line_thickness - 1),
                        tipLength=0.35, line_type=cv2.LINE_AA)

    def draw_anchor(obj, color, label):
        if obj is None:
            return
        cx = int(round(obj['cx']))
        cy = int(round(obj['cy']))
        cv2.circle(canvas, (cx, cy), point_radius + 5, color,
                   thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, label, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    text_thickness, cv2.LINE_AA)

    draw_anchor(target, (255, 0, 0), f"target:{target['class']}")
    if landmark is not None:
        draw_anchor(landmark, (0, 165, 255), f"landmark:{landmark['class']}")

    header = f"{pid} traj={int(traj_id):06d} steps={max(0, len(trajectory_points) - 1)}"
    cv2.putText(canvas, header, (16, 28), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale + 0.1, (255, 255, 255),
                text_thickness + 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ok = cv2.imwrite(output_path, canvas)
    return output_path if ok else None

def estimate_gsd_from_targets(targets):
    """
    用 DOTA OBB 标注 + 类别物理先验, 反推每张图真实 GSD (m/px)。

    算法:
      1. 遍历所有标注, 只看类别在 PRIOR_SIZE_M 中的目标 (物理尺度稳定的类别);
      2. 对每个目标计算 OBB 最长边像素长度 long_edge_px;
      3. 单点估计 gsd_est = prior_m / long_edge_px;
      4. 用加权中位数 (weight 来自 PRIOR_SIZE_M[cls][1]) 对所有估计聚合, 抗离群。

    返回: 反推 GSD (float) 或 None (无可信先验类别时, 调用方应 fallback)。

    防御:
      - long_edge_px < 8 直接丢弃 (标注误差占比过大)。
      - 加权中位数, 而非均值: 哪怕个别 small-vehicle 是大卡车也不会拉飞。
    """
    estimates = []   # [(gsd_estimate, weight), ...]
    for t in targets:
        cls = t['class']
        if cls not in PRIOR_SIZE_M:
            continue
        prior_m, weight = PRIOR_SIZE_M[cls]
        poly = t['poly']
        edges = [
            math.hypot(poly[(i + 1) % 4][0] - poly[i][0],
                       poly[(i + 1) % 4][1] - poly[i][1])
            for i in range(4)
        ]
        long_edge_px = max(edges)
        if long_edge_px < 8:           # 标注太小, 像素离散化误差占比过大
            continue
        estimates.append((prior_m / long_edge_px, weight))

    if not estimates:
        return None                    # 调用方应 fallback 到 FIXED_GSD + 大范围抖动

    estimates.sort(key=lambda x: x[0])
    total_w = sum(w for _, w in estimates)
    cum_w = 0.0
    for gsd, w in estimates:
        cum_w += w
        if cum_w >= total_w / 2:
            return gsd
    return estimates[-1][0]


def build_pid_gsd_index(pid_index, gsd_clip=(0.10, 2.0)):
    """
    对 pid_index 中每张图反推 GSD, 一次扫描全集, 缓存供 batch_generate 复用。
    !!! 必须在 batch_generate 启动前调用一次, 避免每条轨迹重复 parse_dota_annotation。

    Args:
        pid_index: build_pid_to_path_index() 的返回值。
        gsd_clip:  (min, max) m/px, 反推结果超出此区间认为不可信, 转 fallback。
                   默认 [0.05, 5.0] 涵盖卫星 + 航拍 + 高空俯视的合理范围。

    Returns:
        {pid: real_gsd_or_None}, None 表示反推失败 (调用方应用 fallback 大范围抖动)。
    """
    cache = {}
    n_ok = n_fail = n_clipped = 0
    gsd_distribution = []   # 仅做日志统计
    for pid, entry in list(pid_index.items()):
        if entry.get('real_gsd') is not None:
            est = float(entry['real_gsd'])
            cache[pid] = est
            n_ok += 1
            gsd_distribution.append(est)
            continue
        targets, _, _ = parse_dota_annotation(entry['label'], entry['meta'])
        est = estimate_gsd_from_targets(targets)
        if est is None:
            cache[pid] = None
            n_fail += 1
            continue
        if est < gsd_clip[0] or est > gsd_clip[1]:
            del pid_index[pid]         # 反推值过高清或极低清, 直接从 pid 池排除
            n_clipped += 1
            continue
        cache[pid] = est
        n_ok += 1
        gsd_distribution.append(est)

    if gsd_distribution:
        gsd_distribution.sort()
        n = len(gsd_distribution)
        p25 = gsd_distribution[n // 4]
        p50 = gsd_distribution[n // 2]
        p75 = gsd_distribution[(3 * n) // 4]
        print(f"[gsd] 视觉反推成功: {n_ok}/{len(pid_index)} "
              f"({n_ok / max(1, len(pid_index)):.1%}), "
              f"反推 GSD 分布: p25={p25:.3f} p50={p50:.3f} p75={p75:.3f} m/px")
    print(f"[gsd] 反推失败 (无先验类别, fallback大范围抖动): {n_fail}, "
          f"反推超出 {gsd_clip} 区间 (排除): {n_clipped} "
          f"-> 这些极高/极低清图已从索引中剔除")
    return cache


def pixel_per_meter(real_gsd=None, jitter=True, gsd_jitter=None):
    """获取一个 ppm (px/m) 值, 支持视觉反推 GSD + 围绕真值抖动。

    Args:
        real_gsd: float 或 None
            - float: 该轨迹用的视觉反推 GSD (来自 build_pid_gsd_index 缓存);
                     当 jitter=True 时围绕此真值 ±GSD_JITTER_RATIO_REAL 抖动 (默认 ±20%)。
            - None:  反推失败 / 评估端 / 默认。
                     当 jitter=True (训练): 退化到 PLAN §4.1 的大范围抖动 [0.35, 0.7];
                     当 jitter=False (评估): 固定 FIXED_PPM=2.0 (GSD=0.5)。
        jitter: 是否启用抖动
            - True (训练): 启用上述抖动逻辑;
            - False (评估): 用 real_gsd (若有) 或 FIXED_GSD, 不抖动, 保证复现性。

    !!! 调用契约 (违反会导致训练数据彻底污染):
        - 一条 trajectory / 一个 episode 内部全程使用**同一个** ppm。
        - **由调用方 (batch_generate / run_evaluation_episode 等顶层入口) 在轨迹入口
          调用一次, 把返回值显式传给所有下游函数 (sample_episode,
          compute_instruction_context_at_start, generate_trajectory, crop_uav_view 等)**。
        - 严禁下游函数内部再次调用本函数 —— 那样会产生与上游不一致的随机 ppm,
          让 sample_episode 算出的起点像素位置 / D 场景视野判定 / generate_trajectory
          的实际裁剪三者使用不同的视野尺度, 训练数据语义错位 (历史 bug)。
    """
    if gsd_jitter is not None:
        jitter = gsd_jitter
    if real_gsd is not None:
        if jitter:
            gsd = real_gsd * random.uniform(
                1 - GSD_JITTER_RATIO_REAL, 1 + GSD_JITTER_RATIO_REAL)
            return 1.0 / gsd
        return 1.0 / real_gsd
    # real_gsd is None: 反推失败 / 评估端
    if jitter:
        gsd = random.uniform(*GSD_FALLBACK_JITTER)
        return 1.0 / gsd
    return FIXED_PPM   # 2.0 (GSD=0.5)

def safe_crop(large_img, cx, cy, size):
    """
    带边界补黑的中心裁剪, 避免越界 IndexError。
    """
    H, W = large_img.shape[:2]
    x1 = int(cx - size // 2)
    y1 = int(cy - size // 2)
    x2 = x1 + size
    y2 = y1 + size
    pad_l = max(0, -x1); pad_t = max(0, -y1)
    pad_r = max(0, x2 - W); pad_b = max(0, y2 - H)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    crop = large_img[y1c:y2c, x1c:x2c]
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return crop

def parse_dota_obb_annotation(label_path, meta_path=None):
    """
    解析 DOTA v1.5 / v2.0 标签文件, 返回 (targets, gsd_meta, gsd_is_numeric)。

    !!! 关键事实: DOTA v2.0 的 labelTxt **只含坐标行**, 不再包含 imagesource / gsd
        元数据行 (与 v1.0 的旧文档表述不同)。meta/{P-id}.txt 单独存放, 但本地实测
        98.8% 标 gsd:None, 因此当前策略不再做按图动态裁剪 (PLAN §4.1 修订)。
        gsd_meta 仅作 trajectory 元信息记录, 不影响裁剪逻辑。
    标签格式:
        x1 y1 x2 y2 x3 y3 x4 y4 category difficult
        ...
    !!! 未知类别 (不在 DOTA_CATEGORY_MAP 中的) 也会被保留, 由外层自行过滤;
        difficult 字段在 v1.5/v2.0 都是 0/1 整数, 缺失时兜底为 0。
    """
    targets = []
    gsd_meta, gsd_is_numeric = load_dota_gsd_meta(meta_path) if meta_path else (None, False)
    if not os.path.exists(label_path):
        return targets, gsd_meta, gsd_is_numeric
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                # 防御性: 即便误传了含元数据的文件, 也能跳过 (匹配已知 key, 不依赖 isalpha)
                if ':' in line:
                    head = line.split(':', 1)[0].strip().lower()
                    if head in ('imagesource', 'gsd', 'acquisition dates'):
                        continue
                parts = line.split()
                # 标准 DOTA: 8 坐标 + class (+ difficult). 至少 9 字段。
                if len(parts) < 9:
                    continue
                try:
                    coords = [float(p) for p in parts[:8]]
                except ValueError:
                    continue
                cls = parts[8]
                difficult = 0
                if len(parts) >= 10 and parts[9].lstrip('-').isdigit():
                    difficult = int(parts[9])
                # 4 顶点四边形 -> 中心
                cx = sum(coords[0::2]) / 4.0
                cy = sum(coords[1::2]) / 4.0
                targets.append({
                    'class': cls,
                    'cx': cx,
                    'cy': cy,
                    'poly': [(coords[i], coords[i + 1]) for i in range(0, 8, 2)],
                    'difficult': difficult,
                })
    except IOError:
        pass
    return targets, gsd_meta, gsd_is_numeric


def _is_probably_potsdam_label(label_path):
    ext = os.path.splitext(str(label_path).lower())[1]
    return ext in ('.tif', '.tiff', '.png', '.jpg', '.jpeg')


def parse_potsdam_annotation(label_path, meta_path=None):
    """Parse an ISPRS Potsdam semantic label image into pseudo-instances."""
    targets = []
    label = read_cached_image(label_path, cv2.IMREAD_COLOR)
    if label is None:
        return targets, POTSDAM_GSD, True

    kernel = np.ones((3, 3), np.uint8)
    for cls, rgb in POTSDAM_LABEL_COLORS_RGB.items():
        if cls == 'clutter':
            continue
        bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.uint8)
        mask = cv2.inRange(label, bgr, bgr)
        if cls != 'car':
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = POTSDAM_MIN_COMPONENT_AREA.get(cls, 100)
        for comp_id in range(1, n):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[comp_id, cv2.CC_STAT_LEFT])
            y = int(stats[comp_id, cv2.CC_STAT_TOP])
            w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
            h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
            if w <= 1 or h <= 1:
                continue
            cx, cy = centroids[comp_id]
            targets.append({
                'class': cls,
                'cx': float(cx),
                'cy': float(cy),
                'poly': [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                'difficult': 0,
                'area': area,
            })
    return targets, POTSDAM_GSD, True


def parse_dota_annotation(label_path, meta_path=None):
    """Backward-compatible parser name; dispatches DOTA text or Potsdam mask."""
    cache_key = (label_path, meta_path)
    with _ANNOTATION_CACHE_LOCK:
        cached = _ANNOTATION_CACHE.get(cache_key)
        if cached is not None:
            return cached

    if _is_probably_potsdam_label(label_path):
        parsed = parse_potsdam_annotation(label_path, meta_path)
    else:
        parsed = parse_dota_obb_annotation(label_path, meta_path)

    with _ANNOTATION_CACHE_LOCK:
        existing = _ANNOTATION_CACHE.get(cache_key)
        if existing is not None:
            return existing
        _ANNOTATION_CACHE[cache_key] = parsed
    return parsed


parse_dataset_annotation = parse_dota_annotation

# ============================================================
# 数据集路径索引: P-id -> (image, labelTxt, meta) 三件套
# 用途: DOTA v2.0 数据按 part4/part5/part6 子目录分散存放, 且实际下载常出现
#       "标注/元数据齐全但图像缺失" (本地 419/1830). 在批量数据生成前必须先
#       inner-join 这三类文件, 否则 cv2.imread 会静默返回 None 导致后续崩溃。
# ============================================================
def build_dota_path_index(data_root, split='train'):
    """
    扫描 data/{split}/ 目录, 返回 {P-id: {'img': ..., 'label': ..., 'meta': ...}}。
    仅返回三件套齐全的 P-id (有图 ∩ 有标注 ∩ 有 meta)。

    预期目录结构:
        data/{split}/images/part*/P????.png       (可能分散在 part2/part4/part5/part6)
        data/{split}/labelTxt-v2.0/DOTA-v2.0_{split}/P????.txt
        data/{split}/meta/P????.txt

    返回:
        index: dict, 仅包含三件套齐全的 P-id
        stats: dict, 各类缺失统计, 用于实施前的可用性核查
    """
    import glob
    img_idx = {}
    for img_path in glob.glob(os.path.join(data_root, split, 'images', 'part*', 'P*.png')):
        pid = os.path.splitext(os.path.basename(img_path))[0]
        img_idx[pid] = img_path

    label_dir = os.path.join(data_root, split, f'labelTxt-v2.0', f'DOTA-v2.0_{split}')
    meta_dir  = os.path.join(data_root, split, 'meta')

    index = {}
    miss_label = miss_meta = 0
    for pid, img_path in img_idx.items():
        label_path = os.path.join(label_dir, f'{pid}.txt')
        meta_path  = os.path.join(meta_dir, f'{pid}.txt')
        if not os.path.exists(label_path):
            miss_label += 1
            continue
        if not os.path.exists(meta_path):
            miss_meta += 1
            continue
        index[pid] = {'img': img_path, 'label': label_path, 'meta': meta_path}

    stats = {
        'split': split,
        'images_found': len(img_idx),
        'three_way_joined': len(index),
        'missing_label': miss_label,
        'missing_meta':  miss_meta,
    }
    return index, stats


def _potsdam_tile_id(path):
    name = os.path.splitext(os.path.basename(path))[0].lower()
    name = name.replace('top_potsdam_', '')
    for suffix in (
        '_label_noboundary', '_label_no_boundary', '_label',
        '_rgb', '_irrg', '_dsm', '_ndsm',
    ):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def _collect_potsdam_files(root, wanted):
    exts = ('*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg')
    files = []
    for ext in exts:
        files.extend(__import__('glob').glob(os.path.join(root, '**', ext), recursive=True))
    out = {}
    for path in files:
        low = path.lower().replace('\\', '/')
        base = os.path.basename(low)
        is_label = 'label' in base or '/label' in low or '5_labels' in low
        if wanted == 'image':
            if is_label:
                continue
            if (not any(tag in base for tag in ('_rgb', '_irrg'))
                    and '2_ortho' not in low
                    and '3_ortho' not in low
                    and '/images/' not in low):
                continue
        else:
            if not is_label:
                continue
        pid = _potsdam_tile_id(path)
        if not pid:
            continue
        if pid not in out or ('noboundary' in low and 'noboundary' not in out[pid].lower()):
            out[pid] = path
    return out


def build_potsdam_path_index(data_root, split='train'):
    """Build a Potsdam tile index from common ISPRS folder layouts.

    Supported layouts include split folders such as train/images + train/labels,
    or the original ISPRS layout with 2_Ortho_RGB / 3_Ortho_IRRG and
    5_Labels_* directories under data_root.
    """
    split_root = os.path.join(data_root, split)
    scan_root = split_root if os.path.isdir(split_root) else data_root

    image_idx = _collect_potsdam_files(scan_root, 'image')
    label_idx = _collect_potsdam_files(scan_root, 'label')
    index = {}
    for pid, img_path in image_idx.items():
        label_path = label_idx.get(pid)
        if not label_path:
            continue
        index[pid] = {
            'img': img_path,
            'label': label_path,
            'meta': None,
            'dataset': 'potsdam',
            'real_gsd': POTSDAM_GSD,
        }

    stats = {
        'dataset': 'potsdam',
        'split': split,
        'scan_root': scan_root,
        'images_found': len(image_idx),
        'labels_found': len(label_idx),
        'joined': len(index),
        'missing_label': max(0, len(image_idx) - len(index)),
    }
    return index, stats


def build_pid_to_path_index(data_root, split='train', dataset='dota'):
    dataset = (dataset or 'dota').lower()
    if dataset == 'potsdam':
        return build_potsdam_path_index(data_root, split=split)
    if dataset == 'auto':
        potsdam_index, potsdam_stats = build_potsdam_path_index(data_root, split=split)
        if potsdam_index:
            return potsdam_index, potsdam_stats
        return build_dota_path_index(data_root, split=split)
    return build_dota_path_index(data_root, split=split)


def filter_pid_index(pid_index, pid_list):
    """按给定 P-id 列表过滤索引, 保持输入顺序。"""
    return {pid: pid_index[pid] for pid in pid_list if pid in pid_index}


def limit_pid_index(pid_index, max_images=None, seed=DEFAULT_SPLIT_SEED):
    if max_images is None:
        return dict(pid_index)
    max_images = int(max_images)
    if max_images <= 0:
        raise ValueError('--max_images must be a positive integer')
    pids = sorted(pid_index.keys())
    rng = random.Random(seed)
    rng.shuffle(pids)
    keep = sorted(pids[:min(max_images, len(pids))])
    return filter_pid_index(pid_index, keep)


def split_pid_lists(pid_index, train_ratio=DEFAULT_TRAIN_RATIO, seed=DEFAULT_SPLIT_SEED):
    """
    按 DOTA 图像 ID 做 90:10 划分, 整张图作为最小切分单位。
    返回 (train_pids, val_pids)。
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio 必须在 (0, 1) 内, 当前为 {train_ratio}")

    pids = sorted(pid_index.keys())
    if len(pids) <= 1:
        return pids, []

    rng = random.Random(seed)
    rng.shuffle(pids)

    n_train = int(round(len(pids) * train_ratio))
    n_train = max(1, min(len(pids) - 1, n_train))
    train_pids = sorted(pids[:n_train])
    val_pids = sorted(pids[n_train:])
    return train_pids, val_pids


def write_pid_list(pid_list, output_path):
    """写出 image_ids_train.txt / image_ids_val.txt。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for pid in pid_list:
            f.write(f"{pid}\n")


def write_jsonl(samples, output_path):
    """写出 ShareGPT JSONL。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def write_generation_config(config, output_path):
    """写出本次数据生成的目标分布与关键参数, 供校验脚本复用。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write('\n')


def update_dataset_info(dataset_info_path, train_jsonl_path, eval_jsonl_path):
    """同步注册 uav_full / uav_eval 到 dataset_info.json。"""
    dataset_info_dir = os.path.dirname(os.path.abspath(dataset_info_path))
    os.makedirs(dataset_info_dir, exist_ok=True)

    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    def rel_jsonl(path):
        return os.path.relpath(os.path.abspath(path), dataset_info_dir).replace('\\', '/')

    columns = {"messages": "conversations", "images": "images"}
    dataset_info["uav_full"] = {
        "file_name": rel_jsonl(train_jsonl_path),
        "formatting": "sharegpt",
        "columns": columns,
    }
    dataset_info["uav_eval"] = {
        "file_name": rel_jsonl(eval_jsonl_path),
        "formatting": "sharegpt",
        "columns": columns,
    }

    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        f.write('\n')


def allocate_count_by_ratio(total_count, ratios):
    """
    按比例分配离散配额, 通过最大余数法保证总和严格等于 total_count。
    返回 {key: int_count}。
    """
    if total_count <= 0:
        return {k: 0 for k in ratios}

    raw_counts = {k: total_count * ratios[k] for k in ratios}
    counts = {k: int(math.floor(v)) for k, v in raw_counts.items()}
    remain = total_count - sum(counts.values())
    if remain > 0:
        order = sorted(
            ratios.keys(),
            key=lambda k: (raw_counts[k] - counts[k], ratios[k], k),
            reverse=True,
        )
        for idx in range(remain):
            counts[order[idx % len(order)]] += 1
    return counts


def rebalance_count_with_minimums(total_count, base_counts, min_counts):
    """在总数固定前提下, 给指定 key 强制保底配额。"""
    counts = {k: int(base_counts.get(k, 0)) for k in base_counts}
    if total_count <= 0:
        return counts

    minimums = {
        k: max(0, int(min_counts.get(k, 0)))
        for k in counts
        if int(min_counts.get(k, 0)) > 0
    }
    if not minimums:
        return counts

    for key, minimum in minimums.items():
        counts[key] = max(counts.get(key, 0), minimum)

    overflow = sum(counts.values()) - total_count
    if overflow <= 0:
        return counts

    donors = sorted(
        counts.keys(),
        key=lambda k: (counts[k] - minimums.get(k, 0), counts[k], k),
        reverse=True,
    )
    for key in donors:
        removable = counts[key] - minimums.get(key, 0)
        if removable <= 0:
            continue
        delta = min(removable, overflow)
        counts[key] -= delta
        overflow -= delta
        if overflow <= 0:
            break

    if overflow > 0:
        raise ValueError("minimum counts exceed total_count; unable to rebalance quotas.")
    return counts


def choose_weighted_remaining_count(counts):
    """按剩余配额采样一个 key。"""
    positive_items = [(k, v) for k, v in counts.items() if v > 0]
    if not positive_items:
        return None
    keys = [k for k, _ in positive_items]
    weights = [v for _, v in positive_items]
    return random.choices(keys, weights=weights, k=1)[0]


def apply_ratio_floors(ratios, ratio_floors, scale=10000):
    """把名义比例转换成带保底的目标比例。"""
    base_counts = allocate_count_by_ratio(scale, ratios)
    min_counts = {
        k: int(math.ceil(scale * float(v)))
        for k, v in (ratio_floors or {}).items()
        if v > 0
    }
    adjusted = rebalance_count_with_minimums(scale, base_counts, min_counts)
    total = max(1, sum(adjusted.values()))
    return {k: adjusted.get(k, 0) / total for k in ratios}


# ============================================================
# DOTA 类别 -> 中文短语 + 角色 (L=Landmark, T=Target, B=Both)
# 详见 PLAN §4.2.1
# ============================================================
DOTA_CATEGORY_MAP = {
    'tennis-court':       ('网球场',     'L'),
    'basketball-court':   ('篮球场',     'L'),
    'ground-track-field': ('田径场',     'L'),
    'soccer-ball-field':  ('足球场',     'L'),
    'swimming-pool':      ('游泳池',     'L'),
    'baseball-diamond':   ('棒球场',     'L'),
    'roundabout':         ('环岛',       'L'),
    'harbor':             ('港口',       'L'),
    'bridge':             ('桥梁',       'L'),
    'airport':            ('机场',       'L'),
    'helipad':            ('停机坪',     'L'),
    'storage-tank':       ('储油罐',     'B'),
    'plane':              ('飞机',       'B'),
    'ship':               ('船只',       'T'),
    'large-vehicle':      ('大型车辆',   'T'),
    'small-vehicle':      ('小型车辆',   'T'),
    'helicopter':         ('直升机',     'T'),
    'container-crane':    ('集装箱起重机', 'T'),
}

POTSDAM_CATEGORY_MAP = {
    'car': ('车辆', 'T'),
    'building': ('建筑', 'L'),
    'tree': ('树木', 'L'),
    'low-vegetation': ('低矮植被', 'L'),
    'impervious-surface': ('道路或硬化地面', 'L'),
}

DATASET_CATEGORY_MAP = dict(DOTA_CATEGORY_MAP)
DATASET_CATEGORY_MAP.update(POTSDAM_CATEGORY_MAP)

def cls_to_zh(cls_name):
    return DATASET_CATEGORY_MAP.get(cls_name, (cls_name, 'T'))[0]

def cls_role(cls_name):
    return DATASET_CATEGORY_MAP.get(cls_name, (cls_name, 'T'))[1]

def build_instruction(target, landmark=None, spatial_word=None, target_count_in_view=1, ppm=None):
    """
    构造与训练分布一致的自然语言指令。详见 PLAN §4.2.2。

    长尾兜底优先级:
      1) landmark 不为 None -> 显式形 "先定位 L，再到 L 的相对方位寻找 T"
      2) 视野内同类 > 1 且给定 spatial_word -> 强制消歧形 "画面X的T"
      3) target 自身是 L/B 类 -> 简化形 "去 T"
      4) target 是纯 T 类且无 L 类地标可用 -> 必须传 spatial_word, 走消歧形
         (PLAN §4.2.2 长尾分支: 城市俯视图 90%+ 仅含 small-vehicle, 无 L 类时
          必须用空间词替代地标参照, 否则指令丢失语义)
      5) 真·兜底: 仅当调用方既无 landmark 也无 spatial_word 时使用 (不推荐)
    """
    t_zh = cls_to_zh(target['class'])
    # 1) 有地标: 写清 target 相对 landmark 的关系，避免远地标时语义漂移。
    if landmark is not None:
        l_zh = cls_to_zh(landmark['class'])
        relation_word, relation_dist_m = landmark_relation_context(target, landmark, ppm=ppm)
        relation_text = format_landmark_relation(relation_word, relation_dist_m)
        return f"任务: 先定位{l_zh}，再在{l_zh}的{relation_text}寻找{t_zh}，最终飞到{t_zh}。"
    # 2) 视野内同类 > 1: 必须消歧
    if target_count_in_view > 1 and spatial_word:
        return f"任务: 去画面{spatial_word}的{t_zh}。"
    # 3) target 自身可作地标 (L/B)
    if cls_role(target['class']) in ('L', 'B'):
        return f"任务: 去{t_zh}。"
    # 4) 长尾兜底: 纯 T 类 + 无 L 地标 -> 强制 spatial_word
    if spatial_word:
        return f"任务: 去画面{spatial_word}的{t_zh}。"
    # 5) 真·兜底 (调用方应避免走到这里)
    return f"任务: 去{t_zh}。"

def get_spatial_word(dx, dy):
    word = ""
    if dy > 0: word += "前方"
    elif dy < 0: word += "后方"
    if dx > 0: word = "右" + word
    elif dx < 0: word = "左" + word
    return word if word else "正下方"


def get_image_spatial_word(dx, dy):
    """Image-coordinate relative word: x right, y down."""
    word = ""
    if dy > 0:
        word += "下方"
    elif dy < 0:
        word += "上方"
    if dx > 0:
        word = "右" + word
    elif dx < 0:
        word = "左" + word
    return word if word else "附近"


def landmark_relation_context(target, landmark, ppm=None):
    """Describe where the target is relative to the landmark in global image coordinates."""
    if landmark is None:
        return None, None
    dx = target['cx'] - landmark['cx']
    dy = target['cy'] - landmark['cy']
    relation_word = get_image_spatial_word(dx, dy)
    distance_m = None
    if ppm:
        distance_m = math.hypot(dx, dy) / ppm
    return relation_word, distance_m


def relation_word_to_local_offset(relation_word, distance_m, current_yaw, ppm):
    """Convert a global landmark-target relation into the current local frame."""
    if not relation_word or distance_m is None or ppm is None:
        return None
    sx = 0.0
    sy = 0.0
    if "右" in relation_word:
        sx += 1.0
    if "左" in relation_word:
        sx -= 1.0
    if "下" in relation_word:
        sy += 1.0
    if "上" in relation_word:
        sy -= 1.0
    norm = math.hypot(sx, sy)
    if norm <= 1e-6:
        return None
    dx = (sx / norm) * distance_m * ppm
    dy = (sy / norm) * distance_m * ppm
    return world_to_local(dx, dy, current_yaw)


def format_landmark_relation(relation_word, distance_m):
    if not relation_word:
        return "附近"
    if distance_m is None:
        return f"{relation_word}区域"
    return f"{relation_word}约{distance_m:.0f}米处"


def compute_instruction_context_at_start(all_targets, target, start_x, start_y, start_yaw,
                                         ppm, fov_m, grid_size_m):
    """
    起点 snapshot 计算 instruction 的两个动态字段:
      - target_count_in_view: 起点视野内与 target 同类的物体数 (含目标自身)
      - spatial_word: target 相对起点视野中心的方位词 ("左前方" 等)

    !!! 整条轨迹只在起点调用一次, instruction 全程不变 (PLAN §4.2.2 line 217:
        "spatial_word 由调用方根据起点视野下 target 相对当前视野中心的方位预先计算")。
    !!! 训练 (batch_generate) 与评估 (run_evaluation_episode) 必须同源调用此函数,
        保证 instruction 分布完全一致, 杜绝 train-eval OOD。
    !!! 视野判定与 generate_trajectory 完全一致 (方框 max(|x|,|y|), 而非径向距离)。
    """
    grid_pixel = grid_size_m * ppm
    count_in_view = 0
    for t in all_targets:
        if t['class'] != target['class']:
            continue
        vx_px = t['cx'] - start_x
        vy_px = t['cy'] - start_y
        lx_px, ly_px = world_to_local(vx_px, vy_px, start_yaw)
        if max(abs(lx_px / ppm), abs(ly_px / ppm)) <= fov_m / 2:
            count_in_view += 1

    tvx_px = target['cx'] - start_x
    tvy_px = target['cy'] - start_y
    tlx_px, tly_px = world_to_local(tvx_px, tvy_px, start_yaw)
    spatial_word = get_spatial_word(
        round(tlx_px / grid_pixel),
        round(tly_px / grid_pixel),
    )
    return count_in_view, spatial_word


def generate_trajectory(img_path, ann_path, meta_path, target, landmark, start_x, start_y, start_yaw,
                        ppm, grid_size_m=10.0, fov_m=100.0,
                        output_dir='.', pid='UNK', traj_id=0, height_m=25,
                        vis_output_dir=None,
                        max_turn_deg_per_step=MAX_TURN_DEG_PER_STEP,
                        forward_bias_min_ratio=FORWARD_BIAS_MIN_RATIO,
                        rear_turn_slowdown_ratio=REAR_TURN_SLOWDOWN_RATIO,
                        action_grid_size_m=ACTION_GRID_SIZE_M,
                        terminal_action_grid_size_m=TERMINAL_ACTION_GRID_SIZE_M):
    """
    单条轨迹生成。返回 (trajectory list, failure_reason)。
    - 成功: (trajectory, None)
    - 失败: ([], reason)
    trajectory 每条包含 (历史图, 当前图, 历史动作, 累计位移, action, thought)。
    grid_size_m / fov_m / height_m 作为参数, 支持多高度数据生成 (PLAN §4.1.1)。
      - 调用方应先调 sample_height_bin() 取 (height, grid_size_m, fov_m), 再传入。
      - height_m 仅作 trajectory 元信息记录, 不进 prompt (PLAN §4.1.1 修订: 高度信息隐式在视觉中)。
    !!! ppm 必须由调用方 (batch_generate) 在轨迹入口调用 pixel_per_meter(gsd_jitter=True) 一次,
        显式传入并贯穿整条轨迹的所有计算 (crop_size / grid_pixel / STOP 距离 / 视野判定)。
        严禁内部再次调用 pixel_per_meter() —— 会破坏与起点采样 (sample_episode) 和 instruction
        snapshot (compute_instruction_context_at_start) 的 ppm 一致性 (历史 bug)。
        meta_path 仅用于记录原始 gsd 元信息, 不参与裁剪计算。
    !!! 文件名必须包含 pid + traj_id, 否则多进程并发会覆盖 (历史 bug: 所有 step_0.jpg 共享同名文件,
        JSONL 全部指向同一张最后写入的图, 训练数据彻底污染)。output_dir 建议传绝对路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    def fail(reason):
        cleanup_generated_images(trajectory)
        return [], reason

    large_img = read_cached_image(img_path)
    if large_img is None:
        return [], 'image_read_failed'         # 路径错误或损坏图, 直接放弃
    img_shape = large_img.shape

    # 读取原始 meta gsd 仅作元信息记录
    gsd_meta, gsd_is_numeric = load_dota_gsd_meta(meta_path)

    # ppm 由调用方传入 (见 docstring 调用契约), 内部不再 pixel_per_meter()
    grid_pixel = action_grid_size_m * ppm

    # 无地标时只保留近距局部逼近样本；远距无地标会变成无依据盲飞。
    no_landmark = (landmark is None)
    if no_landmark:
        # 起点判定也走方框, 与运行时判定逻辑一致 (PLAN §4.3)。
        start_vec_x_px = target['cx'] - start_x
        start_vec_y_px = target['cy'] - start_y
        start_local_x_px, start_local_y_px = world_to_local(
            start_vec_x_px, start_vec_y_px, start_yaw)
        start_local_x_m = start_local_x_px / ppm
        start_local_y_m = start_local_y_px / ppm
        if max(abs(start_local_x_m), abs(start_local_y_m)) > (fov_m / 2):
            return [], 'no_landmark_far_start'  # 远距 + 无地标: 拒绝, 外层换图重采样

    current_x, current_y = start_x, start_y
    current_yaw = start_yaw
    current_x, current_y, current_yaw, recovery_start = perturb_local_search_start(
        current_x,
        current_y,
        current_yaw,
        landmark,
        target,
        ppm,
        fov_m,
        img_shape,
    )
    trajectory = []
    history_actions = []        # 近 5 步, 灌入 prompt
    yaw_history = []            # 与 action_history 同步, 用于全局位移积分
    action_history_full = []    # 完整动作序列 (含截断, 不限 5 步), 用于积分
    trajectory_points = [(current_x, current_y)]
    heading_samples = [(current_x, current_y, current_yaw)]

    eps_m = stop_eps_m(grid_size_m)        # 统一 STOP 阈值
    landmark_relation_word, landmark_relation_dist_m = landmark_relation_context(
        target, landmark, ppm=ppm
    )
    landmark_relation_text = format_landmark_relation(
        landmark_relation_word, landmark_relation_dist_m
    )
    seen_landmark_once = False
    seen_target_once = False
    lost_landmark_steps = 0
    reached_stop = False

    # 历史图缓存: 第一步用当前帧自身做"伪历史图" (与训练分布对齐)
    last_img_path = None

    for step in range(20):
        # 1. 判定阶段: target / landmark 是否在当前方形视野内。
        target_vec_x_px = target['cx'] - current_x
        target_vec_y_px = target['cy'] - current_y
        target_local_x_px, target_local_y_px = world_to_local(
            target_vec_x_px, target_vec_y_px, current_yaw)
        target_local_x_m = target_local_x_px / ppm
        target_local_y_m = target_local_y_px / ppm
        target_in_fov = (max(abs(target_local_x_m), abs(target_local_y_m))
                         <= (fov_m / 2))

        landmark_visible = False
        landmark_local_x_px = landmark_local_y_px = None
        if landmark is not None:
            landmark_vec_x_px = landmark['cx'] - current_x
            landmark_vec_y_px = landmark['cy'] - current_y
            landmark_local_x_px, landmark_local_y_px = world_to_local(
                landmark_vec_x_px, landmark_vec_y_px, current_yaw)
            landmark_visible = (
                max(abs(landmark_local_x_px / ppm), abs(landmark_local_y_px / ppm))
                <= (fov_m / 2)
            )
        if landmark_visible:
            lost_landmark_steps = 0
        elif seen_landmark_once:
            lost_landmark_steps += 1
        seen_landmark_once = seen_landmark_once or landmark_visible
        seen_target_once = seen_target_once or target_in_fov

        if target_in_fov or landmark is None:
            # 无 L 类地标场景: 起点已在视野内 (远距已被拒绝), 直接逼近目标。
            phase = "APPROACH_TARGET"
            observation_word = spatial_word_from_local(
                target_local_x_px, target_local_y_px, grid_pixel)
        elif landmark_visible or seen_landmark_once:
            phase = "LANDMARK_GUIDED_SEARCH"
            observation_word = spatial_word_from_local(
                landmark_local_x_px, landmark_local_y_px, grid_pixel
            ) if landmark_visible else None
        else:
            phase = "SEARCH_LANDMARK"
            observation_word = None

        # 2. 终止条件: 统一阈值 max(5m, 0.5 * grid_size) -- 与 §1.3 / §3.2 完全一致
        dist_to_final_m = math.hypot(target['cx'] - current_x,
                                     target['cy'] - current_y) / ppm
        terminal_radius_m = max(eps_m, TERMINAL_RADIUS_FACTOR * action_grid_size_m)
        terminal_zone = bool(target_in_fov and dist_to_final_m <= terminal_radius_m)
        should_stop = bool(target_in_fov and dist_to_final_m <= eps_m)
        if dist_to_final_m <= eps_m:
            moved_steps = len(action_history_full)
            if moved_steps < MIN_TRAJECTORY_MOVES:
                return fail('stopped_too_early')
            if landmark is not None:
                target_visible = object_in_fov(target, current_x, current_y, current_yaw, ppm, fov_m)
                if not target_visible:
                    return fail('stop_without_target_visible')
            step_action_grid_size_m = terminal_action_grid_size_m
            action_str = "STOP"
            grid_x, grid_y = 0, 0
            is_stop = True
        else:
            in_terminal_mode = (dist_to_final_m <= terminal_radius_m)
            step_action_grid_size_m = (terminal_action_grid_size_m
                                       if in_terminal_mode else action_grid_size_m)
            candidate_max_step = TERMINAL_MAX_STEP if in_terminal_mode else MAX_STEP

            if phase == "APPROACH_TARGET":
                best_action = search_best_grid_action(
                    current_x,
                    current_y,
                    current_yaw,
                    target,
                    ppm=ppm,
                    action_grid_size_m=step_action_grid_size_m,
                    stop_radius_m=eps_m,
                    candidate_max_step=candidate_max_step,
                    max_turn_deg=max_turn_deg_per_step,
                )
            elif phase == "LANDMARK_GUIDED_SEARCH":
                step_action_grid_size_m = LOCAL_SEARCH_ACTION_GRID_SIZE_M
                if landmark_visible:
                    best_action = relation_guided_search_action(
                        landmark_local_x_px,
                        landmark_local_y_px,
                        landmark_relation_word,
                        landmark_relation_dist_m,
                        current_yaw,
                        ppm,
                        step_action_grid_size_m,
                        step,
                        candidate_max_step=LOCAL_SEARCH_MAX_STEP,
                        max_turn_deg=max_turn_deg_per_step,
                    )
                else:
                    best_action = local_scan_action(
                        step + lost_landmark_steps,
                        candidate_max_step=LOCAL_SEARCH_MAX_STEP,
                    )
            else:
                step_action_grid_size_m = LOCAL_SEARCH_ACTION_GRID_SIZE_M
                best_action = local_scan_action(
                    step,
                    candidate_max_step=LOCAL_SEARCH_MAX_STEP,
                )

            if best_action is None:
                return fail('planner_no_action')

            grid_x, grid_y = best_action['grid_x'], best_action['grid_y']
            if phase == "APPROACH_TARGET":
                step_action_grid_size_m = (terminal_action_grid_size_m
                                           if in_terminal_mode else action_grid_size_m)

            # (0,0) 边界处理: 与 PLAN §4.3 严格对齐 —— 
            #   若 round 后 (grid_x, grid_y) = (0, 0) 但仍未到达 STOP 半径,
            #   **整条轨迹丢弃** (return []), 而不是强行改成 (0, 1)。
            # 理由: 强行 (0, 1) 会让训练数据出现"明明对准了却被迫前进 1 格"的伪标签,
            #   污染 STOP 决策学习; 早期硬补 1 格的方案已被弃用。
            if grid_x == 0 and grid_y == 0:
                return fail('nonterminal_zero_action')
            action_str = f"({grid_x}, {grid_y})"
            is_stop = False

        # 6. CoT 思维链
        dist_grid = math.hypot(grid_x, grid_y)
        action_word = get_spatial_word(grid_x, grid_y)
        target_zh = cls_to_zh(target['class'])
        landmark_zh = cls_to_zh(landmark['class']) if landmark else None
        if is_stop:
            thought = f"观察当前图像，目标【{target_zh}】在我的正下方，即将到达目标。"
        elif phase == "SEARCH_LANDMARK":
            search_word = best_action.get("search_word", action_word)
            thought = (f"观察当前图像，目标【{target_zh}】和地标【{landmark_zh}】都还没有出现在当前视野中。"
                       f"根据已知任务，我需要先朝【{search_word}】搜索地标【{landmark_zh}】。")
        elif phase == "LANDMARK_GUIDED_SEARCH":
            if landmark_visible:
                thought = (f"观察当前图像，目标【{target_zh}】尚未出现在视野中，"
                           f"但我看到了地标【{landmark_zh}】在【{observation_word}】。"
                           f"任务线索说明目标在地标的【{landmark_relation_text}】，"
                           f"所以我需要向【{action_word}】搜索目标。")
            else:
                thought = (f"观察当前图像，目标【{target_zh}】和地标【{landmark_zh}】暂时都不在视野中。"
                           f"我之前已经定位过地标，目标应在地标的【{landmark_relation_text}】，"
                           f"所以继续向【{action_word}】搜索目标。")
        else:
            dist_desc = "距离较远，尚未到达" if dist_grid > 5 else "距离适中，正在接近"
            thought = (f"观察当前图像，目标【{target_zh}】已经出现在视野的【{observation_word}】。"
                       f"目前【{dist_desc}】，所以我需要向【{action_word}】移动。")

        # 7. 切图 (无黑边方案): 按真实 ppm 决定 crop_size, 先切外接圆大图 -> 旋转 -> 中心裁剪
        crop_size = int(fov_m * ppm)
        extend_size = int(crop_size * 1.414) + 2
        if not can_crop_without_padding(img_shape, current_x, current_y, extend_size):
            return fail('crop_out_of_bounds')
        large_crop = safe_crop(large_img, current_x, current_y, extend_size)
        center = (extend_size // 2, extend_size // 2)
        # cv2 旋转: 正角度逆时针; 我们的 yaw 是顺时针 -> 传 -yaw
        M = cv2.getRotationMatrix2D(center, -current_yaw, 1.0)
        rotated = cv2.warpAffine(large_crop, M, (extend_size, extend_size))
        s = (extend_size - crop_size) // 2
        final_crop = rotated[s:s + crop_size, s:s + crop_size]
        final_crop = cv2.resize(final_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        img_name = os.path.join(output_dir,
                                f"{pid}_traj{int(traj_id):06d}_step{step:02d}.jpg")
        if not cv2.imwrite(img_name, final_crop):
            return fail('image_write_failed')

        # 8. 组装历史图 (T-1 帧). step=0 时复制当前帧自身做占位, 与训练-推理分布一致。
        hist_img_name = last_img_path if last_img_path is not None else img_name

        # 9. 计算累计全局位移 (yaw 积分版本, 详见 PLAN §1.2)
        # !!! PLAN §1.2 修订: 单位从"米"改为"格"(无量纲), 消除虚构 grid_size_m 依赖。
        #     公式去掉 × grid_size_m 乘子, 直接积分 |action| × sin/cos(yaw)。
        #     "北/东"始终指地理方向 (DOTA 图像 -y/+x), 与无人机当前朝向无关。
        cum_north_grid, cum_east_grid = 0.0, 0.0
        for past_yaw, (past_gx, past_gy) in zip(yaw_history, action_history_full):
            past_dist_grid = math.hypot(past_gx, past_gy)   # 网格数, 无量纲
            past_yaw_rad = math.radians(past_yaw)
            cum_east_grid  += past_dist_grid * math.sin(past_yaw_rad)
            cum_north_grid += past_dist_grid * math.cos(past_yaw_rad)

        trajectory.append({
            "image_hist": hist_img_name,        # 第 1 张图 (历史 / T-1)
            "image_cur":  img_name,             # 第 2 张图 (当前 / T)
            "action": action_str,
            "thought": thought,
            "yaw": current_yaw,
            "phase": phase,
            "landmark_relation_word": landmark_relation_word,
            "landmark_relation_dist_m": (
                round(landmark_relation_dist_m, 2)
                if landmark_relation_dist_m is not None else None
            ),
            "seen_landmark_once": seen_landmark_once,
            "seen_target_once": seen_target_once,
            "landmark_visible": landmark_visible,
            "target_visible": target_in_fov,
            "terminal_zone": terminal_zone,
            "should_stop": should_stop,
            "lost_landmark_steps": lost_landmark_steps,
            "recovery_start": recovery_start,
            "history_actions": list(history_actions[-5:]),
            "cumulative_state": {
                "north_grid": round(cum_north_grid, 1),   # 北向累计网格数 (地理方向)
                "east_grid":  round(cum_east_grid,  1),   # 东向累计网格数 (地理方向)
                "steps":      step,
            },
            "height_m":      height_m,         # 高度档位元信息 (不进 prompt, 仅用于按高度分桶评估)
            "grid_size_m":   grid_size_m,      # 该轨迹的 grid_size (元信息)
            "action_grid_size_m": step_action_grid_size_m,
            "fov_m":         fov_m,            # 该轨迹的 fov (元信息)
            "gsd_used":      round(1.0 / ppm, 4),  # 实际裁剪用的 gsd (训练抖动后的真值, 评估端 = 0.5)
            "gsd_meta":      gsd_meta,         # 原始 meta 中的 gsd (None 表示元数据未给数值)
            "gsd_is_numeric": gsd_is_numeric,  # True 表示 meta gsd 为有效数值, 仅作质量分析
        })

        if is_stop:
            reached_stop = True
            break

        # 10. 物理动力学更新 (Turn-then-Move) + 噪声
        # execute_grid_action 是训练生成、规则规划和评估闭环的唯一状态转移入口。
        # 不要在调用前手动更新 yaw, 否则训练轨迹会比评估闭环多转一次。
        # 10.1 Move: 动作先转向 atan2(grid_x, grid_y), 再沿更新后的朝向前进。
        # !!! PLAN §4.3 完整噪声模型: X_{t+1} = X_t + d·sin(θ) + noise_x,
        #                              Y_{t+1} = Y_t - d·cos(θ) + noise_y。
        #     - noise_dist (纵向): 沿当前飞行方向的步长控制误差 (PID / 加速度积分误差);
        #     - noise_lateral (横向): 与飞行方向正交的风偏 / IMU 漂移, 直接加到图像坐标系
        #       dx_img / dy_img 上 (各自独立采样, 不投影到飞行方向 — 否则等价于扩张纵向噪声)。
        noise_yaw_std = NOISE_YAW_STD if not in_terminal_mode else NOISE_YAW_STD * 0.25
        noise_dist_std = NOISE_DIST_STD if not in_terminal_mode else NOISE_DIST_STD * 0.25
        noise_lateral_std = NOISE_LATERAL_STD if not in_terminal_mode else NOISE_LATERAL_STD * 0.25
        current_x, current_y, current_yaw, move_dist_m = execute_grid_action(
            current_x, current_y, current_yaw, grid_x, grid_y, ppm,
            step_action_grid_size_m,
            noise_yaw_std=noise_yaw_std,
            noise_dist_std=noise_dist_std,
            noise_lateral_std=noise_lateral_std,
        )
        trajectory_points.append((current_x, current_y))
        heading_samples.append((current_x, current_y, current_yaw))

        # 10.3 更新 yaw / action 历史 (仅基于自身输出, 不用 GPS / 像素真值, 保证 sim-to-real 一致)
        yaw_history.append(current_yaw)
        action_history_full.append((grid_x, grid_y))
        history_actions.append((grid_x, grid_y))
        last_img_path = img_name        # 当前帧成为下一步的"历史图"

    if not reached_stop:
        return fail('max_steps_without_stop')

    if vis_output_dir:
        vis_path = os.path.join(
            vis_output_dir,
            f"{pid}_traj{int(traj_id):06d}_overlay.jpg",
        )
        overlay_path = save_original_trajectory_visualization(
            large_img,
            trajectory_points,
            heading_samples,
            target,
            landmark,
            vis_path,
            pid=pid,
            traj_id=traj_id,
        )
        if trajectory:
            trajectory[0]["trajectory_overlay"] = overlay_path

    return trajectory, None


def format_assistant_response(step, assistant_format="thought_action"):
    """Render the supervised assistant text for one transition."""
    if assistant_format not in ASSISTANT_FORMATS:
        raise ValueError(f"unknown assistant_format: {assistant_format}")
    action_text = f"Action: {step['action']}"
    if assistant_format == "action_only":
        return action_text
    return f"Thought: {step['thought']}\n{action_text}"


def trajectory_to_sharegpt(trajectory, instruction, assistant_format="thought_action"):
    """
    将单条 trajectory 转换为 LLaMA-Factory ShareGPT 多模态样本列表。
    每个 step 都是一条独立训练样本: (历史图 T-1, 当前图 T) -> Action。
    详见 PLAN §2.1 / §1.4 的单阶段 SFT prompt 骨架。

    !!! instruction 参数是"起点 snapshot 指令", 整条轨迹的所有 step 共用同一份。
        必须由调用方 (batch_generate) 在起点用 compute_instruction_context_at_start
        + build_instruction 一次性构造, 严禁每步重算 ——
        否则 spatial_word / count_in_view 会随姿态变化, 训练标签与同一任务漂移。
        评估端 run_evaluation_episode 同样必须在循环外构造一次, 保证 train-eval
        instruction 完全同分布 (PLAN §4.2.2 line 217)。
    !!! PLAN §4.1.1 修订: prompt 不暴露飞行高度 / 网格尺度 (DOTA 无真实高度概念,
        虚构标签无视觉验证手段; 多高度信号已隐式编码在像素中)。
    !!! PLAN §1.2 修订: 累计位移单位"格"(无量纲), 不再用"米"(米需要虚构 grid_size_m)。
    """
    samples = []
    for st in trajectory:
        hist_actions = st['history_actions']
        cum = st['cumulative_state']
        user_text = (
            f"<image>\n<image>\n"
            f"{instruction}\n"
            f"历史动作: {hist_actions}\n"
            f"累计位移: 向北{cum['north_grid']:.1f}格, 向东{cum['east_grid']:.1f}格, 已飞{cum['steps']}步\n"
            f"请输出下一步动作。"
        )
        gpt_text = format_assistant_response(st, assistant_format=assistant_format)
        samples.append({
            "conversations": [
                {"from": "human", "value": user_text},
                {"from": "gpt",   "value": gpt_text},
            ],
            "images": [st['image_hist'], st['image_cur']],
        })
    return samples


def trajectory_to_manifest_entries(trajectory, instruction, split_name, scenario, pid, traj_id,
                                   real_gsd=None):
    """Export a validation/debug sidecar without changing training JSONL.

    Args:
        real_gsd: 该 pid 视觉反推得到的 GSD (m/px), None 表示反推失败 (走 fallback 抖动)。
                  用于事后核对: 同一 pid 不同 trajectory 的 gsd_used 应聚集在 real_gsd ±20% 内。
    """
    entries = []
    for step_idx, st in enumerate(trajectory):
        entries.append({
            "split": split_name,
            "pid": pid,
            "traj_id": int(traj_id),
            "step": int(step_idx),
            "scenario": scenario,
            "phase": st.get("phase"),
            "landmark_relation_word": st.get("landmark_relation_word"),
            "landmark_relation_dist_m": st.get("landmark_relation_dist_m"),
            "seen_landmark_once": st.get("seen_landmark_once"),
            "seen_target_once": st.get("seen_target_once"),
            "landmark_visible": st.get("landmark_visible"),
            "target_visible": st.get("target_visible"),
            "terminal_zone": st.get("terminal_zone"),
            "should_stop": st.get("should_stop"),
            "lost_landmark_steps": st.get("lost_landmark_steps"),
            "recovery_start": st.get("recovery_start"),
            "instruction": instruction,
            "action": st.get("action"),
            "thought": st.get("thought"),
            "yaw": st.get("yaw"),
            "history_actions": st.get("history_actions"),
            "cumulative_state": st.get("cumulative_state"),
            "height_m": st.get("height_m"),
            "grid_size_m": st.get("grid_size_m"),
            "fov_m": st.get("fov_m"),
            "gsd_used": st.get("gsd_used"),
            "gsd_real": round(real_gsd, 4) if real_gsd is not None else None,
            "gsd_meta": st.get("gsd_meta"),
            "gsd_is_numeric": st.get("gsd_is_numeric"),
            "image_hist": st.get("image_hist"),
            "image_cur": st.get("image_cur"),
        })
    return entries


# 注: 早期方案的 to_stage1_sample 已废弃。理由见 PLAN §2.1 / §5.2:
# Stage 1 mask 历史/累计位移字段会教会模型"忽略这两个字段", 与最终目标直接冲突
# (反向 curriculum)。当前采用单阶段全量 SFT, 所有字段一次性参与训练。

def has_landmark_class(pid_index_entry):
    """判断该 P-id 的标注中是否含至少 1 个 L 类目标 (DOTA_CATEGORY_MAP 中 role='L' 或 'B')。
    用于场景 C/D 的图像池预筛选。"""
    targets, _, _ = parse_dota_annotation(
        pid_index_entry['label'], pid_index_entry['meta'])
    return any(cls_role(t['class']) in ('L', 'B') for t in targets)


def compute_scenario_ratios(train_pid_index, include_stop_scenario=True):
    """
    PLAN §4.3.1 动态兜底算法。
    输入: build_pid_to_path_index() 返回的 index dict。
    输出: 重平衡后的最终场景比例 dict {A,B,C,D,E: ratio}, 以及 L 类图 / T-only 图 P-id 列表。
    """
    l_class_pids = [pid for pid, entry in train_pid_index.items()
                    if has_landmark_class(entry)]
    l_pid_set = set(l_class_pids)
    t_only_pids  = [pid for pid in train_pid_index if pid not in l_pid_set]
    n_total = len(train_pid_index)
    max_cd_ratio = len(l_class_pids) / max(1, n_total)

    nominal = dict(NOMINAL_SCENARIO_RATIOS)
    if not include_stop_scenario:
        stop_ratio = nominal.pop('E', 0.0)
        nominal['B'] = nominal.get('B', 0.0) + stop_ratio
        nominal['E'] = 0.0
    cd_target = nominal['C'] + nominal['D']      # 0.40

    if cd_target > max_cd_ratio:
        shrink = max_cd_ratio / cd_target        # 例如 0.33 / 0.40 = 0.825
        final = {
            'A': nominal['A'],
            'B': nominal['B'] + (cd_target - max_cd_ratio),  # 吸收 C/D 缺额
            'C': nominal['C'] * shrink,
            'D': nominal['D'] * shrink,
            'E': nominal['E'],
        }
    else:
        final = dict(nominal)

    print(f"[scenario] L 类图: {len(l_class_pids)}/{n_total} = {max_cd_ratio:.2%}")
    print(f"[scenario] 名义 C+D = {cd_target:.0%}, 实际可达 {max_cd_ratio:.2%}")
    print(f"[scenario] 最终采样比例: {final}")
    return final, l_class_pids, t_only_pids


def compute_expected_height_ratios(pid_index, scenario_ratios, gsd_cache=None,
                                   l_class_pids=None, t_only_pids=None):
    """按 split 的 pid 池估计理论高度分布, 供校验阶段作为目标基线。

    估计方法与实际生成顺序对齐:
      1. 先按场景比例采样场景;
      2. 再从对应 pid 池均匀采样 pid;
      3. 对该 pid 的 allowed_heights 按 HEIGHT_BIN_RATIOS 重新归一化采样高度。
    """
    all_pids = list(pid_index.keys())
    l_class_pids = list(l_class_pids or [])
    t_only_pids = list(t_only_pids or [])
    scenario_pools = {
        'A': all_pids,
        'B': all_pids,
        'C': l_class_pids,
        'D': l_class_pids,
        'E': all_pids,
    }
    accum = {h: 0.0 for h in HEIGHT_BINS}

    for scenario, scenario_ratio in scenario_ratios.items():
        if scenario_ratio <= 0.0:
            continue
        pool = scenario_pools.get(scenario, [])
        if not pool:
            continue
        pid_weight = scenario_ratio / len(pool)
        for pid in pool:
            real_gsd = gsd_cache.get(pid) if gsd_cache is not None else None
            allowed_heights = compatible_height_bins(
                real_gsd,
                dataset_name=pid_index[pid].get('dataset'),
            )
            total_weight = sum(HEIGHT_BIN_RATIOS[h] for h in allowed_heights)
            if total_weight <= 0.0:
                continue
            for h in allowed_heights:
                accum[h] += pid_weight * (HEIGHT_BIN_RATIOS[h] / total_weight)

    total = sum(accum.values())
    if total <= 0.0:
        return dict(HEIGHT_BIN_RATIOS)
    return {h: accum[h] / total for h in HEIGHT_BINS}


def choose_weighted_nearby_landmark(target, candidates, ppm, top_k=3):
    """A/B 场景优先选更近的 landmark, 但保留少量近邻多样性。"""
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda obj: object_distance_m(obj, target, ppm),
    )
    shortlist = ranked[:max(1, min(int(top_k), len(ranked)))]
    weights = []
    for obj in shortlist:
        dist_m = object_distance_m(obj, target, ppm)
        weights.append(1.0 / max(1.0, dist_m))
    return random.choices(shortlist, weights=weights, k=1)[0]


def dispatch_scenario(final_ratios):
    """按权重 dict 采样一个场景标签。权重可为比例, 也可为剩余 quota。"""
    positive_items = [(k, v) for k, v in final_ratios.items() if v > 0]
    if not positive_items:
        return None
    keys = [k for k, _ in positive_items]
    weights = [v for _, v in positive_items]
    return random.choices(keys, weights=weights, k=1)[0]


def pick_pid_for_scenario(scenario, l_class_pids, t_only_pids,
                          cd_pid_usage=None, cd_pid_cap=None):
    """根据场景标签从对应图像池采样 P-id。
    - A/B/E: 全集 (l_class_pids + t_only_pids)
    - C/D:   仅 l_class_pids, 且受"每张图最多 N 条 C/D 轨迹"上限约束
    """
    if scenario in ('C', 'D'):
        if not l_class_pids:
            return None             # 该数据集完全没有 L 类图, 跳过这个场景
        if cd_pid_usage is None or cd_pid_cap is None:
            return random.choice(l_class_pids)
        available = [pid for pid in l_class_pids
                     if cd_pid_usage.get(pid, 0) < cd_pid_cap]
        if not available:
            return None
        min_usage = min(cd_pid_usage.get(pid, 0) for pid in available)
        candidates = [pid for pid in available
                      if cd_pid_usage.get(pid, 0) == min_usage]
        return random.choice(candidates)
    pool = l_class_pids + t_only_pids
    return random.choice(pool) if pool else None


def sample_start_distance(scenario, height_m, grid_size_m=None,
                          min_start_distance_m=MIN_START_DISTANCE_M):
    """按场景 + 高度档位采样起点到目标的距离 (米). PLAN §4.1.1 起点距离档位表。"""
    if scenario == 'E':
        if grid_size_m is None:
            raise ValueError("scenario 'E' 需要显式传入 grid_size_m, 以对齐统一 STOP 阈值。")
        # E 场景保留为近终点样本, 但不再允许起点直接落在 STOP 半径内, 否则会系统性生成 0 步轨迹。
        # 下界至少 1 步终端动作, 上界略大于 STOP 半径, 让模型学会"接近后再停"。
        eps_m = stop_eps_m(grid_size_m)
        terminal_step_m = TERMINAL_ACTION_GRID_SIZE_M
        lo = max(eps_m + 1e-3, terminal_step_m)
        hi = max(lo, eps_m + terminal_step_m * MIN_TRAJECTORY_MOVES)
        return random.uniform(lo, hi)

    scale = height_m / 25.0       # 25m=1.0, 50m=2.0, 75m=3.0
    ranges = {
        'A': (5,  15),    'B': (30, 60),   'C': (70, 120),
        'D': (30, 80),
    }
    lo, hi = ranges[scenario]
    lo = max(lo, float(min_start_distance_m) / max(1e-6, scale))
    hi = max(hi, lo)
    return random.uniform(lo * scale, hi * scale)


def sample_landmark_arrival_distance(height_m, fov_m, min_start_distance_m=MIN_START_DISTANCE_M):
    """GPS arrival error around the landmark for local visual search episodes."""
    lower = max(8.0, min(float(min_start_distance_m), fov_m * 0.20))
    upper = max(lower + 1.0, min(fov_m * 0.45, height_m * 0.90))
    return random.uniform(lower, upper)


def choose_compensation_scenario(failed_scenario, remaining_counts):
    """
    配额无法满足时的补偿目标。
    PLAN / IMPLEMENTATION 明确要求 C/D 缺额优先补给 B; 其他场景若偶发失败, 也优先回填到
    当前仍欠配额的易生成场景。
    """
    preferred_map = {
        'A': ['B', 'E', 'C', 'D'],
        'B': ['A', 'E', 'C', 'D'],
        'C': ['B', 'A', 'E', 'D'],
        'D': ['B', 'A', 'E', 'C'],
        'E': ['B', 'A', 'C', 'D'],
    }
    preferred = preferred_map.get(failed_scenario, ['B', 'A', 'E', 'C', 'D'])
    for scenario in preferred:
        if scenario != failed_scenario and (remaining_counts.get(scenario, 0) > 0 or scenario == 'B'):
            return scenario
    return None


def can_compensate_scenario(failed_scenario, fallback_scenario, target_counts, actual_counts, compensation_counts):
    """限制单个 fallback 场景吞掉过多失败配额。"""
    if fallback_scenario is None:
        return False
    baseline = int(target_counts.get(fallback_scenario, 0))
    if baseline <= 0:
        return True
    cap = max(1, int(math.ceil(sum(target_counts.values()) * SCENARIO_COMPENSATION_CAP_RATIO)))
    current = int(compensation_counts.get(fallback_scenario, 0))
    overflow = max(0, int(actual_counts.get(fallback_scenario, 0)) - baseline)
    return (current + overflow) < cap


def sample_episode(entry, scenario, height_m, grid_size_m, fov_m, ppm,
                   max_attempts=8,
                   landmark_max_distance_m=LANDMARK_MAX_DISTANCE_M,
                   landmark_nearby_distance_m=LANDMARK_NEARBY_DISTANCE_M,
                   min_start_distance_m=MIN_START_DISTANCE_M):
    """
    给定 P-id 三件套 + 场景 + 高度档位 + ppm, 采样单条 episode 的所有起点参数。
    返回 (episode dict, failure_reason)。
    - 成功: (episode, None)
    - 失败: (None, reason)

    !!! ppm 必须由调用方 (batch_generate) 在轨迹入口调一次 pixel_per_meter(gsd_jitter=True)
        统一传入, 与同轨迹后续 compute_instruction_context_at_start / generate_trajectory
        全程同源。严禁此函数内部再次 pixel_per_meter() (历史 bug: 起点像素位置与实际裁剪
        视野尺度不一致, 场景距离档位失效)。

    采样流程 (PLAN §4.2.2 + §4.3):
      1. 解析整张图所有标注 (all_targets), 缓存供 instruction snapshot 复用
      2. 按场景规则选 target (限定 T/B 类: 大型场地作为目标语义弱, 排除)
      3. 按场景规则选 landmark:
         - C/D 强制有 L 类, landmark 与 target 在配置距离内，target 相对 landmark 的方向写入任务
         - A/B 80m 内有 L 类则用, 否则 None (走 spatial_word 兜底)
         - E   landmark = None (起点即终点, 不需要地标参照)
      4. 按 scenario+height 采样 |起点-目标| 距离, 随机方位角 phi 摆放起点
      5. 边界校验: 起点 + 旋转裁剪外接圆必须在图内 (否则 safe_crop 大量黑边)
      6. 随机起点 yaw, 避免目标永远在画面上方 (PLAN §4.3)
      7. 若起点进入 SEARCH_LANDMARK, 需满足“地标在有限步内可搜到”的可达性约束
    """
    targets, _, _ = parse_dota_annotation(entry['label'], entry['meta'])
    if not targets:
        return None, 'no_targets'

    # ppm 由调用方传入 (见 docstring 调用契约), 内部不再 pixel_per_meter()

    # 缓存图像尺寸 (只 imread 一次)
    img_shape = get_cached_image_shape(entry['img'])
    if img_shape is None:
        return None, 'image_shape_unavailable'
    H, W = img_shape
    crop_half_px = (fov_m * ppm * 1.414) / 2 + 4   # 旋转外接圆半径 + 安全余量
    max_guided_landmark_distance_m = max(0.0, float(landmark_max_distance_m))

    # 1. target 候选: 限定 T/B 类
    target_pool = [t for t in targets if cls_role(t['class']) in ('T', 'B')]
    if not target_pool:
        return None, 'no_target_pool'

    # 2. landmark 候选 (随 target 选定后再过滤)
    last_reason = 'episode_sampling_exhausted'
    for _ in range(max_attempts):
        target = random.choice(target_pool)

        other = [t for t in targets if t is not target]
        l_pool = [t for t in other if cls_role(t['class']) in ('L', 'B') and t['class'] != target['class']]

        landmark = None
        if scenario in ('C', 'D'):
            if not l_pool:
                last_reason = 'missing_landmark_pool'
                continue                 # 换 target 重试
            if max_guided_landmark_distance_m <= 0.0:
                last_reason = 'guided_distance_disabled'
                continue
            guided_pool = [
                t for t in l_pool
                if object_distance_m(t, target, ppm) <= max_guided_landmark_distance_m
            ]
            if not guided_pool:
                last_reason = 'guided_landmark_too_far'
                continue
            l_pool_sorted = sorted(
                guided_pool,
                key=lambda t: object_distance_m(t, target, ppm),
            )
            landmark = l_pool_sorted[0]
        elif scenario in ('A', 'B'):
            nearby = [t for t in l_pool
                      if object_distance_m(t, target, ppm) <= landmark_nearby_distance_m]
            landmark = choose_weighted_nearby_landmark(target, nearby, ppm) if nearby else None
        # scenario 'E': landmark 保持 None

        # 3. 起点距离 + 方位角。
        # C/D 模拟真实部署: 上游 GPS 已到地标附近, 大模型只负责局部视觉搜索目标。
        if landmark is not None and scenario in ('C', 'D'):
            dist_m = sample_landmark_arrival_distance(
                height_m,
                fov_m,
                min_start_distance_m=min_start_distance_m,
            )
            anchor = landmark
        else:
            dist_m = sample_start_distance(
                scenario,
                height_m,
                grid_size_m=grid_size_m,
                min_start_distance_m=min_start_distance_m,
            )
            anchor = target
        dist_px = dist_m * ppm
        phi = random.uniform(0, 2 * math.pi)
        start_x = anchor['cx'] + dist_px * math.cos(phi)
        start_y = anchor['cy'] + dist_px * math.sin(phi)

        # 4. 边界校验
        if not (crop_half_px < start_x < W - crop_half_px
                and crop_half_px < start_y < H - crop_half_px):
            last_reason = 'start_out_of_bounds'
            continue                     # 起点越界, 换方位/距离重试

        # 5. 起点 yaw: 有地标任务模拟上游导航已大致朝向地标; 无地标任务保留随机朝向。
        if landmark is not None:
            start_yaw = yaw_towards_point(start_x, start_y, landmark['cx'], landmark['cy'])
            gps_heading_noise = 18.0 if scenario in ('C', 'D') else 10.0
            start_yaw = (start_yaw + random.gauss(0, gps_heading_noise) + 360.0) % 360.0
        else:
            start_yaw = random.uniform(0, 360)

        # 5.1 可达性约束: 若起点 target/landmark 都不在视野内, 则要求地标可在有限步内搜到。
        if landmark is not None:
            target_visible = object_in_fov(target, start_x, start_y, start_yaw, ppm, fov_m)
            landmark_visible = object_in_fov(landmark, start_x, start_y, start_yaw, ppm, fov_m)
            if scenario in ('C', 'D') and (not landmark_visible):
                last_reason = 'gps_arrival_without_landmark_visible'
                continue
            if (not target_visible) and (not landmark_visible):
                if not landmark_reachable_within_search_steps(
                    start_x,
                    start_y,
                    start_yaw,
                    landmark,
                    img_shape,
                    ppm,
                    fov_m,
                ):
                    last_reason = 'landmark_not_reachable'
                    continue

        # 6. D 场景退化防御: 若 target 已在起点视野内, 验证确实有 2+ 个同类目标。
        # 地标锚定局部搜索下, target 常常初始不可见, 此时不能用起点 target 视野约束误杀样本。
        if scenario == 'D' and target_visible:
            count_in_view, _ = compute_instruction_context_at_start(
                targets, target, start_x, start_y, start_yaw,
                ppm=ppm, fov_m=fov_m, grid_size_m=grid_size_m
            )
            if count_in_view < 2:
                last_reason = 'd_insufficient_same_class'
                continue

        return {
            'target': target,
            'landmark': landmark,
            'start_x': start_x,
            'start_y': start_y,
            'start_yaw': start_yaw,
            'all_targets': targets,
        }, None

    return None, last_reason             # max_attempts 内未成功


def generate_single_trajectory(train_pid_index, scenario, output_dir, traj_id,
                               split_name='train', gsd_cache=None,
                               vis_output_dir=None,
                               save_vis=False,
                               max_turn_deg_per_step=MAX_TURN_DEG_PER_STEP,
                               action_grid_size_m=ACTION_GRID_SIZE_M,
                               terminal_action_grid_size_m=TERMINAL_ACTION_GRID_SIZE_M,
                               landmark_max_distance_m=LANDMARK_MAX_DISTANCE_M,
                               landmark_nearby_distance_m=LANDMARK_NEARBY_DISTANCE_M,
                               min_start_distance_m=MIN_START_DISTANCE_M,
                               l_class_pids=None, t_only_pids=None,
                               cd_pid_usage=None, cd_pid_cap=0,
                               max_pid_retries=5,
                               requested_height_m=None):
    """Worker helper that generates one trajectory candidate end-to-end."""
    ep, pid, ppm, real_gsd_used = None, None, None, None
    height_m, grid_size_m, fov_m = None, None, None
    failure_reason = 'pid_sampling_exhausted'
    height_fallback_used = False

    for _ in range(max_pid_retries):
        pid_try = pick_pid_for_scenario(
            scenario,
            l_class_pids or [],
            t_only_pids or [],
            cd_pid_usage=cd_pid_usage or {},
            cd_pid_cap=cd_pid_cap,
        )
        if pid_try is None:
            break

        real_gsd_try = gsd_cache.get(pid_try) if gsd_cache is not None else None
        allowed_h = compatible_height_bins(
            real_gsd_try,
            dataset_name=train_pid_index[pid_try].get('dataset'),
        )
        height_try = requested_height_m
        if requested_height_m is not None and requested_height_m not in allowed_h:
            height_fallback_used = True
            height_try = None
        height_try, grid_size_try, fov_try = sample_height_bin(
            allowed_heights=allowed_h,
            preferred_height=height_try,
        )
        ppm_try = pixel_per_meter(real_gsd=real_gsd_try, jitter=True)
        ep_try, ep_failure = sample_episode(
            train_pid_index[pid_try],
            scenario,
            height_try,
            grid_size_try,
            fov_try,
            ppm=ppm_try,
            landmark_max_distance_m=landmark_max_distance_m,
            landmark_nearby_distance_m=landmark_nearby_distance_m,
            min_start_distance_m=min_start_distance_m,
        )
        if ep_try is not None:
            ep, pid, ppm, real_gsd_used = ep_try, pid_try, ppm_try, real_gsd_try
            height_m, grid_size_m, fov_m = height_try, grid_size_try, fov_try
            break
        failure_reason = ep_failure or 'episode_sampling_failed'

    if ep is None:
        return {
            'ok': False,
            'scenario': scenario,
            'requested_height_m': requested_height_m,
            'failure_reason': failure_reason,
            'height_fallback_used': height_fallback_used,
        }

    count_in_view, spatial_word = compute_instruction_context_at_start(
        ep['all_targets'], ep['target'],
        ep['start_x'], ep['start_y'], ep['start_yaw'],
        ppm=ppm, fov_m=fov_m, grid_size_m=grid_size_m,
    )
    instruction = build_instruction(
        ep['target'],
        landmark=ep['landmark'],
        spatial_word=spatial_word,
        target_count_in_view=count_in_view,
        ppm=ppm,
    )

    entry = train_pid_index[pid]
    traj, traj_failure = generate_trajectory(
        entry['img'], entry['label'], entry['meta'],
        ep['target'], ep['landmark'],
        ep['start_x'], ep['start_y'], ep['start_yaw'],
        ppm=ppm, grid_size_m=grid_size_m, fov_m=fov_m,
        output_dir=output_dir, pid=pid, traj_id=traj_id,
        height_m=height_m,
        vis_output_dir=vis_output_dir if save_vis else None,
        max_turn_deg_per_step=max_turn_deg_per_step,
        action_grid_size_m=action_grid_size_m,
        terminal_action_grid_size_m=terminal_action_grid_size_m,
    )
    if not traj:
        return {
            'ok': False,
            'scenario': scenario,
            'requested_height_m': requested_height_m,
            'height_m': height_m,
            'pid': pid,
            'failure_reason': traj_failure or 'trajectory_generation_failed',
            'height_fallback_used': height_fallback_used,
        }

    return {
        'ok': True,
        'traj': traj,
        'instruction': instruction,
        'scenario': scenario,
        'pid': pid,
        'traj_id': traj_id,
        'height_m': height_m,
        'real_gsd_used': real_gsd_used,
        'requested_height_m': requested_height_m,
        'height_fallback_used': height_fallback_used,
    }


def batch_generate(train_pid_index, n_trajectories, output_dir,
                   max_pid_retries=5, max_slot_attempts=3,
                   return_manifest=False, split_name='train',
                   gsd_cache=None, vis_output_dir=None,
                   max_turn_deg_per_step=MAX_TURN_DEG_PER_STEP,
                   forward_bias_min_ratio=FORWARD_BIAS_MIN_RATIO,
                   rear_turn_slowdown_ratio=REAR_TURN_SLOWDOWN_RATIO,
                   action_grid_size_m=ACTION_GRID_SIZE_M,
                   terminal_action_grid_size_m=TERMINAL_ACTION_GRID_SIZE_M,
                   landmark_max_distance_m=LANDMARK_MAX_DISTANCE_M,
                   landmark_nearby_distance_m=LANDMARK_NEARBY_DISTANCE_M,
                   min_start_distance_m=MIN_START_DISTANCE_M,
                   max_workers=DEFAULT_MAX_WORKERS,
                   include_stop_scenario=True,
                   assistant_format="thought_action",
                   max_original_traj_vis=0,
                   progress_interval=DEFAULT_PROGRESS_INTERVAL):
    """
    批量数据生成主驱动 (PLAN §4.1.1 + §4.3.1 + §4.2.2 综合):
      1. 高度档位采样 → 取 grid_size / fov
      2. 场景比例采样 (已按 L 类图占比动态兜底)
      3. pick_pid + sample_episode 取 target/landmark/start_xy/start_yaw (有重试)
      4. !!! 起点 snapshot 构造 instruction (与 evaluate.py 完全同源, 杜绝 OOD)
      5. 调 generate_trajectory(), trajectory_to_sharegpt(), 累积 JSONL
      6. 全部完成后输出实际场景占比统计表

    !!! 关键: instruction 在循环外只调一次 compute_instruction_context_at_start +
        build_instruction, 然后传给 trajectory_to_sharegpt 作为整条轨迹的固定指令。
        评估端 run_evaluation_episode 同样在循环外构造一次, 两端同源 (PLAN §4.2.2)。

    Args:
        gsd_cache: {pid: real_gsd or None}, 由 build_pid_gsd_index 预计算。
                   - 传入: 每条轨迹的 ppm 围绕该 pid 的视觉反推 GSD ±20% 抖动 (推荐, 解决尺度漂移);
                   - None: 退化到旧行为, 所有轨迹用 [0.35, 0.7] 大范围抖动 (兼容旧调用)。
    """
    final_ratios_raw, l_class_pids, t_only_pids = compute_scenario_ratios(
        train_pid_index,
        include_stop_scenario=include_stop_scenario,
    )
    scenario_ratio_floors = MIN_SCENARIO_RATIO_FLOORS if include_stop_scenario else {}
    final_ratios = apply_ratio_floors(final_ratios_raw, scenario_ratio_floors)
    target_counts = allocate_count_by_ratio(n_trajectories, final_ratios)
    effective_target_counts = dict(target_counts)
    height_target_ratios_raw = compute_expected_height_ratios(
        train_pid_index,
        final_ratios,
        gsd_cache=gsd_cache,
        l_class_pids=l_class_pids,
        t_only_pids=t_only_pids,
    )
    height_target_ratios = apply_ratio_floors(height_target_ratios_raw, MIN_HEIGHT_RATIO_FLOORS)
    target_height_counts = allocate_count_by_ratio(n_trajectories, height_target_ratios)
    actual_counts = {k: 0 for k in SCENARIO_KEYS}
    height_counts = {h: 0 for h in HEIGHT_BINS}
    samples_out = []
    manifest_out = []
    compensation_edges = {}
    compensation_received = {k: 0 for k in SCENARIO_KEYS}
    failure_reason_counts = defaultdict(int)
    failure_combo_counts = defaultdict(int)
    requested_height_fallbacks = defaultdict(int)

    total_cd_target = target_counts['C'] + target_counts['D']
    cd_pid_usage = {pid: 0 for pid in l_class_pids}
    cd_pid_cap = 0
    if l_class_pids and total_cd_target > 0:
        cd_pid_cap = int(math.ceil(total_cd_target / len(l_class_pids)))

    print(f"[scenario] 目标轨迹配额: {target_counts}")
    if scenario_ratio_floors:
        print(f"[scenario] 保底比例: {scenario_ratio_floors}")
    print(f"[height] 目标轨迹配额: {target_height_counts}")
    print(f"[height] 保底比例: {MIN_HEIGHT_RATIO_FLOORS}")
    if cd_pid_cap > 0:
        print(f"[scenario] C/D 每张 L 类图最多回采 {cd_pid_cap} 条轨迹")
    if not include_stop_scenario:
        print("[scenario] 已禁用 E 场景，避免起点贴近 target。")
    if gsd_cache is None:
        print("[gsd] 未传入 gsd_cache, 全部轨迹用 [0.35, 0.7] 大范围抖动 (兼容旧行为)")
    if vis_output_dir:
        print(f"[vis] 最多保存 {max_original_traj_vis} 条原图轨迹叠加图到 {vis_output_dir}")
    print(f"[progress] split={split_name}, target={n_trajectories}, workers={max_workers}")

    # !!! ppm 在 pid 选定后才知道 (因为视觉反推 GSD 是 per-pid 的),
    #     与旧版"先抽 ppm 再选 pid"的顺序相反, 但仍保证: 一条 trajectory 全程同一 ppm。

    max_workers = max(1, int(max_workers))
    next_traj_id = 0
    global_failures = 0
    max_global_failures = max(20, n_trajectories * max_slot_attempts * 2)
    progress_interval = max(1, int(progress_interval))
    max_original_traj_vis = max(0, int(max_original_traj_vis or 0))

    def remaining_scenario_counts():
        return {
            s: max(0, effective_target_counts[s] - actual_counts[s])
            for s in SCENARIO_KEYS
        }

    def remaining_height_counts():
        return {
            h: max(0, target_height_counts[h] - height_counts[h])
            for h in HEIGHT_BINS
        }

    def submit_one(executor, scenario_name, requested_height_m):
        nonlocal next_traj_id
        save_vis = bool(vis_output_dir) and next_traj_id < max_original_traj_vis
        future = executor.submit(
            generate_single_trajectory,
            train_pid_index,
            scenario_name,
            output_dir,
            next_traj_id,
            split_name=split_name,
            gsd_cache=gsd_cache,
            vis_output_dir=vis_output_dir,
            save_vis=save_vis,
            max_turn_deg_per_step=max_turn_deg_per_step,
            action_grid_size_m=action_grid_size_m,
            terminal_action_grid_size_m=terminal_action_grid_size_m,
            landmark_max_distance_m=landmark_max_distance_m,
            landmark_nearby_distance_m=landmark_nearby_distance_m,
            min_start_distance_m=min_start_distance_m,
            l_class_pids=l_class_pids,
            t_only_pids=t_only_pids,
            cd_pid_usage=cd_pid_usage,
            cd_pid_cap=cd_pid_cap,
            max_pid_retries=max_pid_retries,
            requested_height_m=requested_height_m,
        )
        request = {
            'scenario': scenario_name,
            'requested_height_m': requested_height_m,
            'traj_id': next_traj_id,
        }
        next_traj_id += 1
        return future, request

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {}

        while len(pending) < max_workers and sum(actual_counts.values()) + len(pending) < n_trajectories:
            remaining_counts = remaining_scenario_counts()
            if sum(remaining_counts.values()) <= 0:
                break
            scenario = dispatch_scenario(remaining_counts)
            if scenario is None:
                break
            height_request = choose_weighted_remaining_count(remaining_height_counts())
            future, request = submit_one(executor, scenario, height_request)
            pending[future] = request

        while pending:
            future = next(as_completed(pending))
            request = pending.pop(future)
            scenario = request['scenario']
            requested_height_m = request['requested_height_m']
            result = future.result()

            if not result.get('ok'):
                global_failures += 1
                reason = result.get('failure_reason') or 'unknown_failure'
                failure_reason_counts[reason] += 1
                failure_combo_counts[(scenario, requested_height_m, reason)] += 1
                if result.get('height_fallback_used') and requested_height_m is not None:
                    requested_height_fallbacks[requested_height_m] += 1
                remaining_counts = remaining_scenario_counts()
                fallback = choose_compensation_scenario(scenario, remaining_counts)
                if can_compensate_scenario(
                    scenario,
                    fallback,
                    target_counts=target_counts,
                    actual_counts=actual_counts,
                    compensation_counts=compensation_received,
                ):
                    effective_target_counts[scenario] -= 1
                    effective_target_counts[fallback] += 1
                    key = f"{scenario}->{fallback}"
                    compensation_edges[key] = compensation_edges.get(key, 0) + 1
                    compensation_received[fallback] = compensation_received.get(fallback, 0) + 1
                if global_failures >= max_global_failures:
                    print("[warn] 连续失败次数过多, 提前结束本轮生成。")
                    break
            else:
                global_failures = 0
                traj = result['traj']
                instruction = result['instruction']
                scenario = result['scenario']
                pid = result['pid']
                traj_id = result['traj_id']
                height_m = result['height_m']
                real_gsd_used = result['real_gsd_used']

                samples_out.extend(
                    trajectory_to_sharegpt(
                        traj,
                        instruction,
                        assistant_format=assistant_format,
                    )
                )
                if return_manifest:
                    manifest_out.extend(
                        trajectory_to_manifest_entries(
                            traj,
                            instruction=instruction,
                            split_name=split_name,
                            scenario=scenario,
                            pid=pid,
                            traj_id=traj_id,
                            real_gsd=real_gsd_used,
                        )
                    )
                actual_counts[scenario] += 1
                height_counts[height_m] += 1
                if scenario in ('C', 'D'):
                    cd_pid_usage[pid] = cd_pid_usage.get(pid, 0) + 1
                if result.get('height_fallback_used') and requested_height_m is not None:
                    requested_height_fallbacks[requested_height_m] += 1

                completed = sum(actual_counts.values())
                if completed == 1 or completed % progress_interval == 0 or completed >= n_trajectories:
                    print(f"[progress] split={split_name} {completed}/{n_trajectories} "
                          f"({completed / max(1, n_trajectories):.1%})")

            if sum(actual_counts.values()) >= n_trajectories:
                break

            remaining_counts = remaining_scenario_counts()
            if sum(remaining_counts.values()) > 0:
                scenario = dispatch_scenario(remaining_counts)
                if scenario is not None:
                    height_request = choose_weighted_remaining_count(remaining_height_counts())
                    future, request = submit_one(executor, scenario, height_request)
                    pending[future] = request

    # 输出实际占比统计 (PLAN §4.3.1 Step 4)
    total = sum(actual_counts.values())
    effective_total = max(1, sum(effective_target_counts.values()))
    print("\n=== 场景实际占比统计 ===")
    for s in SCENARIO_KEYS:
        actual = actual_counts[s] / max(1, total)
        target = effective_target_counts[s] / effective_total
        delta = actual - target
        flag = " ⚠" if abs(delta) > SCENARIO_DEVIATION_TOL else ""
        print(f"  {s}: {actual:.1%} (目标 {target:.1%}, 偏差 {delta:+.1%}){flag}")
    if compensation_edges:
        print(f"[scenario] 配额补偿: {compensation_edges}")
    if total < n_trajectories:
        print(f"[warn] 仅成功生成 {total}/{n_trajectories} 条轨迹, 未完全达到目标配额。")
    print("\n=== 高度档位实际占比 ===")
    for h in HEIGHT_BINS:
        actual = height_counts[h] / max(1, total)
        target = target_height_counts[h] / max(1, n_trajectories)
        print(f"  {h}m: {actual:.1%} (目标 {target:.1%})")
    if requested_height_fallbacks:
        print(f"[height] 请求高度不兼容回退次数: {dict(sorted(requested_height_fallbacks.items()))}")
    if failure_reason_counts:
        print("\n=== 失败原因统计 ===")
        for reason, count in sorted(failure_reason_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {reason}: {count}")
    if failure_combo_counts:
        print("\n=== 场景/高度失败 Top10 ===")
        top_failures = sorted(
            failure_combo_counts.items(),
            key=lambda item: (-item[1], str(item[0])),
        )[:10]
        for (scenario_name, height_m, reason), count in top_failures:
            print(f"  {scenario_name}@{height_m}m -> {reason}: {count}")

    if return_manifest:
        return samples_out, manifest_out
    return samples_out


def generate_train_eval_jsonl(pid_index, output_dir, n_trajectories,
                              train_ratio=DEFAULT_TRAIN_RATIO,
                              split_seed=DEFAULT_SPLIT_SEED,
                              dataset_info_path=None,
                              max_original_traj_vis=0,
                              max_turn_deg_per_step=MAX_TURN_DEG_PER_STEP,
                              forward_bias_min_ratio=FORWARD_BIAS_MIN_RATIO,
                              rear_turn_slowdown_ratio=REAR_TURN_SLOWDOWN_RATIO,
                              action_grid_size_m=ACTION_GRID_SIZE_M,
                              terminal_action_grid_size_m=TERMINAL_ACTION_GRID_SIZE_M,
                              landmark_max_distance_m=LANDMARK_MAX_DISTANCE_M,
                              landmark_nearby_distance_m=LANDMARK_NEARBY_DISTANCE_M,
                              min_start_distance_m=MIN_START_DISTANCE_M,
                              max_workers=DEFAULT_MAX_WORKERS,
                              include_stop_scenario=True,
                              assistant_format="thought_action",
                              progress_interval=DEFAULT_PROGRESS_INTERVAL):
    """
    按 DOTA 图像 ID 先切分, 再分别导出 train.jsonl / eval.jsonl。
    同时写出 image_ids_train.txt / image_ids_val.txt, 防止 train/eval 共享同一张大图。
    """
    os.makedirs(output_dir, exist_ok=True)

    train_pids, val_pids = split_pid_lists(pid_index, train_ratio=train_ratio, seed=split_seed)
    train_pid_index = filter_pid_index(pid_index, train_pids)
    val_pid_index = filter_pid_index(pid_index, val_pids)

    train_ids_path = os.path.join(output_dir, 'image_ids_train.txt')
    val_ids_path = os.path.join(output_dir, 'image_ids_val.txt')
    write_pid_list(train_pids, train_ids_path)
    write_pid_list(val_pids, val_ids_path)

    n_total_images = max(1, len(train_pids) + len(val_pids))
    n_train_traj = int(round(n_trajectories * len(train_pids) / n_total_images))
    n_train_traj = max(0, min(n_trajectories, n_train_traj))
    if val_pids and n_trajectories > 1:
        n_train_traj = min(n_trajectories - 1, max(1, n_train_traj))
    n_eval_traj = n_trajectories - n_train_traj

    print(f"[split] 图像 ID 划分: train={len(train_pids)}, eval={len(val_pids)}, seed={split_seed}")
    print(f"[split] 轨迹配额: train={n_train_traj}, eval={n_eval_traj}")
    print(f"[split] 已写出: {train_ids_path} / {val_ids_path}")

    # === 视觉反推 GSD 缓存 (PLAN §4.1 二次修订) ===
    # 启动时对所有 pid 扫描一次, 每张图基于 OBB 标注 + PRIOR_SIZE_M 反推真实 GSD,
    # 后续 batch_generate 中每条轨迹用该 pid 的 real_gsd 做小幅抖动 (±20%) 生成 ppm。
    # 比起旧的固定 GSD=0.5 + 大范围抖动 [0.35, 0.7], 极大缓解"有时截太小、有时截太大、
    # resize 后模糊"的视觉退化。反推失败 (~10-15%) 的 pid 仍走 fallback 大范围抖动。
    print("[gsd] 正在视觉反推每张图的 GSD ...")
    gsd_cache = build_pid_gsd_index(pid_index)

    train_scenario_ratios_raw, train_l_class_pids, train_t_only_pids = compute_scenario_ratios(
        train_pid_index,
        include_stop_scenario=include_stop_scenario,
    )
    eval_scenario_ratios_raw, eval_l_class_pids, eval_t_only_pids = compute_scenario_ratios(
        val_pid_index,
        include_stop_scenario=include_stop_scenario,
    )
    scenario_ratio_floors = MIN_SCENARIO_RATIO_FLOORS if include_stop_scenario else {}
    train_scenario_ratios = apply_ratio_floors(train_scenario_ratios_raw, scenario_ratio_floors)
    eval_scenario_ratios = apply_ratio_floors(eval_scenario_ratios_raw, scenario_ratio_floors)
    train_height_target_ratios = apply_ratio_floors(
        compute_expected_height_ratios(
            train_pid_index,
            train_scenario_ratios,
            gsd_cache=gsd_cache,
            l_class_pids=train_l_class_pids,
            t_only_pids=train_t_only_pids,
        ),
        MIN_HEIGHT_RATIO_FLOORS,
    )
    eval_height_target_ratios = apply_ratio_floors(
        compute_expected_height_ratios(
            val_pid_index,
            eval_scenario_ratios,
            gsd_cache=gsd_cache,
            l_class_pids=eval_l_class_pids,
            t_only_pids=eval_t_only_pids,
        ),
        MIN_HEIGHT_RATIO_FLOORS,
    )
    generation_config = {
        'include_stop_scenario': bool(include_stop_scenario),
        'assistant_format': str(assistant_format),
        'landmark_max_distance_m': float(landmark_max_distance_m),
        'landmark_nearby_distance_m': float(landmark_nearby_distance_m),
        'min_start_distance_m': float(min_start_distance_m),
        'height_bin_ratios_nominal': dict(HEIGHT_BIN_RATIOS),
        'scenario_ratio_floors': dict(scenario_ratio_floors),
        'height_ratio_floors': dict(MIN_HEIGHT_RATIO_FLOORS),
        'splits': {
            'train': {
                'pid_count': len(train_pid_index),
                'l_class_pid_count': len(train_l_class_pids),
                'scenario_target_ratios': train_scenario_ratios,
                'height_target_ratios': train_height_target_ratios,
            },
            'eval': {
                'pid_count': len(val_pid_index),
                'l_class_pid_count': len(eval_l_class_pids),
                'scenario_target_ratios': eval_scenario_ratios,
                'height_target_ratios': eval_height_target_ratios,
            },
        },
    }
    generation_config_path = os.path.join(output_dir, 'generation_config.json')
    write_generation_config(generation_config, generation_config_path)
    print(f"[split] generation_config 已写出: {generation_config_path}")

    train_output_dir = os.path.join(output_dir, 'train_images')
    eval_output_dir = os.path.join(output_dir, 'eval_images')
    max_original_traj_vis = max(0, int(max_original_traj_vis or 0))
    train_vis_quota = min(max_original_traj_vis, n_train_traj)
    eval_vis_quota = max(0, min(max_original_traj_vis - train_vis_quota, n_eval_traj))
    train_vis_output_dir = (os.path.join(output_dir, 'train_original_traj_vis')
                            if train_vis_quota > 0 else None)
    eval_vis_output_dir = (os.path.join(output_dir, 'eval_original_traj_vis')
                           if eval_vis_quota > 0 else None)
    if train_pid_index and n_train_traj > 0:
        train_samples, train_manifest = batch_generate(
            train_pid_index,
            n_train_traj,
            train_output_dir,
            return_manifest=True,
            split_name='train',
            gsd_cache=gsd_cache,
            vis_output_dir=train_vis_output_dir,
            max_turn_deg_per_step=max_turn_deg_per_step,
            forward_bias_min_ratio=forward_bias_min_ratio,
            rear_turn_slowdown_ratio=rear_turn_slowdown_ratio,
            action_grid_size_m=action_grid_size_m,
            terminal_action_grid_size_m=terminal_action_grid_size_m,
            landmark_max_distance_m=landmark_max_distance_m,
            landmark_nearby_distance_m=landmark_nearby_distance_m,
            min_start_distance_m=min_start_distance_m,
            max_workers=max_workers,
            include_stop_scenario=include_stop_scenario,
            assistant_format=assistant_format,
            max_original_traj_vis=train_vis_quota,
            progress_interval=progress_interval,
        )
    else:
        train_samples, train_manifest = [], []
    if val_pid_index and n_eval_traj > 0:
        eval_samples, eval_manifest = batch_generate(
            val_pid_index,
            n_eval_traj,
            eval_output_dir,
            return_manifest=True,
            split_name='eval',
            gsd_cache=gsd_cache,
            vis_output_dir=eval_vis_output_dir,
            max_turn_deg_per_step=max_turn_deg_per_step,
            forward_bias_min_ratio=forward_bias_min_ratio,
            rear_turn_slowdown_ratio=rear_turn_slowdown_ratio,
            action_grid_size_m=action_grid_size_m,
            terminal_action_grid_size_m=terminal_action_grid_size_m,
            landmark_max_distance_m=landmark_max_distance_m,
            landmark_nearby_distance_m=landmark_nearby_distance_m,
            min_start_distance_m=min_start_distance_m,
            max_workers=max_workers,
            include_stop_scenario=include_stop_scenario,
            assistant_format=assistant_format,
            max_original_traj_vis=eval_vis_quota,
            progress_interval=progress_interval,
        )
    else:
        eval_samples, eval_manifest = [], []

    train_jsonl_path = os.path.join(output_dir, 'train.jsonl')
    eval_jsonl_path = os.path.join(output_dir, 'eval.jsonl')
    train_manifest_path = os.path.join(output_dir, 'train_manifest.jsonl')
    eval_manifest_path = os.path.join(output_dir, 'eval_manifest.jsonl')
    write_jsonl(train_samples, train_jsonl_path)
    write_jsonl(eval_samples, eval_jsonl_path)
    write_jsonl(train_manifest, train_manifest_path)
    write_jsonl(eval_manifest, eval_manifest_path)
    if dataset_info_path is not None:
        update_dataset_info(dataset_info_path, train_jsonl_path, eval_jsonl_path)

    print(f"[split] train.jsonl: {len(train_samples)} 条 transition, 路径: {train_jsonl_path}")
    print(f"[split] eval.jsonl: {len(eval_samples)} 条 transition, 路径: {eval_jsonl_path}")
    if dataset_info_path is not None:
        print(f"[split] dataset_info 已同步更新: {dataset_info_path}")
    print(f"[split] train_manifest.jsonl: {len(train_manifest)} rows, path: {train_manifest_path}")
    print(f"[split] eval_manifest.jsonl: {len(eval_manifest)} rows, path: {eval_manifest_path}")
    return {
        'train_pids': train_pids,
        'val_pids': val_pids,
        'train_samples': train_samples,
        'eval_samples': eval_samples,
        'train_jsonl_path': train_jsonl_path,
        'eval_jsonl_path': eval_jsonl_path,
        'train_manifest_path': train_manifest_path,
        'eval_manifest_path': eval_manifest_path,
        'train_ids_path': train_ids_path,
        'val_ids_path': val_ids_path,
        'generation_config_path': generation_config_path,
    }


if __name__ == '__main__':
    # 示例运行入口
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data', help='数据集根目录')
    parser.add_argument('--dataset', type=str, default='potsdam',
                        choices=['potsdam', 'dota', 'auto'],
                        help='数据集类型: potsdam 使用 ISPRS Potsdam 正射图+语义标签; dota 保留旧 DOTA OBB 流程; auto 优先尝试 Potsdam。')
    parser.add_argument('--output_dir', type=str, default=None, help='输出轨迹图和 JSONL 的目录, 默认写到 data_root')
    parser.add_argument('--save_original_traj_vis', type=int, nargs='?', const=50, default=0,
                        help='最多额外导出多少条原图轨迹叠加图; 不传或为 0 表示不保存, 仅写 --save_original_traj_vis 时默认保存 50 条')
    parser.add_argument('--max_turn_deg_per_step', type=float, default=MAX_TURN_DEG_PER_STEP,
                        help='单步最大转向角度上限(度), 越小轨迹越平滑')
    parser.add_argument('--forward_bias_min_ratio', type=float, default=FORWARD_BIAS_MIN_RATIO,
                        help='平滑后动作的最小前进分量占比, 越大越不容易横摆')
    parser.add_argument('--rear_turn_slowdown_ratio', type=float, default=REAR_TURN_SLOWDOWN_RATIO,
                        help='目标在机体后方时的降速比例, 越小越不容易原地绕圈')
    parser.add_argument('--action_grid_size_m', type=float, default=ACTION_GRID_SIZE_M,
                        help='动作网格边长(米), 与视野高度解耦; 越小越容易停稳')
    parser.add_argument('--terminal_action_grid_size_m', type=float, default=TERMINAL_ACTION_GRID_SIZE_M,
                        help='近目标 terminal mode 使用的细动作网格边长(米)')
    parser.add_argument('--landmark_max_distance_m', type=float, default=LANDMARK_MAX_DISTANCE_M,
                        help='landmark 与 target 的最大允许距离(米), 过远会重新采样')
    parser.add_argument('--landmark_nearby_distance_m', type=float, default=LANDMARK_NEARBY_DISTANCE_M,
                        help='A/B 场景中可随机使用的近邻 landmark 最大距离(米)')
    parser.add_argument('--min_start_distance_m', type=float, default=MIN_START_DISTANCE_M,
                        help='非 E 场景的最小起点距离(米); 调大它可以避免无人机初始点贴近 target')
    parser.add_argument('--disable_stop_scenario', action='store_true',
                        help='禁用 E 场景, 彻底避免生成近终点/停止类轨迹')
    parser.add_argument('--assistant_format', type=str, default='thought_action',
                        choices=ASSISTANT_FORMATS,
                        help='监督目标格式: thought_action 保留思维链+动作; action_only 仅监督 Action 行')
    parser.add_argument('--max_workers', type=int, default=DEFAULT_MAX_WORKERS,
                        help='并发生成 worker 数量')
    parser.add_argument('--progress_interval', type=int, default=DEFAULT_PROGRESS_INTERVAL,
                        help='每成功生成多少条轨迹打印一次进度')
    parser.add_argument('--dataset_info_path', type=str, default=None,
                        help='dataset_info.json 路径, 默认使用 data_root/dataset_info.json')
    parser.add_argument('--num_trajectories', type=int, default=1000, help='要生成的轨迹总数 (train+eval 合计)')
    parser.add_argument('--trajectories_per_image', type=int, default=None,
                        help='按每张图生成 N 条轨迹自动计算总轨迹数; 设置后覆盖 --num_trajectories')
    parser.add_argument('--max_images', type=int, default=None,
                        help='最多使用多少张图像; 在 train/eval 划分和 trajectories_per_image 计算前生效')
    parser.add_argument('--train_ratio', type=float, default=DEFAULT_TRAIN_RATIO,
                        help='按图像 ID 划分 train/eval 的比例')
    parser.add_argument('--split_seed', type=int, default=DEFAULT_SPLIT_SEED,
                        help='图像 ID 划分随机种子, 保证 train/eval 可复现')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_root
    if args.dataset_info_path is None:
        args.dataset_info_path = os.path.join(args.data_root, 'dataset_info.json')

    print(f'正在扫描 {args.data_root}/train ...')
    pid_index, stats = build_pid_to_path_index(args.data_root, split='train', dataset=args.dataset)
    print('扫描统计:', json.dumps(stats, indent=2, ensure_ascii=False))
    print(f'正在扫描 {args.data_root}/val ...')
    _, val_stats = build_pid_to_path_index(args.data_root, split='val', dataset=args.dataset)
    print('val 扫描统计:', json.dumps(val_stats, indent=2, ensure_ascii=False))

    if args.max_images is not None:
        before_images = len(pid_index)
        pid_index = limit_pid_index(pid_index, max_images=args.max_images, seed=args.split_seed)
        print(f'[split] max_images={args.max_images}: using {len(pid_index)}/{before_images} images')

    if not pid_index:
        print('未找到有效图像-标注-元数据三件套，请检查 data_root。')
        exit(1)

    if args.trajectories_per_image is not None:
        if args.trajectories_per_image <= 0:
            raise ValueError('--trajectories_per_image must be a positive integer')
        args.num_trajectories = len(pid_index) * args.trajectories_per_image
        print(f'[split] trajectories_per_image={args.trajectories_per_image}: '
              f'{len(pid_index)} images -> {args.num_trajectories} trajectories')

    print(f'开始按图像 ID 划分并生成 {args.num_trajectories} 条轨迹数据...')
    result = generate_train_eval_jsonl(
        pid_index,
        output_dir=args.output_dir,
        n_trajectories=args.num_trajectories,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        dataset_info_path=args.dataset_info_path,
        max_original_traj_vis=args.save_original_traj_vis,
        max_turn_deg_per_step=args.max_turn_deg_per_step,
        forward_bias_min_ratio=args.forward_bias_min_ratio,
        rear_turn_slowdown_ratio=args.rear_turn_slowdown_ratio,
        action_grid_size_m=args.action_grid_size_m,
        terminal_action_grid_size_m=args.terminal_action_grid_size_m,
        landmark_max_distance_m=args.landmark_max_distance_m,
        landmark_nearby_distance_m=args.landmark_nearby_distance_m,
        min_start_distance_m=args.min_start_distance_m,
        max_workers=args.max_workers,
        include_stop_scenario=not args.disable_stop_scenario,
        assistant_format=args.assistant_format,
        progress_interval=args.progress_interval,
    )
    n_total_samples = len(result['train_samples']) + len(result['eval_samples'])
    print(f"数据生成完成！共 {n_total_samples} 条 transition (步) 样本。")

    '''
    python train_data_generate.py --output_dir ../data/potsdam_out --train_ratio 0.8 --trajectories_per_image 80 --min_start_distance_m 20 --landmark_nearby_distance_m 50 --landmark_max_distance_m 80 --save_original_traj_vis 600 --progress_interval 300 --max_workers 10 --assistant_format action_only 

    python train_data_generate.py --output_dir ../data/smoke_test --train_ratio 0.8 --trajectories_per_image 10 --min_start_distance_m 60 --landmark_nearby_distance_m 60 --landmark_max_distance_m 80 --save_original_traj_vis 50 --progress_interval 10 --max_workers 10 --assistant_format action_only 

    python validate_generated_data.py --output_dir ../data/potsdam_out

    Qwen模型loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/model.safetensors.index.json
    '''
    webhook_url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=27acf86d-f375-43cb-9fa1-90b1ebfa250c"
    if webhook_url:
        try:
            import requests

            payload = {
                "msgtype": "text",
                "text": {
                    "content": "已完成"
                }
            }
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"[wecom] webhook pushed: HTTP {response.status_code}")
        except Exception as exc:
            print(f"[wecom] webhook push failed: {exc}")
            

"""服务器权威移动校正（authoritative movement reconciliation）。

玩家在网络抖动 / 丢包后看到的“瞬移”，通常不是单个 bug，而是几处机制同时
失效。本模块逐项修复：

1. **输入帧确认**（``frame_ack``）
   服务器只确认它真正处理过的输入帧。重放必须从 ``acked + 1`` 开始：
   旧帧既不能重放（会多走一段，回滚距离被放大），也不能因为迟到或重复的
   确认包把确认号往回拨。未确认帧超过上限时放弃增量重放，改走全量快照。
2. **误差平滑阈值**（``error_smooth`` / ``snap_threshold``）
   小误差直接忽略（保持预测、消除插值抖动），中等误差按与帧率无关的
   指数收敛平滑校正，超过阈值才硬吸附。
3. **共享碰撞查询**（``shared_collision``）
   客户端预测与服务器校验调用同一个碰撞例程：同样的量化步长、同样的
   解析扫掠，两端结果逐位一致；整段扫掠不会在低帧率下穿墙。
4. **技能位移的服务端校验**（``dash_validate`` / ``validation``）
   瞬移 / 冲刺不是免检通道。服务器仍然校验输入帧确认、最大位移、冷却与
   路径碰撞；越界时把角色放在合法的最大位移点，撞墙时放在接触点，而不是
   接受客户端上报的终点。
5. **RTT 平滑**（``rtt_smooth``）
   新样本先按上一估计值做比例钳制，再进 EMA；单次抖动不会把 RTT 估计和
   插值延迟拉飞。
6. **回滚后的表现插值**（``rollback_present``）
   校正后渲染层不瞬移，而是保留一个按指数衰减的视觉偏移，在若干帧内追上
   权威位置；只有偏移超过阈值才允许硬吸附。

其余入口（``predict_step``、``audit``、``fallback``、``metrics``、
``lag_comp``、``authority_state``、``determinism``）提供流程所需的支撑。
模块级函数都是无状态纯函数；``AuthorityMoveReconciler`` 把它们串成一次
完整的预测—确认—校正—表现流程。位置既可以是标量，也可以是元组；碰撞按
二维平面（x, y）处理，第三维及以后原样保留。
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import threading
from typing import Any, Mapping

__all__ = [
    "TICK_RATE",
    "FIXED_DT",
    "POSITION_QUANTUM",
    "MAX_REPLAY_FRAMES",
    "MAX_QUERY_STEP",
    "DASH_MAX_DISTANCE",
    "AuthorityMoveReconciler",
    "frame_ack",
    "error_smooth",
    "shared_collision",
    "dash_validate",
    "rtt_smooth",
    "rollback_present",
    "predict_step",
    "validation",
    "audit",
    "audit_log",
    "fallback",
    "snap_threshold",
    "metrics",
    "lag_comp",
    "authority_state",
    "determinism",
]

#: 服务器权威模拟频率（Hz）。客户端预测必须对齐到该 tick。
TICK_RATE = 60
FIXED_DT = 1.0 / TICK_RATE

#: 权威位置量化步长（1/1024 m）。两端比较前统一量化，消除浮点漂移。
POSITION_QUANTUM = 1.0 / 1024.0

#: 一次最多重放的未确认输入帧数（1.5 秒 @ 60Hz）。
MAX_REPLAY_FRAMES = 90

#: 扫掠碰撞的参考单步长度（米），小于常见墙体厚度。
MAX_QUERY_STEP = 0.05
MAX_SUBSTEPS = 512

#: 单次 predict_step 的最大子步数。碰撞是解析扫掠，步长变大也不会穿透，
#: 因此对超大 dt 只限制迭代次数、不改变总位移。
MAX_PREDICT_SEGMENTS = 64

#: 冲刺 / 瞬移的默认最大位移（米），服务端校验用。
DASH_MAX_DISTANCE = 6.0

#: 误差平滑参数。
DEFAULT_DEAD_ZONE = 0.02       # 小于该距离视为抖动，直接忽略
DEFAULT_SNAP_DISTANCE = 1.5    # 大于该距离直接硬吸附
DEFAULT_CONVERGENCE = 0.08     # 指数收敛时间常数（秒）
PRESENT_DEAD_ZONE = 1e-4

#: RTT 平滑参数：新样本先钳制在上一估计值的 1/max_ratio ~ max_ratio 倍。
RTT_ALPHA = 0.2
RTT_MAX_RATIO = 2.0

#: 延迟补偿上限（秒）。
LAG_COMP_MAX = 0.25

_MISSING = object()

_AUDIT_SEQ = itertools.count(1)
_AUDIT_LOCK = threading.Lock()
_AUDIT_LOG: list[dict[str, Any]] = []
_AUDIT_LOG_LIMIT = 1024


# ---------------------------------------------------------------------------
# 基础工具：数值、向量、量化、规范化
# ---------------------------------------------------------------------------

def _finite(value: Any, default: Any = None) -> float | None:
    """把 ``value`` 转成有限浮点数，失败返回 ``default``。"""
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _as_vec(value: Any) -> tuple[tuple[float, ...], bool]:
    """返回 ``(分量元组, 是否向量输入)``；非法分量按 0 处理。"""
    if isinstance(value, (list, tuple)):
        vec = tuple(_finite(item, 0.0) for item in value)
        return (vec if vec else (0.0,)), True
    return ((_finite(value, 0.0),), False)


def _all_finite(value: Any) -> bool:
    """检查原始输入是否全部为有限数（拒绝 NaN / inf / 非数值）。"""
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_all_finite(item) for item in value)
    return _finite(value, None) is not None


def _fit(vec: tuple[float, ...], dim: int) -> tuple[float, ...]:
    """把向量补零 / 截断到固定维度。"""
    if len(vec) >= dim:
        return vec[:dim]
    return vec + (0.0,) * (dim - len(vec))


def _out(vec: tuple[float, ...], was_vector: bool, dim: int | None = None) -> Any:
    """按输入形态还原输出：标量进标量出，向量进元组出。"""
    if dim is not None:
        vec = tuple(vec[:dim])
    if was_vector:
        return tuple(vec)
    return vec[0] if vec else 0.0


def _align(a: tuple[float, ...], b: tuple[float, ...]
           ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """把两个向量补零到同一维度。"""
    dim = max(len(a), len(b))
    return a + (0.0,) * (dim - len(a)), b + (0.0,) * (dim - len(b))


def _vadd(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b))


def _vsub(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _vscale(a: tuple[float, ...], factor: float) -> tuple[float, ...]:
    return tuple(x * factor for x in a)


def _vnorm(a: tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _quantize(vec: tuple[float, ...]) -> tuple[float, ...]:
    """按 ``POSITION_QUANTUM`` 量化，抵消两端的浮点累计误差。"""
    return tuple(round(component / POSITION_QUANTUM) * POSITION_QUANTUM
                 for component in vec)


def _pad2(vec: tuple[float, ...]) -> tuple[float, float]:
    return (vec[0], vec[1] if len(vec) > 1 else 0.0)


def _canonical(value: Any) -> str:
    """稳定序列化：键排序、紧凑分隔符，保证摘要可复现。"""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=repr, ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. 输入帧确认
# ---------------------------------------------------------------------------

def frame_ack(acked: Any = 0, received: Any = None, pending: Any = None,
              *, limit: int = MAX_REPLAY_FRAMES) -> dict[str, Any]:
    """处理一次输入帧确认，返回可安全重放的区间。

    参数
    ----
    acked
        客户端本地记录的已确认帧号（服务器已处理到的帧，-1 表示尚无）。
    received
        本次回包携带的确认帧号。迟到 / 重传的包可能小于 ``acked``。
    pending
        当前尚未确认的输入帧数量。
    limit
        允许一次性重放的帧数上限。

    语义
    ----
    * 确认号单调不减：``acked = max(acked, received)``；迟到的确认包不能
      把确认号往回拨（``stale`` / ``rewind`` 为 True）。
    * 重放从 ``acked + 1`` 开始，已确认帧绝不重放，否则同一段位移会被
      重复应用，造成“回滚距离过大”。
    * ``pending > limit`` 时不再重放，标记 ``require_snapshot``，改走全量
      快照重同步。
    """
    last = _finite(acked, 0.0)
    last = max(int(last), -1)

    if received is None:
        received_value = last
    else:
        received_raw = _finite(received, float(last))
        received_value = max(int(received_raw), -1)

    stale = received_value < last
    duplicate = received_value <= last
    confirmed = max(last, received_value)

    if pending is None:
        pending_count = 0
    else:
        pending_raw = _finite(pending, 0.0)
        pending_count = max(int(pending_raw), 0)

    limit = max(int(_finite(limit, MAX_REPLAY_FRAMES)), 0)
    allow_replay = pending_count <= limit
    replay_count = pending_count if allow_replay else 0

    return {
        "ok": True,
        "acked": confirmed,
        "received": received_value,
        "next_tick": confirmed + 1,
        "duplicate": duplicate,
        "stale": stale,
        "rewind": stale,
        # 只重放未确认的帧，起点永远是 acked + 1。
        "replay_from": confirmed + 1,
        "replay_to": confirmed + replay_count,
        "replay_count": replay_count,
        "pending": pending_count,
        "limit": limit,
        "allow_replay": allow_replay,
        "require_snapshot": not allow_replay,
        # 已确认帧不会进入重放；字段保留便于审计。
        "discarded_frames": 0,
    }


# ---------------------------------------------------------------------------
# 2. 误差平滑与阈值
# ---------------------------------------------------------------------------

def snap_threshold(error: Any = 0.0, *,
                   dead_zone: float = DEFAULT_DEAD_ZONE,
                   snap_distance: float = DEFAULT_SNAP_DISTANCE
                   ) -> dict[str, Any]:
    """按误差大小选择校正动作：``ignore`` / ``smooth`` / ``snap``。"""
    vec, was_vector = _as_vec(error)
    distance = _vnorm(vec)
    dead_zone = max(_finite(dead_zone, DEFAULT_DEAD_ZONE), 0.0)
    snap_distance = max(_finite(snap_distance, DEFAULT_SNAP_DISTANCE), 0.0)
    if snap_distance < dead_zone:
        snap_distance = dead_zone

    if distance <= dead_zone:
        action = "ignore"
    elif distance > snap_distance:
        action = "snap"
    else:
        action = "smooth"

    ratio = distance / snap_distance if snap_distance > 0 else 1.0
    return {
        "ok": True,
        "distance": distance,
        "dead_zone": dead_zone,
        "snap_distance": snap_distance,
        "action": action,
        "ignore": action == "ignore",
        "smooth": action == "smooth",
        "snap": action == "snap",
        "ratio": ratio,
        "vector": was_vector,
    }


def error_smooth(current: Any = 0.0, target: Any = None, dt: Any = FIXED_DT,
                 *, convergence_time: float = DEFAULT_CONVERGENCE,
                 dead_zone: float = DEFAULT_DEAD_ZONE,
                 snap_distance: float = DEFAULT_SNAP_DISTANCE,
                 max_speed: Any = None) -> Any:
    """把 ``current`` 向 ``target`` 平滑推进一帧，消除校正瞬移。

    * 误差在 ``dead_zone`` 内：维持 ``current``（低速时不发生漂移）。
    * 误差超过 ``snap_distance``：直接取 ``target``（传送 / 复活等大偏差）。
    * 其余情况：``alpha = 1 - exp(-dt / convergence_time)`` 的指数收敛。
      系数只取决于经过的时间，60Hz 与 144Hz 下经过相同时间收敛量一致。

    ``max_speed``（单位/秒）可选，用于限制单帧最大修正速度。
    标量输入返回标量，向量输入返回元组。
    """
    if target is None:
        target = current
    was_vector = isinstance(current, (list, tuple)) or isinstance(
        target, (list, tuple))
    cur_vec, tgt_vec = _align(_as_vec(current)[0], _as_vec(target)[0])

    dt_value = max(_finite(dt, 0.0), 0.0)
    convergence = max(_finite(convergence_time, DEFAULT_CONVERGENCE), 0.0)

    delta = _vsub(tgt_vec, cur_vec)
    distance = _vnorm(delta)

    dead_zone = max(_finite(dead_zone, DEFAULT_DEAD_ZONE), 0.0)
    snap_distance = max(_finite(snap_distance, DEFAULT_SNAP_DISTANCE), 0.0)
    if snap_distance < dead_zone:
        snap_distance = dead_zone

    if distance <= dead_zone:
        return _out(cur_vec, was_vector)
    if distance > snap_distance:
        return _out(tgt_vec, was_vector)
    if dt_value <= 0:
        return _out(cur_vec, was_vector)

    if convergence <= 0:
        alpha = 1.0
    else:
        alpha = 1.0 - math.exp(-dt_value / convergence)
    alpha = min(max(alpha, 0.0), 1.0)

    step = _vscale(delta, alpha)
    speed_cap = _finite(max_speed, None)
    if speed_cap is not None and speed_cap >= 0:
        max_step = speed_cap * dt_value
        step_length = _vnorm(step)
        if max_step <= 0:
            return _out(cur_vec, was_vector)
        if step_length > max_step:
            step = _vscale(step, max_step / step_length)

    return _out(_vadd(cur_vec, step), was_vector)


def rollback_present(presented: Any = 0.0, corrected: Any = None,
                     dt: Any = FIXED_DT, *,
                     convergence_time: float = DEFAULT_CONVERGENCE,
                     snap_distance: float = DEFAULT_SNAP_DISTANCE,
                     dead_zone: float = PRESENT_DEAD_ZONE,
                     max_speed: Any = None) -> Any:
    """回滚后的表现层插值：视觉位置按指数衰减追上权威位置。

    与 ``error_smooth`` 的视角相反：这里保存的是视觉偏移
    ``presented - corrected``，每帧乘以 ``exp(-dt / convergence_time)``
    衰减，因此校正瞬间画面不会瞬移。

    偏移小于 ``dead_zone`` 时直接对齐；超过 ``snap_distance`` 时硬吸附
    （传送 / 复活若做平滑，会表现为角色贴地滑过整张地图）。
    """
    if corrected is None:
        corrected = presented
    was_vector = isinstance(presented, (list, tuple)) or isinstance(
        corrected, (list, tuple))
    present_vec, correct_vec = _align(_as_vec(presented)[0],
                                      _as_vec(corrected)[0])

    dt_value = max(_finite(dt, 0.0), 0.0)
    convergence = max(_finite(convergence_time, DEFAULT_CONVERGENCE), 0.0)
    snap_distance = max(_finite(snap_distance, DEFAULT_SNAP_DISTANCE), 0.0)
    dead_zone = max(_finite(dead_zone, PRESENT_DEAD_ZONE), 0.0)
    if snap_distance < dead_zone:
        snap_distance = dead_zone

    offset = _vsub(present_vec, correct_vec)
    distance = _vnorm(offset)
    if distance <= dead_zone or distance > snap_distance:
        return _out(correct_vec, was_vector)
    if dt_value <= 0:
        return _out(present_vec, was_vector)

    if convergence <= 0:
        remaining = (0.0,) * len(offset)
    else:
        remaining = _vscale(offset, math.exp(-dt_value / convergence))

    speed_cap = _finite(max_speed, None)
    if speed_cap is not None and speed_cap >= 0:
        max_step = speed_cap * dt_value
        travelled = distance - _vnorm(remaining)
        if travelled > max_step:
            if max_step <= 0:
                return _out(present_vec, was_vector)
            remaining = _vscale(offset, max(0.0, 1.0 - max_step / distance))

    return _out(_vadd(correct_vec, remaining), was_vector)


# ---------------------------------------------------------------------------
# 3. 共享碰撞查询
# ---------------------------------------------------------------------------

def _normalize_boxes(obstacles: Any
                     ) -> list[tuple[float, float, float, float, int]]:
    """把障碍物统一成 ``(min_x, min_y, max_x, max_y, original_index)``。

    支持 ``(min_x, min_y, max_x, max_y)``、``(min_point, max_point)``、
    ``{"min": ..., "max": ...}`` 与 ``{"min_x": ...}`` 四种写法。
    """
    boxes: list[tuple[float, float, float, float, int]] = []
    if obstacles is None:
        return boxes
    if isinstance(obstacles, Mapping):
        obstacles = list(obstacles.values())
    try:
        iterable = list(obstacles)
    except TypeError:
        return boxes

    for index, box in enumerate(iterable):
        coords = None
        if isinstance(box, Mapping):
            if all(key in box for key in ("min_x", "min_y", "max_x", "max_y")):
                coords = (box["min_x"], box["min_y"], box["max_x"], box["max_y"])
            elif "min" in box and "max" in box:
                lo = _pad2(_as_vec(box["min"])[0])
                hi = _pad2(_as_vec(box["max"])[0])
                coords = (lo[0], lo[1], hi[0], hi[1])
        elif isinstance(box, (list, tuple)):
            if len(box) >= 4:
                coords = tuple(box[:4])
            elif len(box) == 2:
                lo = _pad2(_as_vec(box[0])[0])
                hi = _pad2(_as_vec(box[1])[0])
                coords = (lo[0], lo[1], hi[0], hi[1])
        if coords is None:
            continue
        values = [_finite(item) for item in coords]
        if any(item is None for item in values):
            continue
        x0, x1 = sorted((values[0], values[2]))
        y0, y1 = sorted((values[1], values[3]))
        # 量化障碍物：与位置使用同一网格，避免“贴边站着却判定不碰”的
        # 两端不一致。盒子按量化后网格向内对齐，不会凭空变薄。
        x0 = math.floor(x0 / POSITION_QUANTUM + 1e-9) * POSITION_QUANTUM
        x1 = math.ceil(x1 / POSITION_QUANTUM - 1e-9) * POSITION_QUANTUM
        y0 = math.floor(y0 / POSITION_QUANTUM + 1e-9) * POSITION_QUANTUM
        y1 = math.ceil(y1 / POSITION_QUANTUM - 1e-9) * POSITION_QUANTUM
        boxes.append((x0, y0, x1, y1, index))

    # 几何排序 + 原始下标，保证多障碍物时命中顺序确定。
    boxes.sort()
    return boxes


def _segment_aabb(p0: tuple[float, float], delta: tuple[float, float],
                  box: tuple[float, float, float, float, int]
                  ) -> tuple[float, tuple[float, float], int] | None:
    """线段 ``p0 + t * delta``（t ∈ [0, 1]）与轴对齐盒的最近交点。

    采用闭合盒语义，但区分“接触面”与“穿透”：

    * 已贴在墙面上向外移动（``t_exit <= 0``）不算命中，否则角色会卡在
      墙面无法脱离；
    * 只在终点处擦到表面（``t_enter >= 1``）不算命中，允许恰好走到墙面；
    * 起点已在盒内则立即命中（``t = 0``），法线退化为零向量。
    """
    x0, y0, x1, y1, index = box
    t_enter, t_exit = -math.inf, math.inf
    normal: list[float] | None = None

    for axis, (low, high) in enumerate(((x0, x1), (y0, y1))):
        origin = p0[axis]
        direction = delta[axis]
        if abs(direction) < 1e-12:
            # 该轴不移动：必须落在盒子的闭区间内。
            if origin < low or origin > high:
                return None
            continue
        t_low = (low - origin) / direction
        t_high = (high - origin) / direction
        if t_low > t_high:
            t_low, t_high = t_high, t_low
        if t_low > t_enter:
            t_enter = t_low
            normal = [0.0, 0.0]
            # 从低侧进入：运动方向为正，接触法线指向 -axis。
            normal[axis] = -1.0 if direction > 0 else 1.0
        if t_high < t_exit:
            t_exit = t_high
        if t_enter > t_exit:
            return None

    if t_enter >= 1.0:
        # 只在终点擦到表面：视为到达，不算被挡住。
        return None
    if t_exit <= 0.0:
        # 盒子在起点身后，或起点贴面且正在离开。
        return None

    if normal is None:
        # 起点已在盒内：立即命中，法线退化为零向量。
        normal = [0.0, 0.0]
    return max(t_enter, 0.0), (normal[0], normal[1]), index


def shared_collision(start: Any = 0.0, end: Any = None,
                     obstacles: Any = None, *, radius: Any = 0.0,
                     max_step: float = MAX_QUERY_STEP) -> dict[str, Any]:
    """客户端与服务器共用的碰撞查询（线段扫掠 vs 轴对齐盒）。

    * 两端先按 ``POSITION_QUANTUM`` 量化输入，再做同一套解析扫掠求交，
      所以同样的输入得到逐位一致的结果。
    * 整段一次求交，而不是每帧采点，高速 / 低帧率下也不会穿过薄墙。
    * ``steps`` 是按 ``max_step`` 预算需要的子步数，仅用于审计与限流。

    返回 ``hit``、最终 ``position``（命中时为接触点）、``point``、
    ``normal``、命中障碍物的原始下标等。
    """
    start_vec, was_vector = _as_vec(start)
    end_vec = start_vec if end is None else _as_vec(end)[0]
    start_vec, end_vec = _align(start_vec, end_vec)
    dim = len(start_vec)
    tail = start_vec[2:]

    p0 = _quantize(_pad2(start_vec))
    p1 = _quantize(_pad2(end_vec))
    delta = _vsub(p1, p0)
    requested_distance = _vnorm(delta)

    radius_value = max(_finite(radius, 0.0), 0.0)
    boxes = _normalize_boxes(obstacles)
    if radius_value > 0:
        boxes = [(x0 - radius_value, y0 - radius_value,
                  x1 + radius_value, y1 + radius_value, index)
                 for x0, y0, x1, y1, index in boxes]

    max_step = max(_finite(max_step, MAX_QUERY_STEP), POSITION_QUANTUM)
    steps = max(1, math.ceil(requested_distance / max_step))
    steps = min(steps, MAX_SUBSTEPS)

    best: tuple[float, tuple[float, float], int] | None = None
    for box in boxes:
        hit = _segment_aabb(p0, delta, box)
        if hit is None:
            continue
        if best is None or hit[0] < best[0]:
            best = hit

    if best is None:
        final = p1
        point_vec: tuple[float, ...] | None = None
        normal = (0.0, 0.0)
        obstacle = None
        hit_flag = False
    else:
        t_value, normal, obstacle = best
        final = _quantize(_vadd(p0, _vscale(delta, t_value)))
        point_vec = final + tail
        hit_flag = True

    final_vec = final + tail
    return {
        "ok": True,
        "hit": hit_flag,
        "position": _out(final_vec, was_vector, dim),
        "point": (_out(point_vec, was_vector, dim)
                  if point_vec is not None else None),
        "normal": normal,
        "obstacle": obstacle,
        "distance": _vnorm(_vsub(final, p0)),
        "requested_distance": requested_distance,
        "start": _out(tuple(p0) + tail, was_vector, dim),
        "end": _out(tuple(p1) + tail, was_vector, dim),
        "steps": steps,
        "max_step": max_step,
        "quantum": POSITION_QUANTUM,
        "radius": radius_value,
    }


# ---------------------------------------------------------------------------
# 4. 技能位移的服务端校验
# ---------------------------------------------------------------------------

def dash_validate(start: Any = 0.0, end: Any = None, *,
                  max_distance: float = DASH_MAX_DISTANCE,
                  obstacles: Any = None, cooldown_ready: Any = True,
                  kind: str = "dash", radius: Any = 0.0) -> dict[str, Any]:
    """校验一次冲刺 / 瞬移位移，返回服务器认可的落点。

    校验顺序与失败原因：
    ``invalid_input`` → ``cooldown`` → ``displacement_exceeded`` →
    ``blocked``。未通过时 ``position`` 仍是合法落点（最大位移点或接触点），
    客户端应直接采用它，而不是回退到旧位置。
    """
    start_vec, was_vector = _as_vec(start)
    end_vec = start_vec if end is None else _as_vec(end)[0]
    start_vec, end_vec = _align(start_vec, end_vec)
    dim = len(start_vec)

    finite_input = _all_finite(start) and (end is None or _all_finite(end))

    max_distance = max(_finite(max_distance, DASH_MAX_DISTANCE), 0.0)
    cooldown_ok = bool(cooldown_ready)

    delta = _vsub(end_vec, start_vec)
    requested = _vnorm(delta)
    within_range = requested <= max_distance + POSITION_QUANTUM

    accepted = True
    reason = "ok"
    clamped = False
    candidate = end_vec

    if not finite_input:
        accepted, reason = False, "invalid_input"
        candidate = start_vec
    elif not cooldown_ok:
        accepted, reason = False, "cooldown"
        candidate = start_vec
    elif not within_range:
        accepted, reason, clamped = False, "displacement_exceeded", True
        if requested > 0:
            candidate = _vadd(start_vec, _vscale(delta, max_distance / requested))
        else:
            candidate = start_vec

    collision = shared_collision(start_vec, candidate, obstacles, radius=radius)
    if collision["hit"]:
        candidate = _as_vec(collision["position"])[0]
        if accepted:
            accepted, reason = False, "blocked"

    checks = {
        "finite": finite_input,
        "cooldown": cooldown_ok,
        "distance": within_range,
        "collision": not collision["hit"],
    }
    return {
        "ok": accepted,
        "accepted": accepted,
        "authoritative": True,
        "reason": reason,
        "position": _out(candidate, was_vector, dim),
        "start": _out(start_vec, was_vector, dim),
        "requested_end": _out(end_vec, was_vector, dim),
        "requested_distance": requested,
        "distance": _vnorm(_vsub(candidate, start_vec)),
        "max_distance": max_distance,
        "clamped": clamped,
        "checks": checks,
        "collision": collision,
        "kind": kind,
        # 位移类技能与普通移动走同一条校验链，瞬移不能绕过校正。
        "bypass": False,
    }


def validation(claimed: Any = None, *, authorized: Any = None,
               max_distance: Any = None,
               obstacles: Any = None, acked: Any = None,
               tick: Any = None, cooldown_ready: Any = True,
               kind: str = "move", radius: Any = 0.0) -> dict[str, Any]:
    """服务器对客户端上报移动的统一校验入口。

    ``claimed`` 可以是位置值，也可以是包含 ``start`` / ``position`` /
    ``tick`` / ``acked`` / ``kind`` 等字段的字典。校验内容：

    * 必须能确定权威起点（``start`` / ``authorized`` 参数），否则位移无从
      校验，返回 ``missing_authoritative_start``；
    * 输入帧必须已被确认（``tick <= acked + 1``），防止用旧输入帧申请位移；
    * 位移距离、冷却与路径碰撞交给 ``dash_validate``，与冲刺共用一套规则。
    """
    payload: dict[str, Any] = {}
    if isinstance(claimed, Mapping):
        payload = dict(claimed)
        start = payload.get("start", payload.get(
            "authorized", payload.get("from")))
        if start is None:
            start = authorized
        end = payload.get("position", payload.get(
            "end", payload.get("target")))
        tick = payload.get("tick", tick)
        acked = payload.get("acked", acked)
        cooldown_ready = payload.get("cooldown_ready", cooldown_ready)
        kind = payload.get("kind", kind)
        obstacles = payload.get("obstacles", obstacles)
        if max_distance is None:
            max_distance = payload.get("max_distance")
    elif authorized is None:
        start, end = claimed, claimed
    else:
        start, end = authorized, claimed

    if max_distance is None:
        max_distance = DASH_MAX_DISTANCE

    if start is None:
        # 没有权威起点就无法校验位移：拒绝，但返回结构仍然完整。
        reason = ("invalid_input" if end is None
                  else "missing_authoritative_start")
        return {
            "ok": False, "valid": False, "reason": reason,
            "position": None,
            "checks": {"finite": end is not None, "ack": True,
                       "cooldown": bool(cooldown_ready),
                       "distance": False, "collision": True},
            "detail": None, "tick": tick, "acked": acked, "kind": kind,
            "authoritative": True, "bypass": False,
        }

    movement = dash_validate(start, end, max_distance=max_distance,
                             obstacles=obstacles,
                             cooldown_ready=cooldown_ready,
                             kind=kind, radius=radius)

    ack_ok = True
    ack_reason = None
    tick_value = _finite(tick, None)
    acked_value = _finite(acked, None)
    if tick_value is not None and acked_value is not None:
        if int(tick_value) > int(acked_value) + 1:
            ack_ok = False
            ack_reason = "input_frame_not_acknowledged"

    checks = {
        "finite": movement["checks"]["finite"],
        "ack": ack_ok,
        "distance": movement["checks"]["distance"],
        "collision": movement["checks"]["collision"],
        "cooldown": movement["checks"]["cooldown"],
    }
    ok = movement["accepted"] and ack_ok
    reason = ack_reason if not ack_ok else movement["reason"]

    return {
        "ok": ok,
        "valid": ok,
        "authoritative": True,
        "reason": reason,
        "position": movement["position"],
        "start": movement["start"],
        "distance": movement["distance"],
        "checks": checks,
        "detail": movement,
        "tick": tick,
        "acked": acked,
        "kind": kind,
        "bypass": False,
    }


# ---------------------------------------------------------------------------
# 5. RTT 平滑
# ---------------------------------------------------------------------------

def rtt_smooth(previous: Any = 0.0, sample: Any = None, *,
               alpha: float = RTT_ALPHA,
               max_ratio: float = RTT_MAX_RATIO) -> float:
    """RTT 估计的钳制 + 指数平滑。

    新样本先被限制在上一估计值的 ``1/max_ratio ~ max_ratio`` 倍之间，
    再做 EMA。一次网络抖动不会让 RTT 估计突变，插值延迟因此保持稳定。
    首个样本（``previous <= 0``）直接采用样本值。
    """
    prev_value = _finite(previous, None)
    sample_value = _finite(sample, None)
    if prev_value is None or prev_value <= 0:
        return float(sample_value) if sample_value is not None else 0.0
    if sample_value is None or sample_value < 0:
        return float(prev_value)

    alpha = min(max(_finite(alpha, RTT_ALPHA), 0.0), 1.0)
    max_ratio = max(_finite(max_ratio, RTT_MAX_RATIO), 1.0)

    lower = prev_value / max_ratio
    upper = prev_value * max_ratio
    clamped = min(max(sample_value, lower), upper)
    return alpha * clamped + (1.0 - alpha) * prev_value


def lag_comp(rtt: Any = 0.0, max_comp: float = LAG_COMP_MAX, *,
             unit: str = "ms") -> float:
    """延迟补偿的回退时长：单程延迟 = rtt / 2，且不超过上限（秒）。

    ``unit`` 指定输入单位：``"ms"``（默认）或 ``"s"``。
    """
    rtt_value = max(_finite(rtt, 0.0), 0.0)
    if unit == "ms":
        rtt_value /= 1000.0
    limit = max(_finite(max_comp, LAG_COMP_MAX), 0.0)
    return min(rtt_value / 2.0, limit)


# ---------------------------------------------------------------------------
# 6. 预测步进（与服务器共用物理与碰撞）
# ---------------------------------------------------------------------------

def predict_step(position: Any = 0.0, velocity: Any = 0.0,
                 dt: Any = FIXED_DT, *, obstacles: Any = None,
                 tick_rate: float = TICK_RATE,
                 radius: Any = 0.0) -> dict[str, Any]:
    """按固定服务器 tick 推进预测位置。

    ``dt`` 先拆成整数个服务器 tick 和一段余量，每个 tick 固定
    ``1 / tick_rate`` 秒；无论调用方帧率如何，总位移都精确等于
    ``velocity * dt``，且每一小段都走共享碰撞扫掠，因此客户端预测与
    服务器校正的碰撞结果一致。被墙挡住时停在接触点。
    """
    position_vec, was_vector = _as_vec(position)
    velocity_vec = _as_vec(velocity)[0]
    position_vec, velocity_vec = _align(position_vec, velocity_vec)
    dim = len(position_vec)

    dt_value = max(_finite(dt, FIXED_DT), 0.0)
    rate = max(_finite(tick_rate, TICK_RATE), 1.0)
    tick_dt = 1.0 / rate

    if dt_value <= 0:
        segments: list[float] = []
    else:
        full_ticks = int(dt_value // tick_dt)
        remainder = dt_value - full_ticks * tick_dt
        # 被子步上限截断时，把剩余时间合并成等长的大步：解析扫掠不会因
        # 步长变大而穿透，总位移因此保持精确，且不会构造巨量子步。
        if full_ticks + (1 if remainder > 1e-12 else 0) > MAX_PREDICT_SEGMENTS:
            count = MAX_PREDICT_SEGMENTS
            segments = [dt_value / count] * count
        else:
            segments = [tick_dt] * full_ticks
            if remainder > 1e-12:
                segments.append(remainder)
            if not segments:
                segments = [dt_value]

    current = position_vec
    blocked = False
    travelled = 0.0
    steps = 0
    last_collision = None

    for segment in segments:
        target = _vadd(current, _vscale(velocity_vec, segment))
        collision = shared_collision(current, target, obstacles, radius=radius)
        steps += collision["steps"]
        next_position = _as_vec(collision["position"])[0]
        travelled += _vnorm(_vsub(next_position, current))
        current = next_position
        if collision["hit"]:
            blocked = True
            last_collision = collision
            break

    return {
        "ok": not blocked,
        "position": _out(current, was_vector, dim),
        "blocked": blocked,
        "ticks": len(segments),
        "requested_dt": dt_value,
        "tick_dt": tick_dt,
        "tick_rate": rate,
        "distance": travelled,
        "steps": steps,
        "collision": last_collision,
        "quantum": POSITION_QUANTUM,
    }


# ---------------------------------------------------------------------------
# 支撑能力
# ---------------------------------------------------------------------------

def audit(event: Any = None, **fields: Any) -> dict[str, Any]:
    """写一条审计记录：单调序号 + 事件摘要，便于回放两端比对。

    负载平铺在记录顶层（除 ``seq`` / ``event`` / ``digest`` 外），这样
    ``metrics()`` 可以直接吃 ``audit_log()`` 的输出。
    """
    with _AUDIT_LOCK:
        seq = next(_AUDIT_SEQ)
        payload = copy.deepcopy(fields)
        record: dict[str, Any] = {
            "ok": True,
            "seq": seq,
            "event": event,
            **payload,
        }
        record["digest"] = _digest({"seq": seq, "event": event,
                                    "fields": payload})
        _AUDIT_LOG.append(record)
        if len(_AUDIT_LOG) > _AUDIT_LOG_LIMIT:
            del _AUDIT_LOG[:len(_AUDIT_LOG) - _AUDIT_LOG_LIMIT]
        return copy.deepcopy(record)


def audit_log() -> list[dict[str, Any]]:
    """返回审计记录的副本。"""
    with _AUDIT_LOCK:
        return copy.deepcopy(_AUDIT_LOG)


def fallback(state: Any = None, *, pending: Any = 0, error: Any = 0.0,
             snapshot: Any = None, limit: int = MAX_REPLAY_FRAMES,
             snap_distance: float = DEFAULT_SNAP_DISTANCE) -> dict[str, Any]:
    """校正兜底策略：增量重放 / 硬吸附 / 全量快照重同步。

    未确认帧超过上限、或误差超过吸附阈值且提供了快照时，放弃增量重放，
    改用快照，避免一次回滚跨度过大。
    """
    pending_count = max(int(_finite(pending, 0.0)), 0)
    error_value = abs(_finite(error, 0.0))
    limit = max(int(_finite(limit, MAX_REPLAY_FRAMES)), 0)
    snap_distance = max(_finite(snap_distance, DEFAULT_SNAP_DISTANCE), 0.0)

    too_many = pending_count > limit
    too_far = error_value > snap_distance

    if too_many and snapshot is not None:
        mode, reason = "snapshot", "replay_window_exceeded"
        result = copy.deepcopy(snapshot)
    elif too_far and snapshot is not None:
        mode, reason = "snapshot", "error_exceeded"
        result = copy.deepcopy(snapshot)
    elif too_far:
        mode, reason = "snap", "error_exceeded"
        result = copy.deepcopy(state)
    elif too_many:
        mode, reason = "snapshot", "replay_window_exceeded"
        result = copy.deepcopy(snapshot if snapshot is not None else state)
    else:
        mode, reason = "replay", "within_window"
        result = copy.deepcopy(state)

    return {
        "ok": True,
        "mode": mode,
        "reason": reason,
        "state": result,
        "replay_allowed": mode == "replay",
        "require_snapshot": mode == "snapshot",
        "pending": pending_count,
        "limit": limit,
        "error": error_value,
        "snap_distance": snap_distance,
    }


def metrics(events: Any = None, **extra: Any) -> dict[str, Any]:
    """汇总校正相关的运行指标。``events`` 为事件字典的可迭代对象。"""
    records: list[Mapping[str, Any]] = []
    if isinstance(events, Mapping):
        records = [events]
    elif events is not None and not isinstance(events, (str, bytes)):
        try:
            records = [item for item in events if isinstance(item, Mapping)]
        except TypeError:
            records = []

    summary: dict[str, Any] = {
        "ok": True,
        "events": 0,
        "corrections": 0,
        "snaps": 0,
        "smooths": 0,
        "ignored": 0,
        "rollback_distance_max": 0.0,
        "rollback_distance_sum": 0.0,
        "rtt_last": None,
        "rtt_sum": 0.0,
        "rtt_count": 0,
    }

    for record in records:
        summary["events"] += 1
        action = record.get("action") or record.get("mode") or record.get("type")
        if action in ("snap", "hard", "teleport"):
            summary["snaps"] += 1
        elif action in ("smooth", "correct", "reconcile"):
            summary["smooths"] += 1
        elif action in ("ignore", "none"):
            summary["ignored"] += 1
        if record.get("corrected"):
            summary["corrections"] += 1

        rollback = _finite(record.get("rollback_distance",
                                      record.get("distance")), None)
        if rollback is not None:
            rollback = abs(rollback)
            summary["rollback_distance_sum"] += rollback
            summary["rollback_distance_max"] = max(
                summary["rollback_distance_max"], rollback)

        rtt_value = _finite(record.get("rtt"), None)
        if rtt_value is not None:
            summary["rtt_last"] = rtt_value
            summary["rtt_sum"] += rtt_value
            summary["rtt_count"] += 1

    summary["rtt_mean"] = (summary["rtt_sum"] / summary["rtt_count"]
                           if summary["rtt_count"] else None)
    summary["rollback_distance_mean"] = (
        summary["rollback_distance_sum"] / summary["events"]
        if summary["events"] else 0.0)
    summary.update(extra)
    return summary


def authority_state(state: Any = None, *, tick: Any = 0,
                    position: Any = None, velocity: Any = None
                    ) -> dict[str, Any]:
    """构造服务器权威快照（深拷贝，避免与调用方共享可变状态）。"""
    raw = copy.deepcopy(state)
    if isinstance(raw, Mapping):
        payload = dict(raw)
        if position is None:
            position = payload.get("position")
        if velocity is None:
            velocity = payload.get("velocity")
        if "tick" in payload:
            tick = payload["tick"]
    elif raw is None:
        payload = {}
    else:
        payload = {"value": raw}
        if position is None:
            position = raw

    tick_value = _finite(tick, 0.0)
    if position is None:
        position = (0.0, 0.0)
    if velocity is None:
        velocity = (0.0, 0.0)

    snapshot = {
        "ok": True,
        "authoritative": True,
        "tick": int(tick_value),
        "position": copy.deepcopy(position),
        "velocity": copy.deepcopy(velocity),
        "state": payload,
    }
    snapshot["digest"] = _digest({"tick": snapshot["tick"],
                                  "position": snapshot["position"],
                                  "velocity": snapshot["velocity"],
                                  "state": snapshot["state"]})
    return snapshot


def determinism(state: Any = _MISSING, other: Any = _MISSING, *,
                repeat: int = 2) -> dict[str, Any]:
    """检查同一输入是否得到逐位一致的结果。

    * 单个参数：返回规范化摘要（``ok`` 恒为 True）。
    * 两个参数：比较两个输入的摘要，``ok`` 表示是否一致。
    * ``state`` 为可调用对象：调用 ``repeat`` 次并比较，用于验证预测 /
      碰撞例程的确定性。
    """
    if callable(state):
        times = max(int(_finite(repeat, 2) or 2), 1)
        digests = [_digest(state()) for _ in range(times)]
        consistent = len(set(digests)) == 1
        return {
            "ok": consistent,
            "match": consistent,
            "mode": "callable",
            "digest": digests[0],
            "digests": digests,
            "repeat": times,
        }

    if state is _MISSING:
        state = None
    digest = _digest(state)

    if other is _MISSING:
        return {
            "ok": True,
            "match": True,
            "mode": "single",
            "digest": digest,
            "other_digest": digest,
            "canonical": _canonical(state),
        }

    other_digest = _digest(other)
    match = digest == other_digest
    return {
        "ok": match,
        "match": match,
        "mode": "compare",
        "digest": digest,
        "other_digest": other_digest,
        "canonical": _canonical(state),
    }


# ---------------------------------------------------------------------------
# 整合入口：预测 — 确认 — 校正 — 表现
# ---------------------------------------------------------------------------

class AuthorityMoveReconciler:
    """把各纯函数串成一条权威移动校正流水线。

    典型用法::

        rec = AuthorityMoveReconciler((0.0, 0.0))
        rec.submit_input(0, (1.0, 0.0))
        rec.submit_input(1, (1.0, 0.0))
        rec.acknowledge(0)              # 服务器只确认了第 0 帧
        rec.reconcile({"position": (1.0, 0.0), "tick": 0})
        visual = rec.present(1 / 60)    # 表现层平滑追上

    关键约束：``reconcile`` 只重放 ``acked + 1`` 之后的输入；未确认帧过多
    时整体改用快照；``present`` 永远不瞬移（除非偏差超过阈值）。
    """

    def __init__(self, position: Any = (0.0, 0.0), *,
                 tick_rate: float = TICK_RATE,
                 max_replay: int = MAX_REPLAY_FRAMES,
                 snap_distance: float = DEFAULT_SNAP_DISTANCE,
                 convergence_time: float = DEFAULT_CONVERGENCE,
                 obstacles: Any = None) -> None:
        self._was_vector = isinstance(position, (list, tuple))
        self.tick_rate = max(_finite(tick_rate, TICK_RATE), 1.0)
        self.dt = 1.0 / self.tick_rate
        self.max_replay = max(int(_finite(max_replay, MAX_REPLAY_FRAMES)), 0)
        self.snap_distance = max(_finite(snap_distance,
                                         DEFAULT_SNAP_DISTANCE), 0.0)
        self.convergence_time = max(_finite(convergence_time,
                                            DEFAULT_CONVERGENCE), 0.0)
        self.obstacles = list(obstacles) if obstacles else []
        start = _as_vec(position)[0]
        self.authoritative = start
        self.predicted = start
        self.visual = start
        self.acked = -1
        self.pending: list[dict[str, Any]] = []
        self.rtt: float | None = None
        self.events: list[dict[str, Any]] = []
        self.last_correction: dict[str, Any] | None = None
        self._visual_offset = (0.0,) * len(start)

    # -- 内部 ----------------------------------------------------------
    def _position_out(self, vec: tuple[float, ...]) -> Any:
        return _out(vec, self._was_vector)

    def _advance(self, position: tuple[float, ...],
                 velocity: Any = 0.0, dash_end: Any = None,
                 kind: str = "move") -> tuple[tuple[float, ...], dict[str, Any]]:
        """从 ``position`` 推进一帧，位移技能走同一套校验。"""
        if dash_end is not None:
            checked = dash_validate(position, dash_end,
                                    obstacles=self.obstacles, kind=kind)
            return _as_vec(checked["position"])[0], checked
        step = predict_step(position, velocity, self.dt,
                            obstacles=self.obstacles,
                            tick_rate=self.tick_rate)
        return _as_vec(step["position"])[0], step

    # -- 输入 ----------------------------------------------------------
    def submit_input(self, tick: Any = None, velocity: Any = 0.0,
                     *, dash_end: Any = None, kind: str = "move"
                     ) -> dict[str, Any]:
        """记录一帧本地输入并立即预测；越序 / 重复帧被丢弃。"""
        next_tick = (int(_finite(tick, self.acked + 1))
                     if tick is not None else self.acked + 1)
        if next_tick <= self.acked:
            self.events.append(audit("input_stale", tick=next_tick,
                                     acked=self.acked))
            return {"ok": False, "accepted": False, "reason": "stale_frame",
                    "tick": next_tick, "acked": self.acked}
        if any(item["tick"] == next_tick for item in self.pending):
            self.events.append(audit("input_duplicate", tick=next_tick))
            return {"ok": False, "accepted": False,
                    "reason": "duplicate_frame", "tick": next_tick,
                    "acked": self.acked}

        target, disposition = self._advance(self.predicted, velocity,
                                            dash_end, kind)
        self.predicted = target
        self.pending.append({"tick": next_tick, "velocity": velocity,
                             "dash_end": dash_end, "kind": kind})
        self.pending.sort(key=lambda item: item["tick"])
        return {"ok": True, "accepted": True, "tick": next_tick,
                "position": self._position_out(target),
                "predicted": target, "detail": disposition}

    # -- 确认 ----------------------------------------------------------
    def acknowledge(self, acked: Any) -> dict[str, Any]:
        """服务器确认到 ``acked``：只保留仍未确认的帧。"""
        info = frame_ack(self.acked, acked, pending=len(self.pending),
                         limit=self.max_replay)
        self.acked = max(self.acked, info["acked"])
        dropped = [item for item in self.pending if item["tick"] <= self.acked]
        self.pending = [item for item in self.pending
                        if item["tick"] > self.acked]
        info["confirmed_frames"] = len(dropped)
        info["remaining"] = len(self.pending)
        self.events.append(audit("ack", acked=self.acked,
                                 pending=len(self.pending)))
        return info

    def update_rtt(self, sample: Any) -> float:
        """更新 RTT 估计（带钳制与平滑）。"""
        previous = 0.0 if self.rtt is None else self.rtt
        self.rtt = rtt_smooth(previous, sample)
        return self.rtt

    # -- 校正 ----------------------------------------------------------
    def reconcile(self, snapshot: Any, *, tick: Any = None
                  ) -> dict[str, Any]:
        """用服务器快照校正预测，只重放未确认输入。"""
        authoritative = authority_state(snapshot,
                                        tick=0 if tick is None else tick)
        target = _as_vec(authoritative["position"])[0]
        previous_predicted = self.predicted
        dim = max(len(target), len(previous_predicted))
        target = _fit(target, dim)
        previous_predicted = _fit(previous_predicted, dim)
        error = _vnorm(_vsub(target, previous_predicted))

        explicit_teleport = bool(
            isinstance(snapshot, Mapping) and snapshot.get("teleport"))
        decision = snap_threshold(error, snap_distance=self.snap_distance)

        # 预测永远以权威位置为基准：回滚到快照后只重放服务器尚未确认的
        # 输入。阈值只决定表现层是忽略 / 平滑 / 吸附，绝不阻止预测回滚，
        # 否则小误差会持续累积成慢速漂移。
        if len(self.pending) > self.max_replay:
            mode = "snapshot"
        else:
            mode = "replay"

        if mode == "snapshot" or explicit_teleport:
            self.pending = []
            self.predicted = target
            replayed = 0
        else:
            replay_position = target
            replayed = 0
            for item in self.pending:
                replay_position, _ = self._advance(
                    replay_position, item["velocity"], item["dash_end"],
                    item["kind"])
                replayed += 1
            self.predicted = replay_position

        # 表现层动作：大偏差（或传送）硬吸附，小偏差直接对齐，其余平滑。
        presentation = "snap" if explicit_teleport else decision["action"]

        # 校正表现为“视觉偏移”：视觉位置 = 新预测位置 + 残余偏移。
        # 正常前进时偏移为 0，视觉紧跟预测；只有校正制造的那一段差量
        # 会被平滑衰减，因此既不会瞬移，也不会冻住角色。
        previous_offset = _fit(self._visual_offset,
                               len(previous_predicted))
        rendered = _vadd(previous_predicted, previous_offset)
        if presentation == "snap":
            self._visual_offset = (0.0,) * dim
        else:
            offset = _vsub(_fit(rendered, dim), self.predicted)
            # 修正量已在死区内：直接对齐，不制造视觉偏移。
            if presentation == "ignore":
                offset = (0.0,) * dim
            self._visual_offset = offset

        self.authoritative = target
        correction = {
            "ok": True,
            "mode": mode,
            "action": presentation,
            "error": error,
            # 回滚深度：预测被拉回权威快照的距离。
            "rollback_distance": _vnorm(_vsub(target, previous_predicted)),
            # 净校正：重放未确认输入后，预测相对校正前的实际变化量。
            "net_correction": _vnorm(_vsub(self.predicted, previous_predicted)),
            "replayed": replayed,
            "pending": len(self.pending),
            "corrected": presentation != "ignore",
            "position": target,
            "predicted": self.predicted,
            "snapshot": authoritative,
        }
        self.last_correction = correction
        self.events.append(audit(
            "reconcile", mode=mode, action=presentation, error=error,
            replayed=replayed, corrected=correction["corrected"],
            rollback_distance=correction["rollback_distance"],
            rtt=self.rtt))
        return correction

    # -- 表现 ----------------------------------------------------------
    def present(self, dt: Any = None) -> Any:
        """渲染位置：跟随预测前进，只有校正偏移被平滑衰减。

        视觉位置 = 当前预测位置 + 残余视觉偏移。偏移由 ``reconcile`` 在
        校正时产生（即“不平滑就会瞬移的那一段”），之后按指数衰减。
        正常移动时偏移为 0，视觉与预测严格同步；校正瞬间又不瞬移。
        """
        delta = self.dt if dt is None else dt
        predicted = self.predicted
        offset = _fit(self._visual_offset, len(predicted))
        decayed = rollback_present(_vadd(predicted, offset), predicted, delta,
                                   convergence_time=self.convergence_time,
                                   snap_distance=self.snap_distance)
        self._visual_offset = _vsub(_fit(decayed, len(predicted)), predicted)
        self.visual = _vadd(predicted, self._visual_offset)
        return self._position_out(self.visual)

    # -- 状态 ----------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "ok": True,
            "acked": self.acked,
            "pending": [dict(item) for item in self.pending],
            "pending_count": len(self.pending),
            "authoritative": self.authoritative,
            "predicted": self.predicted,
            "visual": self.visual,
            "rtt": self.rtt,
            "last_correction": copy.deepcopy(self.last_correction),
            "digest": _digest({"acked": self.acked,
                               "authoritative": self.authoritative,
                               "predicted": self.predicted}),
        }

    def reset(self, position: Any = (0.0, 0.0)) -> dict[str, Any]:
        """重连 / 传送后清空队列，避免旧输入残留。"""
        self.__init__(position, tick_rate=self.tick_rate,
                      max_replay=self.max_replay,
                      snap_distance=self.snap_distance,
                      convergence_time=self.convergence_time,
                      obstacles=self.obstacles)
        return self.state()

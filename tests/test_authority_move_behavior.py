"""题述缺陷场景的行为级回归测试（与契约测试并存）。

覆盖：输入帧确认、误差平滑阈值、共享碰撞查询、技能位移的服务端校验、
RTT 平滑、回滚后的表现插值，以及把它们串起来的整体校正流程。
"""

import math

import pytest

from gameplay import authority_move_reconcile as mod
from gameplay.authority_move_reconcile import (
    MAX_REPLAY_FRAMES,
    AuthorityMoveReconciler,
    dash_validate,
    determinism,
    error_smooth,
    fallback,
    frame_ack,
    predict_step,
    rollback_present,
    rtt_smooth,
    shared_collision,
    snap_threshold,
    validation,
)

DT = 1.0 / 60.0
WALL = (2.0, -1.0, 2.2, 1.0)  # 薄墙：x ∈ [2, 2.2]


# ---------------------------------------------------------------------------
# 1. 输入帧确认：重放只从未确认帧开始，确认号单调
# ---------------------------------------------------------------------------

class TestFrameAck:
    def test_replay_starts_after_acknowledged_frame(self):
        info = frame_ack(10, 12, pending=3)
        assert info["acked"] == 12
        assert info["replay_from"] == 13
        assert info["replay_count"] == 3
        assert info["replay_to"] == 15

    def test_late_ack_never_rewinds_confirmation(self):
        # 迟到的重传包携带旧确认号，确认号不能被往回拨
        info = frame_ack(10, 8, pending=2)
        assert info["stale"] is True
        assert info["rewind"] is True
        assert info["acked"] == 10
        assert info["replay_from"] == 11

    def test_duplicate_ack_is_harmless(self):
        info = frame_ack(10, 10, pending=0)
        assert info["duplicate"] is True
        assert info["acked"] == 10
        assert info["replay_count"] == 0

    def test_oversized_backlog_falls_back_to_snapshot(self):
        # 未确认帧超过上限时不再增量重放，避免一次回滚跨度过大
        info = frame_ack(10, 10, pending=MAX_REPLAY_FRAMES + 1)
        assert info["allow_replay"] is False
        assert info["require_snapshot"] is True
        assert info["replay_count"] == 0

    def test_old_input_frames_are_never_replayed(self):
        # 已确认的帧号之后才允许重放：acked=5 时第 5 帧不能再出现在重放里
        info = frame_ack(5, 5, pending=0)
        assert info["replay_from"] == 6


# ---------------------------------------------------------------------------
# 2. 误差平滑阈值：小误差忽略、大误差吸附、中间平滑且与帧率无关
# ---------------------------------------------------------------------------

class TestErrorSmoothing:
    def test_dead_zone_ignores_small_error(self):
        assert snap_threshold(0.005)["action"] == "ignore"
        assert error_smooth(1.0, 1.01) == pytest.approx(1.0)

    def test_mid_error_is_smoothed(self):
        action = snap_threshold(0.5)["action"]
        assert action == "smooth"
        value = error_smooth(0.0, 0.5, DT)
        assert 0.0 < value < 0.5

    def test_large_error_snaps(self):
        assert snap_threshold(3.0)["action"] == "snap"
        assert error_smooth(0.0, 3.0, DT) == pytest.approx(3.0)

    def test_convergence_is_frame_rate_independent(self):
        duration = 0.5
        # 关闭死区，便于与解析解对比
        dead = 0.0
        snap = 100.0
        steps_60 = int(round(duration * 60))
        value_60 = 0.0
        for _ in range(steps_60):
            value_60 = error_smooth(value_60, 1.0, 1 / 60.0,
                                    dead_zone=dead, snap_distance=snap)
        steps_120 = int(round(duration * 120))
        value_120 = 0.0
        for _ in range(steps_120):
            value_120 = error_smooth(value_120, 1.0, 1 / 120.0,
                                     dead_zone=dead, snap_distance=snap)
        # 同样的真实时间应得到同样的收敛量
        assert value_60 == pytest.approx(value_120, rel=1e-6)
        assert value_60 == pytest.approx(1.0 - math.exp(-duration / 0.08),
                                         rel=1e-6)

    def test_smoothing_never_overshoots(self):
        value = 0.0
        for _ in range(120):
            value = error_smooth(value, 1.0, DT)
            assert -1e-12 <= value <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# 3. 共享碰撞查询：两端结果一致，薄墙不穿透
# ---------------------------------------------------------------------------

class TestSharedCollision:
    def test_thin_wall_stops_sweep(self):
        result = shared_collision((0.0, 0.0), (10.0, 0.0), [WALL])
        assert result["hit"] is True
        assert result["position"][0] == pytest.approx(2.0)
        assert result["position"][1] == pytest.approx(0.0)
        assert result["normal"] == (-1.0, 0.0)

    def test_result_is_deterministic_for_same_inputs(self):
        first = shared_collision((0.0, 0.0), (10.0, 0.0), [WALL])
        second = shared_collision((0.0, 0.0), (10.0, 0.0), [WALL])
        assert determinism(first, second)["ok"] is True
        assert first == second

    def test_quantization_removes_cross_side_float_drift(self):
        # 两端算出的同一点只差 1e-12，量化后完全一致
        a = shared_collision((0.0, 0.0), (1.0 + 1e-12, 0.0), [])
        b = shared_collision((0.0, 0.0), (1.0, 0.0), [])
        assert a["position"] == b["position"]

    def test_path_clear_returns_destination(self):
        result = shared_collision((0.0, 0.0), (1.0, 1.0), [WALL])
        assert result["hit"] is False
        assert result["position"] == pytest.approx((1.0, 1.0))
        assert result["point"] is None

    def test_sweep_budget_covers_distance(self):
        result = shared_collision((0.0, 0.0), (10.0, 0.0), [], max_step=0.05)
        assert result["steps"] == pytest.approx(200, abs=1)

    def test_moving_away_from_wall_is_not_a_collision(self):
        # 站在墙面上向外走不能被判成碰撞，否则角色会永久卡在墙里
        result = shared_collision((2.2, 0.0), (3.0, 0.0), [WALL])
        assert result["hit"] is False
        assert result["position"] == pytest.approx((3.0, 0.0))

    def test_stopping_exactly_on_wall_face_is_allowed(self):
        result = shared_collision((1.0, 0.0), (2.0, 0.0), [WALL])
        assert result["hit"] is False

    def test_starting_inside_wall_hits_immediately(self):
        result = shared_collision((2.1, 0.0), (3.0, 0.0), [WALL])
        assert result["hit"] is True
        assert result["position"][0] == pytest.approx(2.1, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. 技能位移的服务端校验：瞬移不能绕过校正
# ---------------------------------------------------------------------------

class TestDashValidation:
    def test_legal_dash_is_accepted(self):
        result = dash_validate((0.0, 0.0), (5.0, 0.0), max_distance=6.0)
        assert result["ok"] is True
        assert result["reason"] == "ok"
        assert result["position"] == pytest.approx((5.0, 0.0))
        assert result["bypass"] is False

    def test_overlong_dash_is_clamped_to_valid_point(self):
        result = dash_validate((0.0, 0.0), (100.0, 0.0), max_distance=6.0)
        assert result["ok"] is False
        assert result["reason"] == "displacement_exceeded"
        assert result["clamped"] is True
        assert result["position"] == pytest.approx((6.0, 0.0))

    def test_dash_through_wall_stops_at_contact_point(self):
        result = dash_validate((0.0, 0.0), (5.0, 0.0),
                               max_distance=6.0, obstacles=[WALL])
        assert result["ok"] is False
        assert result["reason"] == "blocked"
        assert result["position"][0] == pytest.approx(2.0)

    def test_dash_respects_cooldown(self):
        result = dash_validate((0.0, 0.0), (1.0, 0.0), cooldown_ready=False)
        assert result["ok"] is False
        assert result["reason"] == "cooldown"
        assert result["position"] == pytest.approx((0.0, 0.0))

    def test_teleport_kind_is_still_validated(self):
        result = dash_validate((0.0, 0.0), (50.0, 0.0), max_distance=6.0,
                               kind="teleport")
        assert result["ok"] is False
        assert result["reason"] == "displacement_exceeded"
        assert result["bypass"] is False

    def test_validation_rejects_unacknowledged_input_frame(self):
        result = validation({"start": (0.0, 0.0), "position": (1.0, 0.0),
                             "tick": 9, "acked": 3})
        assert result["ok"] is False
        assert result["reason"] == "input_frame_not_acknowledged"
        assert result["checks"]["ack"] is False

    def test_validation_accepts_acknowledged_input_frame(self):
        result = validation({"start": (0.0, 0.0), "position": (1.0, 0.0),
                             "tick": 4, "acked": 3})
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# 5. RTT 平滑：单次抖动被钳制，估计不突变
# ---------------------------------------------------------------------------

class TestRttSmoothing:
    def test_first_sample_is_adopted(self):
        assert rtt_smooth(0.0, 100.0) == pytest.approx(100.0)

    def test_spike_is_ratio_clamped_then_smoothed(self):
        # 100 -> 1000：先钳到 200，再按 alpha=0.2 平滑 => 120
        assert rtt_smooth(100.0, 1000.0) == pytest.approx(120.0)

    def test_dip_is_ratio_clamped(self):
        # 100 -> 0：钳到 50 => 90
        assert rtt_smooth(100.0, 0.0) == pytest.approx(90.0)

    def test_estimate_never_jumps_more_than_ratio_in_one_step(self):
        previous = 100.0
        for sample in (5000.0, 1.0, 480.0, 30.0):
            updated = rtt_smooth(previous, sample)
            assert previous / 2.0 - 1e-9 <= updated <= previous * 2.0 + 1e-9
            previous = updated

    def test_invalid_sample_keeps_previous(self):
        assert rtt_smooth(120.0, float("nan")) == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# 6. 回滚后的表现插值：不瞬移、逐步追上、大偏差才吸附
# ---------------------------------------------------------------------------

class TestRollbackPresentation:
    def test_first_frame_does_not_teleport(self):
        corrected = rollback_present((0.0, 0.0), (1.0, 0.0), DT)
        assert 0.0 < corrected[0] < 1.0

    def test_visual_offset_decays_monotonically(self):
        presented = (0.0, 0.0)
        target = (1.0, 0.0)
        previous_gap = 1.0
        for _ in range(30):
            presented = rollback_present(presented, target, DT)
            gap = abs(target[0] - presented[0])
            assert gap <= previous_gap + 1e-12
            previous_gap = gap

    def test_converges_to_authoritative_position(self):
        presented = (0.0, 0.0)
        for _ in range(120):
            presented = rollback_present(presented, (1.0, 0.0), DT)
        assert presented == pytest.approx((1.0, 0.0), abs=1e-4)

    def test_large_teleport_snaps_immediately(self):
        presented = rollback_present((0.0, 0.0), (100.0, 0.0), DT,
                                     snap_distance=1.5)
        assert presented == pytest.approx((100.0, 0.0))


# ---------------------------------------------------------------------------
# 7. 整体流程：预测 — 确认 — 校正 — 表现
# ---------------------------------------------------------------------------

class TestReconciler:
    def test_reconcile_replays_only_unacknowledged_inputs(self):
        rec = AuthorityMoveReconciler((0.0, 0.0))
        for tick in range(4):                    # 4 帧输入，每帧 0.5 m
            rec.submit_input(tick, (30.0, 0.0))
        assert rec.predicted[0] == pytest.approx(2.0)

        rec.acknowledge(1)                       # 服务器只确认到第 1 帧
        correction = rec.reconcile({"position": (1.0, 0.0), "tick": 1})

        # 只重放第 2、3 帧，而不是全部 4 帧（旧实现会多走 1 m）
        assert correction["replayed"] == 2
        assert correction["predicted"][0] == pytest.approx(2.0)
        assert correction["rollback_distance"] == pytest.approx(1.0)

    def test_stale_ack_does_not_move_confirmation_backwards(self):
        rec = AuthorityMoveReconciler((0.0, 0.0))
        rec.submit_input(0, (30.0, 0.0))
        rec.submit_input(1, (30.0, 0.0))
        rec.acknowledge(1)
        info = rec.acknowledge(0)
        assert info["stale"] is True
        assert rec.acked == 1

    def test_duplicate_and_stale_inputs_are_dropped(self):
        rec = AuthorityMoveReconciler((0.0, 0.0))
        assert rec.submit_input(0, (30.0, 0.0))["ok"] is True
        assert rec.submit_input(0, (30.0, 0.0))["reason"] == "duplicate_frame"
        rec.acknowledge(0)
        assert rec.submit_input(0, (30.0, 0.0))["reason"] == "stale_frame"

    def test_dash_in_reconciler_uses_authoritative_clamp(self):
        rec = AuthorityMoveReconciler((0.0, 0.0))
        result = rec.submit_input(0, dash_end=(100.0, 0.0), kind="dash")
        # 超出最大位移的瞬移被服务端规则压回 6 m，而不是接受 100 m
        assert result["detail"]["clamped"] is True
        assert rec.predicted[0] == pytest.approx(6.0)

    def test_presentation_follows_correction_without_teleport(self):
        rec = AuthorityMoveReconciler((0.0, 0.0))
        rec.submit_input(0, (30.0, 0.0))
        rec.acknowledge(0)                       # 该输入已被服务器确认
        rec.present()                            # 视觉先追上预测 0.5

        # 服务器说实际只走到 0.3：这是 0.2 的中等误差，应当平滑而非瞬移
        rec.reconcile({"position": (0.3, 0.0), "tick": 0})
        assert rec.predicted[0] == pytest.approx(0.3)
        first = rec.present()
        # 视觉仍停留在旧位置附近，逐步收回，而不是瞬间跳到 0.3
        assert first[0] > rec.predicted[0]
        assert first[0] < 0.5

    def test_normal_movement_is_not_frozen_by_small_error(self):
        # 一直向前跑且误差始终在死区内时，视觉必须跟着预测前进
        rec = AuthorityMoveReconciler((0.0, 0.0))
        for tick in range(5):
            rec.submit_input(tick, (30.0, 0.0))
            rec.present()
        # 视觉距离预测不超过一个 tick 的位移量
        assert abs(rec.visual[0] - rec.predicted[0]) <= 0.5 + 1e-9
        assert rec.visual[0] > 1.5

    def test_oversized_backlog_switches_to_snapshot(self):
        rec = AuthorityMoveReconciler((0.0, 0.0), max_replay=2)
        for tick in range(5):
            rec.submit_input(tick, (6.0, 0.0))
        correction = rec.reconcile({"position": (0.0, 0.0), "tick": 0})
        assert correction["mode"] == "snapshot"
        assert correction["replayed"] == 0
        assert rec.pending == []

    def test_packet_loss_rolls_back_only_unacknowledged_inputs(self):
        """完整链路：丢包 → 只确认部分输入 → 回滚 → 平滑表现。"""
        rec = AuthorityMoveReconciler((0.0, 0.0))
        for tick in range(10):
            rec.submit_input(tick, (60.0, 0.0))   # 每帧 1 m
            rec.present()
        assert rec.predicted[0] == pytest.approx(10.0)

        # 服务器只收到前 6 帧（4 帧丢失），权威位置是 6 m
        rec.acknowledge(5)
        correction = rec.reconcile({"position": (6.0, 0.0), "tick": 5})

        # 只重放第 6~9 帧，终点仍是 10 m，而不是把 10 帧全部重放成 16 m
        assert correction["replayed"] == 4
        assert correction["predicted"][0] == pytest.approx(10.0)
        assert correction["rollback_distance"] == pytest.approx(4.0)
        # 未被服务器确认的输入仍然保留，等待下一轮确认
        assert rec.state()["pending_count"] == 4

        # 表现层从校正后的预测位置继续前进，偏移指数收敛
        previous_gap = abs(rec.visual[0] - rec.predicted[0])
        for _ in range(60):
            rec.present()
            gap = abs(rec.visual[0] - rec.predicted[0])
            assert gap <= previous_gap + 1e-9
            previous_gap = gap
        assert previous_gap == pytest.approx(0.0, abs=1e-3)

    def test_high_rtt_jitter_does_not_disturb_interpolation(self):
        rec = AuthorityMoveReconciler((0.0, 0.0))
        rec.update_rtt(80.0)
        previous = 80.0
        for sample in (2000.0, 5.0, 90.0):
            estimate = rec.update_rtt(sample)
            # 每步都被钳制在上一估计的 1/2 ~ 2 倍之间
            assert previous / 2.0 <= estimate <= previous * 2.0 + 1e-9
            previous = estimate
        # 单次 2000 ms 抖动没有把估计拉到 200 ms 以上
        assert rec.rtt is not None and rec.rtt < 200.0


# ---------------------------------------------------------------------------
# 支撑能力
# ---------------------------------------------------------------------------

class TestSupport:
    def test_predict_step_matches_fixed_tick_distance(self):
        one = predict_step((0.0, 0.0), (60.0, 0.0), DT)
        two = predict_step((0.0, 0.0), (60.0, 0.0), 2 * DT)
        assert one["position"] == pytest.approx((1.0, 0.0))
        assert two["position"] == pytest.approx((2.0, 0.0))

    def test_prediction_and_server_share_collision_result(self):
        predicted = predict_step((0.0, 0.0), (600.0, 0.0), 0.1,
                                 obstacles=[WALL])
        authoritative = dash_validate((0.0, 0.0), (60.0, 0.0),
                                      max_distance=60.0, obstacles=[WALL])
        assert predicted["blocked"] is True
        assert authoritative["reason"] == "blocked"
        assert predicted["position"][0] == pytest.approx(2.0)
        assert authoritative["position"][0] == pytest.approx(2.0)

    def test_predict_step_handles_huge_dt_without_blowup(self):
        # 超大 dt 不能构造巨量子步列表；解析扫掠保证总位移仍然精确
        result = predict_step((0.0, 0.0), (1.0, 0.0), 1e6)
        assert result["position"] == pytest.approx((1e6, 0.0))
        assert result["ticks"] <= 64

    def test_fallback_selects_snapshot_when_backlog_too_large(self):
        result = fallback({"pos": 1}, pending=MAX_REPLAY_FRAMES + 5,
                          snapshot={"pos": 2})
        assert result["mode"] == "snapshot"
        assert result["require_snapshot"] is True
        assert result["state"] == {"pos": 2}

    def test_fallback_replays_within_window(self):
        result = fallback({"pos": 1}, pending=3, error=0.01)
        assert result["mode"] == "replay"

    def test_lag_comp_is_half_rtt_and_capped(self):
        assert mod.lag_comp(100.0) == pytest.approx(0.05)
        # unit="s" 时 100 秒的一半远超上限，仍被夹到 0.25 s
        assert mod.lag_comp(100.0, unit="s") == pytest.approx(0.25)
        assert mod.lag_comp(0.2, max_comp=10.0, unit="s") == pytest.approx(0.1)
        assert mod.lag_comp(10_000.0) == pytest.approx(0.25)

    def test_authority_state_is_a_deep_copy(self):
        source = {"position": [1.0, 2.0], "velocity": [0.0, 0.0], "tick": 7}
        snapshot = mod.authority_state(source)
        source["position"][0] = 99.0
        assert snapshot["position"] == [1.0, 2.0]
        assert snapshot["tick"] == 7

    def test_metrics_summarize_rollback_and_rtt(self):
        events = [
            {"mode": "smooth", "rollback_distance": 1.0, "rtt": 100.0},
            {"mode": "snap", "rollback_distance": 3.0, "rtt": 120.0},
            {"mode": "ignore", "distance": 0.0},
        ]
        summary = mod.metrics(events)
        assert summary["smooths"] == 1
        assert summary["snaps"] == 1
        assert summary["ignored"] == 1
        assert summary["rollback_distance_max"] == pytest.approx(3.0)
        assert summary["rtt_mean"] == pytest.approx(110.0)

    def test_determinism_detects_divergent_results(self):
        assert determinism({"a": 1}, {"a": 1})["ok"] is True
        assert determinism({"a": 1}, {"a": 2})["ok"] is False
        assert determinism(lambda: predict_step((0.0, 0.0), (1.0, 0.0)),
                           repeat=3)["ok"] is True

    def test_audit_records_are_monotonic(self):
        first = mod.audit("test_event", detail=1)
        second = mod.audit("test_event", detail=2)
        assert second["seq"] == first["seq"] + 1
        assert first["digest"] != second["digest"]

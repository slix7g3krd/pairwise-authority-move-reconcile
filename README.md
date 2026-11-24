# pairwise-authority-move-reconcile

## 领域

游戏、图形与多媒体

## 主题

权威移动校正

## 说明

输入帧确认、误差平滑与服务端校验。

初始状态下 `gameplay/authority_move_reconcile.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 实现要点

针对题述六项缺陷，`gameplay/authority_move_reconcile.py` 的修复策略：

1. **输入帧确认** —— `frame_ack()` 保证确认号单调不减（迟到的重传包不会
   把 `acked` 往回拨），重放窗口恒为 `acked + 1` 起；已确认帧绝不重放，
   否则同一段位移会被重复应用、回滚距离被放大。未确认帧超过
   `MAX_REPLAY_FRAMES`（90 帧 ≈ 1.5 s）时返回 `require_snapshot`，改走
   全量快照重同步。
2. **误差平滑阈值** —— `snap_threshold()` 把误差分成 `ignore` /
   `smooth` / `snap` 三档；`error_smooth()` 用
   `alpha = 1 - exp(-dt / τ)` 做指数收敛，收敛量与帧率无关（60 Hz 与
   120 Hz 经过相同时间得到相同结果），小误差不产生漂移，大误差（传送）
   直接吸附。
3. **共享碰撞查询** —— `shared_collision()` 是客户端预测与服务器校验共用
   的同一套例程：输入先按 `POSITION_QUANTUM` 量化，再做解析线段扫掠，
   两端结果逐位一致；整段一次求交而非逐帧采点，低帧率下也不会穿过薄墙。
4. **技能位移的服务端校验** —— `dash_validate()` / `validation()` 让冲刺、
   瞬移与普通移动走同一条校验链：输入帧必须已确认、位移不得超过
   `max_distance`、冷却必须就绪、路径必须无碰撞。越界时返回合法的最大
   位移点，撞墙时返回接触点；`bypass` 恒为 `False`。
5. **RTT 平滑** —— `rtt_smooth()` 先将新样本钳制在上一估计值的
   `1/2 ~ 2` 倍之间，再做 EMA，单次抖动不会把 RTT 估计拉飞，插值延迟
   因此保持稳定。
6. **回滚后的表现插值** —— `rollback_present()` 让渲染位置携带一个按
   指数衰减的视觉偏移追上权威位置：校正瞬间不瞬移，正常移动时视觉与
   预测严格同步（偏移为 0），只有超过阈值的大偏差才硬吸附。

`AuthorityMoveReconciler` 把上述能力串成完整流水线：
`submit_input()` 预测 → `acknowledge()` 清理已确认输入 →
`reconcile()` 回滚到权威位置并只重放未确认帧 → `present()` 平滑表现。
配套的 `predict_step()`（固定服务器 tick 步进）、`fallback()`、
`metrics()`、`audit()`、`authority_state()` 与 `determinism()` 提供支撑。

测试：`tests/test_authority_move_reconcile.py` 为契约测试，
`tests/test_authority_move_behavior.py` 为题述缺陷场景的行为级回归测试。

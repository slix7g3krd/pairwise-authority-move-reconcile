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

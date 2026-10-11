# Countdown-RLVR｜可验证奖励的数学推理强化学习

面向 **Countdown 数学推理任务**的可验证奖励强化学习（RLVR）系统：
模型用给定数字、每个恰好一次、通过加减乘除凑出目标数，奖励完全由规则判定
（无需奖励模型），用 GRPO / PPO 训练小模型自举推理能力。

> 本项目复用开源复现工作的代码与 **veRL** 训练框架（上游在 Countdown 任务上
> 复现了 DeepSeek-R1-Zero 的思路）；**以下内容是在我们复现基础上的改进**。

---

## 我们做了什么

1. **安全分层奖励验证器**：用受限 **AST** 解析 + **Fraction** 精确运算替换不安全的
   `eval()`，并给出四档奖励——无标签 0 / 非法表达式 0.1 / 合法且每个数字用一次 0.2 /
   正确 1.0，同时输出格式、语法、数字、合法、正确 6 个二值诊断指标给 W&B。
2. **GRPO 组内优势 + 训练信号监控**：按 prompt 分组计算相对优势 `(score−mean)/(std+eps)`，
   并监控 **零方差组比例**与**有效优势 Token 比例**，刻画"有效信号是否枯竭"。
3. **零方差组自适应重采样（DAPO 式）**：只对奖励全同（无梯度贡献）的组重新采样一组
   （组大小不变、只采一次），用**重采样恢复率**衡量效果。
4. **修复伪优势数值 bug**：定位到全等奖励在浮点减法后残留极小值、被 `/(std+eps)`
   放大成"伪非零优势"；对 `std ≤ 阈值` 的组显式将优势置 0。
5. **PPO/GAE 对照**：统一 Rollout 预算下比较 GRPO 与 PPO，并用 **explained variance**
   诊断 critic 的价值预测能力。
6. **固定协议评测**：固定验证集与随机流，报告 pass@1 / pass@5、格式率与合法表达式率。

---

## 结果（2×RTX 4090D 48GB，Qwen2.5-1.5B-Instruct）

| 实验 | pass@1 | pass@5 |
|---|---:|---:|
| 原始模型 | 5.08% | 16.80% |
| 标准 GRPO（100 步） | **29.69%** | 31.84% |
| Adaptive GRPO（100 步） | 29.30% | 32.23% |
| PPO（60 步） | 23.63% | 37.30% |

- Adaptive 重采样把 100 步平均零方差组比例从 62.75% 降到 57.08%、有效优势 Token 比例
  从 34.65% 提到 41.85%，但恢复率后期从 62% 衰减到 2.47%，**延缓了信号枯竭、未稳定提升精度**。
- GRPO 更快提高单次命中率（pass@1），PPO 候选多样性更强（pass@5）；PPO 代价是独立 critic
  与约 2.24 倍单步耗时。

---

## 复现

- 训练脚本：`scripts/train_countdown_grpo_2x4090d.sh`、`scripts/train_countdown_ppo_2x4090d.sh`
- 实验手册（配置/协议/结果）：`docs/EXPERIMENT_GUIDE_GRPO_2X4090D.md`
- 奖励实现：`verl/utils/reward_score/countdown.py`
- 优势/重采样实现：`verl/trainer/ppo/core_algos.py`、`verl/trainer/ppo/ray_trainer.py`

```bash
# 数据（20,000 训练 / 512 验证）
python examples/data_preprocess/countdown.py --local_dir <data_dir> \
  --template_type qwen-instruct --train_size 20000 --test_size 512 --seed 42

# 标准 GRPO（100 步）
bash scripts/train_countdown_grpo_2x4090d.sh
# Adaptive GRPO（开启零方差组重采样）
RESAMPLE_ZERO_VARIANCE=true bash scripts/train_countdown_grpo_2x4090d.sh
# PPO 基线
bash scripts/train_countdown_ppo_2x4090d.sh
```

---

## 致谢与许可

- 复现基线来自开源工作（TinyZero），训练框架为 veRL。
- 保留上游 `LICENSE` 与 `Notice.txt`（Apache-2.0）。
- 本仓库的改进部分见上文「我们做了什么」。

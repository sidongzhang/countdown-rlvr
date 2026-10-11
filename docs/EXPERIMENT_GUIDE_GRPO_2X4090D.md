# GRPO、Adaptive GRPO 与 PPO 实验手册（2×RTX 4090D 48GB）

本文档记录实验操作、配置和真实运行结果。Pilot与正式实验必须分开标注，任何性能与效果数字都只能从对应日志中填写。

## 1. 实验目标

本项目基于TinyZero内置的veRL训练流程完成以下改造：

1. 使用GRPO组内相对优势估计，不启动独立Critic模型；
2. 使用安全AST解析器构建分层Countdown规则奖励；
3. 检测组内奖励零方差的Prompt，并仅对这些Prompt重新rollout一次；
4. 记录reward、策略熵、KL、组内奖励方差、有效优势比例和重采样恢复率；
5. 在固定验证集上记录pass@1、pass@5、格式正确率和合法表达式比例。
6. 在相同Prompt batch、rollout数量和验证协议下补充PPO基线，比较Critic成本、单次命中率与多次采样覆盖率。

首轮实验固定使用 Qwen/Qwen2.5-1.5B-Instruct。0.5B只作为后续规模消融，不要同时修改模型、奖励和采样策略。

首版不启用LoRA。当前TinyZero内置的veRL版本没有完整打通LoRA Actor与vLLM rollout之间的权重同步；1.5B模型使用双卡FSDP可以控制在目标硬件范围内，也能避免为了简历技术栈引入未经验证的组件。

## 2. 默认硬件与训练配置

| 配置项 | 默认值 | 设计原因 |
|---|---:|---|
| GPU | 2×RTX 4090D 48GB | 单机双卡 |
| 训练精度 | BF16 | 4090D支持且比FP16稳定 |
| Actor并行 | FSDP，2卡 | 分片参数、梯度和优化器状态 |
| Rollout并行 | vLLM TP=1、DP=2 | 1.5B模型无需TP=2，两卡分别生成可提高吞吐 |
| Prompt batch | 64 | 每张卡32个Prompt |
| Group size | 8 | 每个Prompt生成8条回答 |
| 最大Prompt长度 | 256 | Countdown提示词足够 |
| 最大Response长度 | 1024 | 利用48GB显存保留完整推理过程 |
| Actor mini-batch | 32个Prompt | 与旧版veRL的group-size换算逻辑匹配 |
| Actor micro-batch | 4个Prompt | 每卡换算后为16条response |
| vLLM显存比例 | 0.40 | Pilot在0.45下峰值过于接近显存上限，正式运行统一降低 |
| Reference参数 | CPU offload | 降低常驻显存 |
| 学习率 | 1e-6 | GRPO小模型后训练的保守起点 |
| KL系数 | 0.001 | 限制策略偏离初始模型 |
| 重采样次数 | 1 | 控制额外rollout成本 |
| 正式首阶段训练步数 | 100 | 先观察完整学习曲线，避免直接承担500步成本与坍缩风险 |
| 验证/保存频率 | 每20步 | 记录20/40/60/80/100步曲线并保留对应Checkpoint |
| 训练/验证随机种子 | 42/42 | 固定训练batch顺序，并让各Checkpoint复用相同验证采样随机流 |

注意：该仓库是TinyZero固定的旧版veRL代码，batch参数会在worker中根据world size和group size重新换算。因此脚本里的mini/micro batch单位按“Prompt数”理解，不要直接照搬新版veRL配置。

Group size和pass@k相互独立：训练时group size=8表示每个Prompt生成8条回答用于计算相对优势；验证时pass@5表示每道题独立采样5次，只要至少一次正确就记为通过。训练和验证不要求使用相同的采样数。

PPO对照仍使用64个Prompt、每题8条rollout和相同Actor学习率，但将优势估计改为GAE，并加载一份独立Qwen2.5-1.5B Critic。Critic参数和优化器状态使用CPU Offload，vLLM显存比例降为0.30；这是为了在双卡48GB上容纳Actor、Reference、Rollout和Critic，不代表PPO与GRPO显存配置完全相同。

## 3. 环境准备

建议在项目根目录执行：

    cd /root/TinyZero
    conda create -n tinyzero-grpo python=3.9 -y
    conda activate tinyzero-grpo
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip install vllm==0.6.3 ray==2.42.1 transformers==4.47.1 tensordict==0.5.0
    pip install datasets==3.3.2 wandb==0.19.7 hydra-core==1.3.2 codetiming==1.4.0
    pip install flash-attn==2.7.4.post1 --no-build-isolation
    pip uninstall -y uvloop
    pip install -e .

这里卸载`uvloop`是有意为之：Ray 2.42.1的worker启动流程与`uvloop 0.22.x`组合时，可能在导入dashboard state API阶段报`There is no current event loop`。本实验不启动vLLM HTTP服务，不依赖`uvloop`；移除后Ray会使用Python标准asyncio事件循环。

当前代码准备环境使用过的主要版本为：

    torch==2.4.0
    vllm==0.6.3
    ray==2.42.1
    transformers==4.47.1
    tensordict==0.5.0
    datasets==3.3.2
    wandb==0.19.7
    flash_attn==2.7.4.post1

启动付费服务器后先记录硬件信息：

    mkdir -p outputs
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    python --version
    python -m pip list --format=freeze > outputs/environment.txt

## 4. 准备模型

训练脚本默认接受Hugging Face模型ID并自动下载：Qwen/Qwen2.5-1.5B-Instruct。

为了避免每次租用服务器重复下载，可提前保存到数据盘：

    hf download Qwen/Qwen2.5-1.5B-Instruct \
      --local-dir /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct

之后启动脚本时设置：

    export MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct

## 5. 准备数据

首轮使用20,000条训练数据和512条固定验证数据：

    cd /root/TinyZero
    python examples/data_preprocess/countdown.py \
      --local_dir /root/autodl-tmp/tinyzero-grpo-data \
      --template_type qwen-instruct \
      --train_size 20000 \
      --test_size 512 \
      --seed 42

确认文件存在：

    ls -lh /root/autodl-tmp/tinyzero-grpo-data

所有实验必须复用同一组train.parquet和test.parquet，否则对比结果无效。

训练脚本还固定`TRAIN_SEED=42`与`VAL_SEED=42`。前者传给训练DataLoader以固定shuffle顺序；后者在每次完整验证开始时重置vLLM请求采样种子，使原始模型、标准GRPO和Adaptive GRPO在相同验证题目及相同采样随机流下比较。训练rollout不为每个请求重复指定固定seed，否则对同一个Prompt做自适应重采样时可能复现完全相同的回答组。

## 6. WandB配置

默认脚本使用离线模式，避免网络异常中断训练：

    export WANDB_MODE=offline

训练结束后再同步：

    wandb sync /root/TinyZero/wandb/offline-run-*

如果网络稳定，也可以在运行前设置：

    wandb login
    export WANDB_MODE=online

不要把WandB API Key写入脚本或提交到Git。

## 7. 正式实验前的最小运行

代码准备阶段不执行本节。服务器计费开始后，先用2次参数更新确认显存和数据流：

    cd /root/TinyZero
    TOTAL_STEPS=2 \
    EXPERIMENT_NAME=smoke_grpo_adaptive \
    WANDB_MODE=offline \
    bash scripts/train_countdown_grpo_2x4090d.sh \
      trainer.val_before_train=false \
      trainer.test_freq=-1 \
      trainer.save_freq=-1

需要确认日志中出现：

- 两个Ray GPU worker；
- adv_estimator=grpo；
- zero_variance_group_ratio；
- resampled_group_count和resample_recovery_rate；
- actor/entropy_loss、actor/kl_loss和actor/grad_norm；
- 两次参数更新均完成且没有CUDA OOM；
- 训练结束后完成一次pass@1/pass@5验证。

旧训练循环曾在更新后先递增计数再判断退出，使`TOTAL_STEPS=2`只执行1次参数更新。首次冒烟运行发现该问题后，退出条件已调整为在当前更新完成后判断，因此现在`TOTAL_STEPS=N`严格对应N次Actor更新。

### 2026-09-08冒烟运行记录

修复环境与Step计数后，`smoke_grpo_adaptive_v2`完整执行2次Actor更新和一次最终验证，无CUDA OOM、NaN或Ray worker退出。

| 指标 | Step 1 | Step 2 | 两步汇总/均值 |
|---|---:|---:|---:|
| 初始零方差组 | 3/64（4.69%） | 2/64（3.13%） | 5/128（3.91%） |
| 重采样后零方差组 | 1/64（1.56%） | 0/64（0%） | 1/128（0.78%） |
| 恢复组数 | 2/3 | 2/2 | 4/5（80%） |
| 有效优势Token比例 | 86.0% | 94.0% | 90.0% |
| Step耗时 | 60.752秒 | 73.692秒 | 67.222秒 |
| 初始Rollout耗时 | 26.122秒 | 31.247秒 | 28.685秒 |
| 重采样耗时 | 12.532秒 | 19.322秒 | 15.927秒 |
| Actor更新耗时 | 17.623秒 | 18.634秒 | 18.129秒 |

最终验证（512题、每题5次采样）：pass@1=4.10%，pass@5=15.82%，Format rate=80.04%，Valid expression rate=21.41%。这是两步冒烟数据，只用于证明训练与评估链路可运行，不能作为原始模型基线或正式算法效果结论。轮询观察到的最高显存约33.3GB/卡，但正式实验仍需用5秒采样文件统计真实峰值。

20步标准GRPO Pilot首次运行到第16步时主动停止：随着模型快速学会格式，大量组的8条回答都获得0.1奖励，零方差组比例升至32.8%；但`effective_advantage_token_ratio`错误显示100%。定位发现float32的0.1在减去组均值后可能残留约1e-8，继续除以1e-6会产生伪非零优势。优势计算现已对`std <= zero_variance_threshold`的组显式写入零，避免无信息组产生虚假策略梯度。停止的首次Pilot只保留为排障记录，不纳入实验结果。

修复后的20步标准GRPO Pilot（`qwen2_5_1_5b_grpo_baseline_pilot20_v2`）完整结束，无OOM或NaN。20步平均零方差组比例18.52%，平均有效优势Token比例78.40%；后5步两者分别为47.50%和50.96%，第20步达到62.50%和34.50%，说明模型学会统一输出合法表达式后，无信息组迅速增加。训练正确率由前5步平均2.84%升至后5步8.92%，平均Step耗时29.062秒，20步纯训练耗时581.236秒。最终验证pass@1=13.28%（68/512）、pass@5=30.47%（156/512）、Format rate=100%、Valid expression rate=98.48%（2521/2560）。相对原始模型分别提高8.20和13.67个百分点，但这只是20步Pilot，不填写为100步或500步正式结果。

修复后的20步Adaptive GRPO Pilot（`qwen2_5_1_5b_grpo_adaptive_pilot20_v2`）也完整结束。20步共检测到225个初始零方差组，重采样恢复97个，聚合恢复率43.11%；初始和最终零方差组比例分别为17.58%和10.00%。平均有效优势Token比例为86.43%，后5步为63.48%，分别比标准GRPO高8.03和12.52个百分点。最终验证pass@1=14.26%（73/512）、pass@5=31.84%（163/512）、Format rate=100%、Valid expression rate=98.95%（2533/2560）。相较标准20步Pilot，pass@1和pass@5分别提高0.98和1.37个百分点。

Adaptive平均Step耗时38.086秒，20步纯训练耗时761.723秒，比标准GRPO增加31.05%；其中初始Rollout、重采样、Reference前向和Actor更新平均分别为15.508、7.887、3.146和11.501秒。第4步出现一次`grad_norm=13.088`，被1.0阈值裁剪；其余19步均值0.339，中位数0.355，之后未出现NaN、OOM或连续异常。第20步初始零方差组升至65.63%，一次重采样只恢复5/42，最终有效优势Token降至41.10%，同时策略熵降至0.400，说明继续长跑需要密切观察策略收敛，而不是只看格式率。

5秒间隔GPU监控捕获到标准组GPU0/GPU1峰值48,124/48,208MiB，Adaptive组为48,388/38,552MiB。GPU0距离每卡49,140MiB容量仅约752MiB。两次Pilot均未OOM，但正式运行统一将`GPU_MEMORY_UTILIZATION`降至0.40，保证对照公平并增加长跑余量。

### Pilot直接对比

| 指标 | 标准GRPO 20步 | Adaptive GRPO 20步 | 差值/开销 |
|---|---:|---:|---:|
| pass@1 | 13.28%（68/512） | 14.26%（73/512） | +0.98个百分点 |
| pass@5 | 30.47%（156/512） | 31.84%（163/512） | +1.37个百分点 |
| 合法表达式率 | 98.48% | 98.95% | +0.47个百分点 |
| 平均最终零方差组比例 | 18.52% | 10.00% | -8.52个百分点 |
| 平均有效优势Token比例 | 78.40% | 86.43% | +8.03个百分点 |
| 后5步有效优势Token比例 | 50.96% | 63.48% | +12.52个百分点 |
| 平均Step耗时 | 29.062秒 | 38.086秒 | +31.05% |

以上都是20步Pilot结果，只用于决定正式实验配置，不能写成100步或500步正式结论。

原始日志与显存记录分别保存在：

- `outputs/qwen2_5_1_5b_grpo_baseline_pilot20_v2.log`
- `outputs/qwen2_5_1_5b_grpo_baseline_pilot20_v2_gpu.csv`
- `outputs/qwen2_5_1_5b_grpo_adaptive_pilot20_v2.log`
- `outputs/qwen2_5_1_5b_grpo_adaptive_pilot20_v2_gpu.csv`

## 8. 实验顺序

### 实验A：原始模型能力

只运行pass@1/pass@5验证，不更新模型：

    EXPERIMENT_NAME=base_model_eval \
    TOTAL_STEPS=1 \
    bash scripts/train_countdown_grpo_2x4090d.sh \
      trainer.val_only=true \
      trainer.save_freq=-1 \
      trainer.test_freq=-1

2026-09-08已完成：固定512题、`VAL_SEED=42`、每题5次采样时，原始模型pass@1为5.08%（26/512），pass@5为16.80%（86/512），Format rate为75.86%（1942/2560），Valid expression rate为21.48%（550/2560），平均分层奖励为0.1327。运行全程未更新参数。

### 实验B：标准GRPO基线

关闭零方差组重采样，其余参数与改进组完全相同。Pilot显示20步时格式与合法表达式已接近饱和，同时后期零方差和熵下降明显，因此正式首阶段固定运行100步，而不是直接运行500步；两组每20步验证且都完整跑满100步，不依据中途验证结果提前停止。标准组每20步保存一次：

    EXPERIMENT_NAME=qwen2_5_1_5b_grpo_baseline \
    RESAMPLE_ZERO_VARIANCE=false \
    TOTAL_STEPS=100 \
    TEST_FREQ=20 \
    SAVE_FREQ=20 \
    GPU_MEMORY_UTILIZATION=0.40 \
    bash scripts/train_countdown_grpo_2x4090d.sh

2026-09-08已完成，实际实验名为`qwen2_5_1_5b_grpo_baseline_formal100`。训练、验证与5个Checkpoint均正常完成，无OOM或NaN。

### 实验C：Adaptive GRPO

开启一次零方差组重采样。实际运行时数据盘只剩14GB，无法再保存5个约6.7GB的Checkpoint；为保留标准组全部Checkpoint且不擅自删除实验产物，Adaptive只保存第100步，验证仍保持每20步一次。保存频率不参与梯度计算，不影响算法对照：

    EXPERIMENT_NAME=qwen2_5_1_5b_grpo_adaptive \
    RESAMPLE_ZERO_VARIANCE=true \
    MAX_RESAMPLE_ATTEMPTS=1 \
    TOTAL_STEPS=100 \
    TEST_FREQ=20 \
    SAVE_FREQ=100 \
    GPU_MEMORY_UTILIZATION=0.40 \
    bash scripts/train_countdown_grpo_2x4090d.sh

2026-09-08已完成，实际实验名为`qwen2_5_1_5b_grpo_adaptive_formal100`。100步与最终Checkpoint均正常完成，无OOM或NaN。

### 实验D：PPO算法基线

PPO使用与GRPO相同的模型、训练/验证数据、Prompt batch=64、每个Prompt采样8条回答、训练种子和验证参数。区别是使用GAE与独立Critic，并关闭Checkpoint保存以避免填满数据盘：

    EXPERIMENT_NAME=qwen2_5_1_5b_ppo_formal100 \
    TOTAL_STEPS=100 \
    TEST_FREQ=20 \
    SAVE_FREQ=-1 \
    GPU_MEMORY_UTILIZATION=0.30 \
    bash scripts/train_countdown_ppo_2x4090d.sh

2026-09-08启动后，根据运行成本决定以第60步验证作为正式截点。第60步验证已完整写入日志；轮询截停存在延迟，随后又完成了第61、62次参数更新，并在第63步开始前终止。第61、62步没有验证结果且不纳入任何正式统计。由于本组未保存Checkpoint，只保留第60步的验证与前60步训练日志，不能把终止时内存中的模型称为“第60步Checkpoint”。

### 实验E：重采样次数消融（可选）

只在实验C确实提高有效优势比例后运行：

    EXPERIMENT_NAME=qwen2_5_1_5b_grpo_resample2 \
    RESAMPLE_ZERO_VARIANCE=true \
    MAX_RESAMPLE_ATTEMPTS=2 \
    TOTAL_STEPS=100 \
    TEST_FREQ=20 \
    SAVE_FREQ=20 \
    GPU_MEMORY_UTILIZATION=0.40 \
    bash scripts/train_countdown_grpo_2x4090d.sh

100步结束后比较20/40/60/80/100步学习曲线。正式结果显示80到100步没有形成稳定的Adaptive精度优势，且后20步恢复率只剩2.47%，因此当前不继续300或500步，也不运行两次重采样消融。旧版本veRL训练器没有在本项目中验证过断点续训，不能把“从100步继续”写进当前运行方案。

## 9. 必须记录的指标

### 效果指标

| 实验 | pass@1 | pass@5 | Format rate | Valid expression rate |
|---|---:|---:|---:|---:|
| 原始模型 | 5.08% | 16.80% | 75.86% | 21.48% |
| 标准GRPO（100步） | 29.69%（152/512） | 31.84%（163/512） | 100% | 约99.9% |
| Adaptive GRPO（100步） | 29.30%（150/512） | 32.23%（165/512） | 100% | 100% |
| PPO（60步） | 23.63%（121/512） | 37.30%（191/512） | 100% | 97.0% |

GRPO两组报告第100步，PPO报告第60步，因此上表只能展示各实验截点，不能直接用于算法公平排名。第100步GRPO与第60步PPO的pass计数均由固定512题和控制台三位小数唯一还原；离线W&B完整保留了GRPO 0至80步的未四舍五入值，但本次旧版logger退出前未把末尾history完整写入W&B二进制文件，因此第100步合法表达式率只按控制台精度报告。

### 正式验证学习曲线

| Step | 标准pass@1 | Adaptive pass@1 | 标准pass@5 | Adaptive pass@5 |
|---:|---:|---:|---:|---:|
| 0 | 5.08% | 5.08% | 16.80% | 16.80% |
| 20 | 12.30% | 14.84% | 27.34% | 32.23% |
| 40 | 22.07% | 22.46% | 31.25% | 34.57% |
| 60 | 26.37% | 24.02% | 32.03% | 30.86% |
| 80 | 27.93% | 26.37% | 31.25% | 30.47% |
| 100 | 29.69% | 29.30% | 31.84% | 32.23% |

Adaptive在20和40步的pass@5分别领先4.88和3.32个百分点，但优势没有持续；100步时pass@1低0.39个百分点、pass@5高0.39个百分点，只相差2道题，不能解读为稳定精度提升。

### 60步PPO公平对比

三组在前60步都使用64个Prompt/步、8条Response/Prompt，即各消费30,720条训练Response；验证集、验证采样数和随机流也相同。

| Step | 标准GRPO pass@1/pass@5 | Adaptive GRPO pass@1/pass@5 | PPO pass@1/pass@5 |
|---:|---:|---:|---:|
| 0 | 5.08% / 16.80% | 5.08% / 16.80% | 5.08% / 16.80% |
| 20 | 12.30% / 27.34% | 14.84% / 32.23% | 7.62% / 23.44% |
| 40 | 22.07% / 31.25% | 22.46% / 34.57% | 16.21% / 34.38% |
| 60 | 26.37% / 32.03% | 24.02% / 30.86% | 23.63% / 37.30% |

第60步PPO的pass@1比标准GRPO低2.73个百分点（少14题），与Adaptive GRPO仅低0.39个百分点（少2题）；pass@5则比两者分别高5.27和6.45个百分点（多27题和33题）。这说明当前PPO策略的单次采样可靠性不占优，但五次采样的候选覆盖率更高。不能仅凭pass@5把PPO描述为“效果更好”，因为它同时具有更高策略熵、更低单次命中率和更高计算成本。

按512题计算的95% Wilson区间中，标准GRPO与PPO的pass@1分别为22.74%–30.35%和20.16%–27.50%，pass@5分别为28.14%–36.19%和33.22%–41.57%，区间均有重叠。由于三组使用同一批题，严格差异检验应基于逐题成败做配对检验；当前日志没有保存完整逐题布尔结果，又只有单一训练种子，因此不宣称差异具有统计显著性。

### 训练稳定性

| 实验 | 初始零方差组比例 | 最终零方差组比例 | 重采样恢复率 | 有效优势Token比例 | 最终策略熵 |
|---|---:|---:|---:|---:|---:|
| 标准GRPO（100步均值） | 不适用 | 64.20% | 不适用 | 34.65% | 0.184 |
| Adaptive GRPO（100步均值） | 62.75% | 57.08% | 9.04%（363/4016） | 41.85% | 0.176 |

标准GRPO关闭重采样后，grpo/zero_variance_group_ratio是最终比例；Adaptive GRPO同时记录initial和final指标。

| Adaptive阶段 | 重采样组数 | 恢复组数 | 聚合恢复率 | 平均最终零方差 | 平均有效优势Token |
|---|---:|---:|---:|---:|---:|
| Step 1–20 | 149 | 93 | 62.42% | 4.38% | 92.10% |
| Step 21–40 | 739 | 110 | 14.88% | 49.14% | 49.96% |
| Step 41–60 | 906 | 80 | 8.83% | 64.54% | 35.01% |
| Step 61–80 | 1088 | 52 | 4.78% | 80.94% | 18.71% |
| Step 81–100 | 1134 | 28 | 2.47% | 86.40% | 13.49% |

### 性能与资源

| 实验 | 峰值显存/GPU | 平均step耗时 | rollout耗时 | tokens/s | 总训练时长 |
|---|---:|---:|---:|---:|---:|
| 标准GRPO | 48,508/48,456MiB | 21.006秒 | 9.308秒 | 未单独记录 | 2,100.612秒 |
| Adaptive GRPO | 48,452/47,872MiB | 28.153秒 | 9.474+6.927秒 | 未单独记录 | 2,815.295秒 |
| PPO（前60步） | 48,350/47,650MiB | 53.113秒 | 15.181秒 | 未单独记录 | 3,186.772秒 |

平均Step和总训练时长已扣除周期验证与Checkpoint保存时间。Adaptive纯训练平均每步增加34.02%；标准组保存5次、Adaptive只保存1次，因此不使用包含保存时间的总墙钟时间做算法效率比较。

为做同一步数比较，标准GRPO、Adaptive GRPO和PPO前60步的纯训练平均耗时分别为23.752、29.759和53.113秒。PPO比标准GRPO慢123.6%（约2.24倍），比Adaptive GRPO慢78.5%（约1.78倍）；主要额外成本来自Critic更新（平均21.581秒/步）和Value前向（3.110秒/步）。PPO显存峰值来自5秒间隔监控，并非瞬时理论峰值。

使用独立终端每5秒记录一次GPU状态：

    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
      --format=csv \
      -l 5 > outputs/gpu_metrics.csv

## 10. OOM处理顺序

一次只调整一个变量，并保存修改后的命令。

1. 将GPU_MEMORY_UTILIZATION从0.40降至0.35；
2. 将MAX_RESPONSE_LENGTH从1024降至768；
3. 将TRAIN_BATCH_SIZE从64降至32，同时通过命令行把Actor mini-batch从32改为16；
4. 开启actor_rollout_ref.actor.fsdp_config.optimizer_offload=true；
5. 最后才将GROUP_SIZE从8降至4。

示例：

    GPU_MEMORY_UTILIZATION=0.35 \
    MAX_RESPONSE_LENGTH=768 \
    bash scripts/train_countdown_grpo_2x4090d.sh

如果把TRAIN_BATCH_SIZE改成32，运行时同时追加：

    actor_rollout_ref.actor.ppo_mini_batch_size=16

不要首先把rollout tensor parallel改为2。对于1.5B模型，TP=1可以让两张GPU形成两个数据并行rollout worker；改成TP=2通常会降低本实验的生成吞吐。

## 11. 异常判定

- zero_variance_group_ratio长期接近1：模型几乎拿不到有区分度的奖励，先检查奖励分布和输出格式；
- 策略熵快速接近0：可能发生策略坍缩，应减小学习率或增大KL系数；
- KL持续快速上升：将kl_loss_coef从0.001提高到0.002；
- response_length/clip_ratio过高：大量回答被1024 token截断，需要检查Prompt或适度提高长度；
- Format rate很高但pass@1不提升：模型可能只学会格式，应检查valid expression rate和正确奖励；
- 重采样恢复率很低：额外rollout没有产生有效信号，不应继续增加重采样次数。

## 12. 100步正式实验结论

> 在固定512题验证集、相同初始化与生成参数下，标准GRPO和Adaptive GRPO训练100步后的pass@1分别为29.69%和29.30%，pass@5分别为31.84%和32.23%。Adaptive将100步平均零方差组比例从重采样前62.75%降至57.08%，平均有效优势Token比例相较标准GRPO由34.65%提高到41.85%，但平均Step耗时增加34.02%。重采样恢复率由前20步62.42%下降到后20步2.47%，因此额外训练信号没有转化为稳定的最终精度提升。

这个结果支持“动态重采样能延缓有效信号枯竭”，但不支持“当前一次无差别重采样能稳定提高最终准确率”。下一版更合理的方向是根据在线恢复率、训练阶段或Prompt难度控制重采样预算，在恢复率过低时自动关闭，而不是增加重采样次数。不要把Reward收敛或有效优势增加直接等同于推理能力提升。

## 13. PPO 60步对照结论

PPO的验证曲线为：Step 0的5.08%/16.80%，Step 20的7.62%/23.44%，Step 40的16.21%/34.38%，以及Step 60的23.63%/37.30%（均为pass@1/pass@5）。第60步相对原始模型分别提高18.55和20.51个百分点。

PPO的Critic在第1步由随机初始化的Value Head产生`vf_loss=3.702`、`vf_explained_var=-214.036`和裁剪前`grad_norm=19.288`。初始化异常值消退后，Step 21–40与41–60的平均Explained Variance仍只有-0.245和-0.132，第60步为-0.047，说明Critic基本没有学到能解释Return变化的价值函数。PPO并未数值发散：全程无NaN或OOM，后20步平均Value Loss为0.0496，但低Value Loss不能抵消Explained Variance接近或低于0所揭示的弱预测能力。

第60步PPO的策略熵为1.841，而标准与Adaptive GRPO分别为0.151和0.194。结合PPO较低的pass@1和较高的pass@5，更合理的解释是PPO保持了更多候选多样性，而不是已经学出更可靠的单条回答。当前实验因此得到两个互补结论：GRPO在固定60步预算下更快提高单次命中率，PPO则在5次采样覆盖率上更强，但以独立Critic和约2.24倍于标准GRPO的平均Step耗时为代价。

该PPO结果只有单随机种子和512题验证集，且没有第60步Checkpoint。简历和面试中可以陈述已观察到的权衡与诊断，不能写成“证明GRPO全面优于PPO”“PPO最终效果达到37.3%”或“Adaptive稳定提升准确率”。37.30%明确是pass@5，不是pass@1。

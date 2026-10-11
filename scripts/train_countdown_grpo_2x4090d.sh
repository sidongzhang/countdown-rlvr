#!/usr/bin/env bash
set -euo pipefail

# Defaults for 2 x RTX 4090D 48GB. Every value can be
# overridden without editing this file, for example:
#   TOTAL_STEPS=2 WANDB_MODE=offline bash scripts/train_countdown_grpo_2x4090d.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/tinyzero-grpo-data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/root/autodl-tmp/tinyzero-checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2_5_1_5b_grpo_adaptive}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
TEST_FREQ="${TEST_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
RESAMPLE_ZERO_VARIANCE="${RESAMPLE_ZERO_VARIANCE:-true}"
MAX_RESAMPLE_ATTEMPTS="${MAX_RESAMPLE_ATTEMPTS:-1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
VAL_SEED="${VAL_SEED:-42}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${TRAINING_OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"
cd "${PROJECT_ROOT}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.grpo.epsilon=1.0e-6 \
    algorithm.grpo.normalize_by_std=true \
    algorithm.grpo.zero_variance_threshold=1.0e-6 \
    algorithm.grpo.resample_zero_variance_groups="${RESAMPLE_ZERO_VARIANCE}" \
    algorithm.grpo.max_resample_attempts="${MAX_RESAMPLE_ATTEMPTS}" \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size=64 \
    data.max_prompt_length=256 \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1.0e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n="${GROUP_SIZE}" \
    reward_model.enable=false \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.project_name=tinyzero_adaptive_grpo \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="['console','wandb']" \
    trainer.seed="${TRAIN_SEED}" \
    trainer.val_before_train=true \
    trainer.val_num_samples=5 \
    trainer.val_seed="${VAL_SEED}" \
    trainer.val_temperature=0.7 \
    trainer.val_top_p=0.95 \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_epochs=10 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${CHECKPOINT_DIR}/${EXPERIMENT_NAME}" \
    "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"

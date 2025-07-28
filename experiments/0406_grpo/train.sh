#!/bin/bash
set -eu

# Default values
ngpus=8
model_path="Qwen/Qwen2.5-7B-Instruct"
batch_size=$((ngpus * 1))
lr=1e-6
n_rollout=8
max_prompt_length=2048
max_response_length=1024
total_steps=1000
epochs=10
vllm_gpu_util=0.75
per_gpu_batch_size=1
tensor_parallel=2
max_checkpoints=2
eval_freq=50
save_freq=10
actor_offload=False
resume_mode=auto
log_val_n=10
reward_fn=score_lenient
train_size=8k
entropy_coeff=0
rollout_temp=1.0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --ngpus) ngpus="$2"; shift 2 ;;
        --model_path) model_path="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --per_gpu_batch_size) per_gpu_batch_size="$2"; shift 2 ;;
        --lr) lr="$2"; shift 2 ;;
        --n_rollout) n_rollout="$2"; shift 2 ;;
        --max_prompt_length) max_prompt_length="$2"; shift 2 ;;
        --max_response_length) max_response_length="$2"; shift 2 ;;
        --total_steps) total_steps="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --vllm_gpu_util) vllm_gpu_util="$2"; shift 2 ;;
        --tensor_parallel) tensor_parallel="$2"; shift 2 ;;
        --max_checkpoints) max_checkpoints="$2"; shift 2 ;;
        --graph_spec) graph_spec="$2"; shift 2 ;;
        --save_freq) save_freq="$2"; shift 2 ;;
        --eval_freq) eval_freq="$2"; shift 2 ;;
        --actor_offload) actor_offload="$2"; shift 2 ;;
        --resume_mode) resume_mode="$2"; shift 2 ;;
        --log_val_n) log_val_n="$2"; shift 2 ;;
        --reward_fn) reward_fn="$2"; shift 2 ;;
        --train_size) train_size="$2"; shift 2 ;;
        --entropy_coeff) entropy_coeff="$2"; shift 2 ;;
        --rollout_temp) rollout_temp="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --ngpus <int> --model_path <path> --batch_size <int> --per_gpu_batch_size <int> --lr <float>"
            echo "  --n_rollout <int> --max_prompt_length <int> --max_response_length <int>"
            echo "  --total_steps <int> --epochs <int> --vllm_gpu_util <float> --tensor_parallel <int>"
            echo "  --max_checkpoints <int> --graph_spec <str> --save_freq <int> --eval_freq <int>"
            echo "  --actor_offload <bool> --resume_mode <auto|disable> --log_val_n <int> --reward_fn <str> --train_size <str>"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

graph_spec=n10v2
project_name=brian-rlxf-sweep
model_name=$(basename "$model_path" | sed 's/[^a-zA-Z0-9]/_/g')
experiment_name="exp_${graph_spec}_model_${model_name}_bs${batch_size}_lr${lr}_roll${n_rollout}_p${max_prompt_length}_r${max_response_length}_ts${train_size}_rwd${reward_fn}_ent${entropy_coeff}_rt${rollout_temp}"

# Output results
echo "ngpus: $ngpus"
echo "model_path: $model_path"
echo "batch_size: $batch_size"
echo "per_gpu_batch_size: $per_gpu_batch_size"
echo "lr: $lr"
echo "n_rollout: $n_rollout"
echo "max_prompt_length: $max_prompt_length"
echo "max_response_length: $max_response_length"
echo "total_steps: $total_steps"
echo "epochs: $epochs"
echo "vllm_gpu_util: $vllm_gpu_util"
echo "tensor_parallel: $tensor_parallel"
echo "max_checkpoints: $max_checkpoints"
echo "graph_spec: $graph_spec"
echo "save_freq: $save_freq"
echo "eval_freq: $eval_freq"
echo "actor_offload: $actor_offload"
echo "resume_mode: $resume_mode"
echo "log_val_n: $log_val_n"
echo "reward_fn: $reward_fn"
echo "train_size: $train_size"
echo "entropy_coeff: $train_size"
echo "rollout_temp: $rollout_temp"
echo "experiment_name: $experiment_name"


########################   MAIN SCRIPT #########################
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN}"
export HF_HOME="${HF_HOME:-YOUR_HF_HOME}"
export WANDB_API_KEY=8a464cf7b440ea91becb29da1874822e4f5273ed
export VLLM_ATTENTION_BACKEND=XFORMERS

set +eu
source $(conda info --base)/etc/profile.d/conda.sh
conda activate scgl
set -eu

kl_coef=0.001
log_val_n=10 # log to wandb 10 generations for validation

actor_offload=$actor_offload
grad_ckpt=True

train_file=data/ppo/${graph_spec}/${train_size}/train.parquet
val_file=data/ppo/${graph_spec}/monitor/dev.parquet

# use bf16 for model_dtype to avoid OOM when loading checkpoints
# TODO: watch for patches or patch this issue myself
python3 -u -m verl.trainer.main_ppo \
   +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
   +trainer.total_training_steps=${total_steps} \
    custom_reward_function.path=${reward_fn}.py \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_file} \
    data.val_files=${val_file} \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${per_gpu_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.model.enable_gradient_checkpointing=${grad_ckpt} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${per_gpu_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_parallel} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_gpu_util} \
    actor_rollout_ref.rollout.n=${n_rollout} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${per_gpu_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.temperature=${rollout_temp} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${ngpus} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${eval_freq} \
    trainer.total_epochs=${epochs} \
    trainer.max_actor_ckpt_to_keep=${max_checkpoints} \
    trainer.log_val_generations=${log_val_n} \
    trainer.resume_from_path=checkpoints/${project_name}/${experiment_name} \
    trainer.resume_mode=auto \
    > logs/${experiment_name}.out 2> logs/${experiment_name}.err
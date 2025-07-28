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
max_num_gen_batches=30
filter=mean
filter_metric=correctness
filter_min=0.0
filter_max=1.0


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
        --max_num_gen_batches) max_num_gen_batches="$2"; shift 2 ;;
        --filter) filter="$2"; shift 2 ;;
        --filter_metric) filter_metric="$2"; shift 2 ;;
        --filter_min) filter_min="$2"; shift 2 ;;
        --filter_max) filter_max="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --ngpus <int> --model_path <path> --batch_size <int> --per_gpu_batch_size <int> --lr <float>"
            echo "  --n_rollout <int> --max_prompt_length <int> --max_response_length <int>"
            echo "  --total_steps <int> --epochs <int> --vllm_gpu_util <float> --tensor_parallel <int>"
            echo "  --max_checkpoints <int> --graph_spec <str> --save_freq <int> --eval_freq <int>"
            echo "  --actor_offload <bool> --resume_mode <auto|disable> --log_val_n <int> --reward_fn <str> --train_size <str> --max_num_gen_batches <int> --filter <str> --filter_metric <str> --filter_min <float>"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

graph_spec=n10v2
project_name=brian-rlxf-sweep
model_name=$(basename "$model_path" | sed 's/[^a-zA-Z0-9]/_/g')
experiment_name="exp_dapo_${graph_spec}_model_${model_name}_bs${batch_size}_lr${lr}_roll${n_rollout}_p${max_prompt_length}_r${max_response_length}_ts${train_size}_rwd${reward_fn}_ent${entropy_coeff}_rt${rollout_temp}_flt${filter}_${filter_metric}_fltmin${filter_min}_fltmax${filter_max}"

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
echo "max_num_gen_batches: $max_num_gen_batches"
echo "filter: $filter"
echo "filter_metric: $filter_metric"
echo "filter_min: $filter_min"
echo "filter_max: $filter_max"
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

train_file="['data/ppo/${graph_spec}/${train_size}/d0/train.parquet', 'data/ppo/${graph_spec}/${train_size}/d1/train.parquet', 'data/ppo/${graph_spec}/${train_size}/d2/train.parquet', 'data/ppo/${graph_spec}/${train_size}/d3/train.parquet', 'data/ppo/${graph_spec}/${train_size}/d4/train.parquet']"
val_file=data/ppo/${graph_spec}/2k/dev.parquet
echo $train_file
################ DAPO script adapted from recipe ###################

adv_estimator=grpo

# use_kl_in_reward=False  
# kl_coef=0.0 # # algorithm.kl_ctrl.kl_coef=${kl_coef}
# use_kl_loss=False
use_kl_loss=True 
kl_loss_coef=$kl_coef

clip_ratio_low=0.2
clip_ratio_high=0.28 # this param controls positive exploration, raised from 0.2 to 
clip_ratio_c=10.0 # this param controls negative exploration, again, raised from 3 to encourage exploration

# enable_overlong_buffer=True
# overlong_buffer_len=512
# overlong_penalty_factor=1.0
enable_overlong_buffer=False

loss_agg_mode="token-mean"

enable_filter_groups=True
max_num_gen_batches=0
train_prompt_bsz=$batch_size
gen_prompt_bsz=$train_prompt_bsz # $((train_prompt_bsz * 3))
train_prompt_mini_bsz=$batch_size

# Algorithm
rollout_temp=$rollout_temp
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Mathematically equivalent
use_dynamic_bsz=True
# infer_micro_batch_size=null
# train_micro_batch_size=null
offload=$actor_offload

lib=../../lib/verl
eval_freq=$eval_freq
save_freq=$save_freq
nepochs=$epochs
log_val_n=$log_val_n
val_sample=False
val_before_train=False

#     actor_rollout_ref.actor.optim.weight_decay=0.1 \
#     actor_rollout_ref.actor.grad_clip=1.0 \
#     actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ngpus} \
#     actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
#     actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
#     reward_model.overlong_buffer.len=${overlong_buffer_len} \
#     reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \

export PYTHONPATH=$lib
export HYDRA_FULL_ERROR=1
python3 -u -m recipe.dapo.src.main_dapo \
   +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
   +trainer.total_training_steps=$total_steps \
    custom_reward_function.path=${reward_fn}.py \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="${train_file}" \
    data.val_files="${val_file}" \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=${grad_ckpt} \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.rule.statistic=${filter} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_metric} \
    algorithm.filter_groups.rule.min=${filter_min} \
    algorithm.filter_groups.rule.max=${filter_max} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.n=${n_rollout} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_gpu_util} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_parallel} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${rollout_temp} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_sample} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=${ngpus} \
    trainer.nnodes=1 \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${eval_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${nepochs} \
    trainer.max_actor_ckpt_to_keep=${max_checkpoints} \
    trainer.log_val_generations=${log_val_n} \
    trainer.resume_from_path=checkpoints/${project_name}/${experiment_name} \
    trainer.resume_mode=auto \
    > logs/${experiment_name}.out 2> logs/${experiment_name}.err

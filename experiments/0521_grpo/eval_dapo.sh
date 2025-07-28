#!/bin/bash
set -eu
shopt -s nullglob

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
experiment_name="exp_short_prompt_dapo_${graph_spec}_model_${model_name}_bs${batch_size}_lr${lr}_roll${n_rollout}_p${max_prompt_length}_r${max_response_length}_ts${train_size}_rwd${reward_fn}_ent${entropy_coeff}_rt${rollout_temp}_flt${filter}_${filter_metric}_fltmin${filter_min}_fltmax${filter_max}"

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

last_step=$(cat checkpoints/${project_name}/${experiment_name}/latest_checkpointed_iteration.txt)
ckpt_path_prefix=checkpoints/${project_name}/${experiment_name}/global_step_${last_step}/actor
ckpt_path=${ckpt_path_prefix}/huggingface # this will be default used by maybe_merge_actor

maybe_merge_actor() {
    local base_path="$1"
    local hf_model_path="$2"
    local python_script="$(realpath ../../)/lib/scripts/model_merger_fsdp.py"
    local hf_path="$base_path/huggingface"
    safetensors_files=("$hf_path"/*.safetensors)

    if [ ! -d "$hf_path" ] || [ ${#safetensors_files[@]} -eq 0 ]; then
        echo "Directory '$hf_path' does not exist. Running merge script..."
        python3 "$python_script" \
            --backend fsdp \
            --local_dir $base_path \
            --hf_model_path $hf_model_path \
            --target_dir $base_path/huggingface

    else
        echo "Directory '$hf_path' already exists. Skipping merge script."
    fi
}

maybe_merge_actor ${ckpt_path_prefix} ${model_path}
sleep 5

path_to_id() {
    local filepath="$1"
    local filename=$(basename "${filepath%.*}")
    local dirpath=$(dirname "$filepath" | sed 's/\//_/g')
    echo "${dirpath}_${filename}"
}
for type in dev # train
do
for subsplit in 2k/d0/${type}.parquet 2k/d1/${type}.parquet 2k/d2/${type}.parquet 2k/d3/${type}.parquet 2k/d4/${type}.parquet 2k/d5/${type}.parquet 
do
split=${graph_spec}/${subsplit}
data_path=data/ppo/${split}
gen_path=generations/${project_name}/${experiment_name}/global_step_$last_step/${split}
metrics_path=generations/${project_name}/${experiment_name}/global_step_$last_step/${split}.metrics.jsonl

id=$(path_to_id ${subsplit})

# generate
echo "generating for $split"
python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${ngpus} \
        data.path=${data_path} \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.output_path=${gen_path} \
        model.path=${ckpt_path} \
        +model.trust_remote_code=True \
        rollout.temperature=0.0 \
        rollout.do_sample=False \
        rollout.top_k=-1 \
        rollout.top_p=1.0 \
        rollout.prompt_length=${max_prompt_length} \
        rollout.response_length=${max_response_length} \
        rollout.tensor_model_parallel_size=${tensor_parallel} \
        rollout.gpu_memory_utilization=${vllm_gpu_util} \
    > logs/${experiment_name}.${id}.gen.out 2> logs/${experiment_name}.${id}.gen.err

# evaluate
echo "evaluating for $split"
python3 -u $(realpath ../../)/lib/scripts/main_eval.py \
    data.path=${gen_path} \
    save_path=${metrics_path} \
    custom_reward_function.path=score_strict.py > logs/${experiment_name}.${id}.eval.out 2> logs/${experiment_name}.${id}.eval.err

done
done
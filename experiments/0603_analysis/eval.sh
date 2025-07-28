#!/bin/bash
set -eu
shopt -s nullglob

# Default values
ngpus=8
base_model_path="Qwen/Qwen2.5-7B-Instruct"
ckpt_model_path=
max_prompt_length=2048
max_response_length=4096
vllm_gpu_util=0.9
tensor_parallel=1
graph_spec_eval=n10v2
reward_fn=score_tvd
prefix=0406_grpo
merge="yes"
splits=()
data_prefix="ppo"
# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --ngpus) ngpus="$2"; shift 2 ;;
        --base_model_path) base_model_path="$2"; shift 2 ;;
        --ckpt_model_path) ckpt_model_path="$2"; shift 2 ;;
        --max_prompt_length) max_prompt_length="$2"; shift 2 ;;
        --max_response_length) max_response_length="$2"; shift 2 ;;
        --vllm_gpu_util) vllm_gpu_util="$2"; shift 2 ;;
        --tensor_parallel) tensor_parallel="$2"; shift 2 ;;
        --graph_spec_eval) graph_spec_eval="$2"; shift 2 ;;
        --reward_fn) reward_fn="$2"; shift 2 ;;
        --split) splits+=("$2"); shift 2 ;;
        --prefix) prefix="$2"; shift 2 ;;
        --merge) merge="$2"; shift 2 ;;
        --data_prefix) data_prefix="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --ngpus <int> --base_model_path <path> --ckpt_model_path <path>"
            echo "  --max_prompt_length <int> --max_response_length <int>"
            echo "  --vllm_gpu_util <float> --tensor_parallel <int>"
            echo "  --graph_spec_eval <str> --reward_fn <str> --split <str> --prefix <str> --merge <yes or no> --data_prefix <str>"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
if [[ ${#splits[@]} -eq 0 ]]; then
    splits=(dev)
fi

if [[ -z $base_model_path ]]; then
    echo "Error: base_model_path is empty" >&2
    exit 1
fi

if [[ -z $ckpt_model_path ]]; then
    echo "Error: ckpt_model_path is empty" >&2
    exit 1
fi


# Output results
echo "ngpus: $ngpus"
echo "base_model_path: $base_model_path"
echo "prefix": $prefix
echo "ckpt_model_path: $ckpt_model_path"
echo "max_prompt_length: $max_prompt_length"
echo "max_response_length: $max_response_length"
echo "vllm_gpu_util: $vllm_gpu_util"
echo "tensor_parallel: $tensor_parallel"
echo "reward_fn: $reward_fn"
echo "graph_spec_eval: $graph_spec_eval"
echo "data_prefix: $data_prefix"
echo "merge: $merge"


########################   MAIN SCRIPT #########################
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HOME="${HF_HOME:-/export/a02/huggingface}"
export WANDB_API_KEY=8a464cf7b440ea91becb29da1874822e4f5273ed
export VLLM_ATTENTION_BACKEND=XFORMERS

set +eu
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tmp-scgl
set -eu


if [[ $merge == "yes" ]]; then

    ckpt_path_prefix=../${prefix}/${ckpt_model_path}/actor

    if [ ! -d "$ckpt_path_prefix" ]; then
        echo "Error: $ckpt_path_prefix does not exist"
        exit 1
    fi
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

    maybe_merge_actor ${ckpt_path_prefix} ${base_model_path}
    sleep 5
else
    ckpt_path=../${prefix}/${ckpt_model_path}
fi

path_to_id() {
    local filepath="$1"
    local filename=$(basename "${filepath%.*}")
    local dirpath=$(dirname "$filepath" | sed 's/\//_/g')
    echo "${dirpath}_${filename}"
}
for type in ${splits[@]}
do
for subsplit in 2k/${type}.parquet
do
split=${graph_spec_eval}/${subsplit}
data_path=$data_prefix/${split}
gen_path=generations/$prefix/$(python3 get_name.py $ckpt_model_path)/${split}
metrics_path=generations/$prefix/$(python3 get_name.py $ckpt_model_path)/${split}.${reward_fn}.jsonl

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
        > logs/gen.out 2> logs/gen.err

# evaluate
echo "evaluating for $split"
python3 -u $(realpath ../../)/lib/scripts/main_eval.py \
    data.path=${gen_path} \
    save_path=${metrics_path} \
    custom_reward_function.path=${reward_fn}.py \
    > logs/reward.out 2> logs/reward.err

done
done
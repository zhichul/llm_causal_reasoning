
for graph_spec in n10v2
do
for data_source in ../../data/2025-06-03-2/interventional_distribution
do
for seed in 42 43 44 45
do
data_name=0406_intv_dist

# generate main split
python3 prepare_ppo_data.py \
    --local_dir data/sft/${graph_spec} \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --data_name ${data_name} \
    --seed ${seed}

# split by ancestor depth, small training set
python3 prepare_ppo_data.py \
    --monitor_count 8000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --local_dir data/sft/${graph_spec}/8k/seed_${seed}/ \
    --data_name ${data_name} \
    --seed ${seed}

for i in $(seq 0 5)
do
python3 prepare_ppo_data.py \
    --monitor_count 8000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --filter_key ancestor_depth \
    --filter_val ${i} \
    --local_dir data/sft/${graph_spec}/8k/seed_${seed}/d${i} \
    --data_name ${data_name} \
    --seed ${seed}
done

# split by ancestor depth, small training set
python3 prepare_ppo_data.py \
    --monitor_count 16000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --local_dir data/sft/${graph_spec}/16k/seed_${seed}/ \
    --data_name ${data_name} \
    --seed ${seed}

for i in $(seq 0 5)
do
python3 prepare_ppo_data.py \
    --monitor_count 16000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --filter_key ancestor_depth \
    --filter_val ${i} \
    --local_dir data/sft/${graph_spec}/16k/seed_${seed}/d${i} \
    --data_name ${data_name} \
    --seed ${seed}
done
done
done
done
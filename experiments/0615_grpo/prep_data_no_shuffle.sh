
for graph_spec in n5v2 n15v2
do
data_source=../../data/2025-05-21/interventional_distribution
data_name=0406_intv_dist
seed=42

# generate main split
python3 prepare_ppo_data.py \
    --local_dir data/ppo/${graph_spec} \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --data_name ${data_name}

# generate monitor split (small subsets of 100 so that in-training-loop eval don't take forever)
python3 prepare_ppo_data.py \
    --first 100 \
    --local_dir data/ppo/${graph_spec}/monitor \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --data_name ${data_name}


# split by ancestor depth
for i in $(seq 0 5)
do
python3 prepare_ppo_data.py \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --filter_key ancestor_depth \
    --filter_val ${i} \
    --local_dir data/ppo/${graph_spec}/d${i} \
    --data_name ${data_name}
done

# split by ancestor depth, small training set 2k
python3 prepare_ppo_data.py \
    --first 2000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --local_dir data/ppo/${graph_spec}/2k/no_graph_shuffle/ \
    --data_name ${data_name} \

for i in $(seq 0 5)
do
python3 prepare_ppo_data.py \
    --first 2000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --filter_key ancestor_depth \
    --filter_val ${i} \
    --local_dir data/ppo/${graph_spec}/2k/no_graph_shuffle/seed_${seed}/d${i} \
    --monitor_count 250 \
    --data_name ${data_name} \
    --seed ${seed}
done

for i in $(seq 0 5)
do
python3 prepare_ppo_data.py \
    --first 2000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --filter_key ancestor_depth \
    --filter_val ${i} \
    --local_dir data/ppo/${graph_spec}/2k/no_graph_shuffle/d${i} \
    --data_name ${data_name}
done

# split by ancestor depth, small training set 4k
python3 prepare_ppo_data.py \
    --first 4000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --local_dir data/ppo/${graph_spec}/4k/no_graph_shuffle/ \
    --data_name ${data_name} \

for i in $(seq 0 5)
do
python3 prepare_ppo_data.py \
    --first 4000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --filter_key ancestor_depth \
    --filter_val ${i} \
    --local_dir data/ppo/${graph_spec}/4k/no_graph_shuffle/seed_${seed}/d${i} \
    --monitor_count 500 \
    --data_name ${data_name} \
    --seed ${seed}
done

# split by ancestor depth, small training set
python3 prepare_ppo_data.py \
    --first 8000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --local_dir data/ppo/${graph_spec}/8k/no_graph_shuffle/ \
    --data_name ${data_name} \

for i in $(seq 0 5)
do
python3 prepare_ppo_data.py \
    --first 8000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --filter_key ancestor_depth \
    --filter_val ${i} \
    --local_dir data/ppo/${graph_spec}/8k/no_graph_shuffle/seed_${seed}/d${i} \
    --monitor_count 1000 \
    --data_name ${data_name} \
    --seed ${seed}
done

# split by ancestor depth, small training set
python3 prepare_ppo_data.py \
    --first 16000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --local_dir data/ppo/${graph_spec}/16k/no_graph_shuffle/ \
    --data_name ${data_name} \

for i in $(seq 0 5)
do
python3 prepare_ppo_data.py \
    --first 16000 \
    --data_source $data_source \
    --data_subsplits ${graph_spec} \
    --filter_key ancestor_depth \
    --filter_val ${i} \
    --local_dir data/ppo/${graph_spec}/16k/no_graph_shuffle/seed_${seed}/d${i} \
    --monitor_count 2000 \
    --data_name ${data_name} \
    --seed ${seed}
done
done
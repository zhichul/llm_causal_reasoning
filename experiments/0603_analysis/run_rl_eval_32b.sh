for folder in 0406_grpo
do
for gs in n10v2 n15v2 n5v2
do 
bash eval.sh --ckpt_model_path checkpoints/brian-rlxf-sweep/exp_dapo_n10v2_model_Qwen2_5_32B_Instruct_bs8_lr1e-6_roll32_p2048_r4096_ts8k_rwdscore_strict_ent0.001_rt1.0_fltmean_correctness_fltmin-0.01_fltmax1.01/global_step_5000 --prefix ${folder} --merge yes --graph_spec_eval $gs --data_prefix ../${folder}/data/ppo --reward_fn score_tvd --max_response_length 4096

bash eval.sh --ckpt_model_path checkpoints/brian-rlxf-sweep/exp_dapo_n10v2_model_Qwen2_5_32B_Instruct_bs8_lr1e-6_roll32_p2048_r4096_ts8k_rwdscore_lenient_ent0.001_rt1.0_fltmean_correctness_fltmin-0.01_fltmax1.01/global_step_5000 --prefix ${folder} --merge yes --graph_spec_eval $gs --data_prefix ../${folder}/data/ppo --reward_fn score_tvd --max_response_length 4096

bash eval.sh --ckpt_model_path checkpoints/brian-rlxf-sweep/exp_dapo_n10v2_model_Qwen2_5_32B_Instruct_bs8_lr1e-6_roll32_p2048_r4096_ts8k_rwdscore_strict_ent0.001_rt1.0_fltmean_correctness_fltmin0.09_fltmax0.91/global_step_5000 --prefix ${folder} --merge yes --graph_spec_eval $gs --data_prefix ../${folder}/data/ppo --reward_fn score_tvd --max_response_length 4096
done
done

exit

for folder in 0521_grpo
do
for gs in n10v2 n15v2 n5v2
do 
bash eval.sh --ckpt_model_path checkpoints/brian-rlxf-sweep/exp_short_prompt_dapo_n10v2_model_Qwen2_5_32B_Instruct_bs8_lr1e-6_roll32_p2048_r4096_ts8k_rwdscore_strict_ent0.001_rt1.0_fltmean_correctness_fltmin-0.01_fltmax1.01/global_step_5000 --prefix ${folder} --merge yes --graph_spec_eval $gs --data_prefix ../${folder}/data/ppo --reward_fn score_tvd --max_response_length 4096 --base_model_path Qwen/Qwen2.5-32B-Instruct

bash eval.sh --ckpt_model_path checkpoints/brian-rlxf-sweep/exp_short_prompt_dapo_n10v2_model_Qwen2_5_32B_Instruct_bs8_lr1e-6_roll32_p2048_r4096_ts8k_rwdscore_lenient_ent0.001_rt1.0_fltmean_correctness_fltmin-0.01_fltmax1.01/global_step_5000 --prefix ${folder} --merge yes --graph_spec_eval $gs --data_prefix ../${folder}/data/ppo --reward_fn score_tvd --max_response_length 4096 --base_model_path Qwen/Qwen2.5-32B-Instruct

bash eval.sh --ckpt_model_path checkpoints/brian-rlxf-sweep/exp_short_prompt_dapo_n10v2_model_Qwen2_5_32B_Instruct_bs8_lr1e-6_roll32_p2048_r4096_ts8k_rwdscore_strict_ent0.001_rt1.0_fltmean_correctness_fltmin0.09_fltmax0.91/global_step_2500 --prefix ${folder} --merge yes --graph_spec_eval $gs --data_prefix ../${folder}/data/ppo --reward_fn score_tvd --max_response_length 4096 --base_model_path Qwen/Qwen2.5-32B-Instruct
done
done
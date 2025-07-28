for folder in 0521 0604
do
for gs in n10v2 n15v2 n5v2
do 
bash eval.sh --ckpt_model_path checkpoints/brian-sft-sweep/exp_${folder}_n10v2_model_Qwen2_5_7B_Instruct_bs8_lr1e-6_len2048_ts8k_lora0/global_step_best_loss --prefix 0603_sft --merge no --graph_spec_eval $gs --data_prefix ../${folder}_grpo/data/sft --reward_fn score_tvd_sft --max_response_length 1024

bash eval.sh --ckpt_model_path checkpoints/brian-sft-sweep/exp_${folder}_n10v2_model_Qwen2_5_7B_Instruct_bs8_lr1e-6_len2048_ts8k_lora0/global_step_best_reward --prefix 0603_sft --merge no --graph_spec_eval $gs --data_prefix ../${folder}_grpo/data/sft --reward_fn score_tvd_sft --max_response_length 1024 

done
done
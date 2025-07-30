# LLM Causal Reasoning 
This codebase provides pipelines for synthetic data generation, training, and analysis in an ongonging study of RLVR's generalization behavior in LLM algorithmic reasoning.


## Installation
Installation
```
bash install_dependencies.sh
```

## Synthetic Data Generation
We synthesize training examples from different levels of the causal hiearchy (observational, interventional, counterfactual). Each example comes with a parametrized causal DAG, some interventions / observations, and a probability query, and a numerical answer to the query. This problem can be solved algorithmically by graphical model inference on (modified) graphs, and we are interested in using this data to study if LLMs trained with RL recover interesting or existing algorithmic solutions.

Code for synthesizing causal graphs over discrete and continuous random variables are included in  [`causal_graph`](https://github.com/zhichul/synthetic-causal-graph-learning/tree/32332339eb7dbc965be2ecdc7a025066a080ca6b/causal_graph), along with some graphical model inference algorithms such as variable elimination.

Code for generating training examples and datasets are included in [`qa_data`](https://github.com/zhichul/synthetic-causal-graph-learning/tree/32332339eb7dbc965be2ecdc7a025066a080ca6b/qa_data).

Our current experiments focus on generalization within a causal level, on interventional graphs over 5/10/15 variables, and each graph comes with multiple (intervention, query, answer) triplets. Additional experiments have been planned to study genearlization across levels of the causal hiearchy.

See example configurations of synthetic under [configs/](configs/).

## GRPO, DAPO and SFT Training
Our experiments study RL's generalization across different configurations (GRPO vs. DAPO, soft vs strict reward functions, specific versus generic prompts, smaller vs. larger models).

We use [VeRL](https://github.com/volcengine/verl) for RL and SFT training. Scripts for training (including reward function definitions) and lightweight dataprep into VeRL accepted format is inclued in directories under `experiments`. Stable experiment setups include `0406_grpo`, `0521_grpo`, and `0603_sft`. You'll need to change the huggingface tokens and wandb tokens in the scripts (ones included in scripts are invalidated for safety). Other experiments are either under development or for archival purposes.

We have a unified set of scripts for evaluation in `0603_analysis`, which merges sharded model checkpoints automatically and runs generation using VeRL.

Example usage from 0406_grpo's `train.sh`
```
Usage: train.sh [options]
Options:
  --ngpus <int> --model_path <path> --batch_size <int> --per_gpu_batch_size <int> --lr <float>
  --n_rollout <int> --max_prompt_length <int> --max_response_length <int>
  --total_steps <int> --epochs <int> --vllm_gpu_util <float> --tensor_parallel <int>
  --max_checkpoints <int> --graph_spec <str> --save_freq <int> --eval_freq <int>
```

## Analyzing Reasoning Traces 
We study both raw accuracy as well as patterns in the reasoning traces to answer our research question about whether models learn genearlizable algorithmic solutions from RLVR training on causal inference problems. As well as compare it to SFT baselines. We find that 7B/32B LLMs learn to modify the graph and margianlize out parents recursively to compute estimate the query, but makes significant modeling errors (e.g. false independence assumptions) during its execution of the strategy.

Scripts for runing inference with [VeRL](https://github.com/volcengine/verl) using vllm as backend is included in `experiments/0603_analysis`.

A pipeline for automatically generating plots and tables, choosing random examples, and performing statistical testing given a set of generations from different systems is included in `experiments/0628_analysis`.

Code and data for manual and fine-grained analysis of reasoning traces with the help of LLMs are included in `experiments/0701_trace`.

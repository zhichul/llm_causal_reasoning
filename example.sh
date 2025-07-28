#!/usr/bin/env bash

# will generate data to data/2025-02-23, as specified by the yaml file
# python3 -m qa_data.observational configs/2025-02-23/observational.yaml
# python3 -m qa_data.interventional configs/2025-02-24/interventional.yaml
# python3 -m qa_data.counterfactual configs/2025-02-24/counterfactual.yaml

# python3 -m qa_data.interventional configs/2025-03-10/causal_sufficient_sample.yaml
# python3 -m qa_data.interventional configs/2025-03-10/causal_sufficient_distribution.yaml
# python3 -m qa_data.interventional configs/2025-03-10/causal_sufficient_argmax.yaml

# python3 -m qa_data.interventional configs/2025-03-29/causal_sufficient_distribution_debug.yaml
# python3 -m qa_data.interventional configs/2025-03-29/causal_sufficient_distribution_small.yaml
# python3 -m qa_data.interventional configs/2025-03-29/causal_sufficient_distribution_large.yaml
# python3 -u -m qa_data.interventional configs/2025-04-01/causal_sufficient_distribution_small.yaml > 0401_small.out 2> 0401_small.err &
# pid1=$!
# python3 -u -m qa_data.interventional configs/2025-04-01/causal_sufficient_distribution_mid.yaml > 0401_mid.out 2> 0401_mid.err &
# pid2=$!
# python3 -u -m qa_data.interventional configs/2025-04-01/causal_sufficient_distribution_large.yaml > 0401_large.out 2> 0401_large.err &
# pid3=$!
# wait $pid1
# echo "First job (PID $pid1) completed."

# wait $pid2
# echo "Second job (PID $pid2) completed."

# wait $pid3
# echo "Third job (PID $pid3) completed."

# echo "All jobs done."

# python3 -u -m qa_data.interventional configs/2025-04-06/causal_sufficient_distribution_small.yaml > 0406_small.out 2> 0406_small.err &
# pid1=$!
# python3 -u -m qa_data.interventional configs/2025-04-06/causal_sufficient_distribution_mid.yaml > 0406_mid.out 2> 0406_mid.err &
# pid2=$!
# python3 -u -m qa_data.interventional configs/2025-04-06/causal_sufficient_distribution_large.yaml > 0406_large.out 2> 0406_large.err &
# pid3=$!
# wait $pid1
# echo "First job (PID $pid1) completed."

# wait $pid2
# echo "Second job (PID $pid2) completed."

# wait $pid3
# echo "Third job (PID $pid3) completed."

# echo "All jobs done."

# python3 -u -m qa_data.observational configs/2025-04-19/observational/causal_sufficient_distribution_small.yaml > 0419_small.out 2> 0419_small.err &
# pid1=$!

# python3 -u -m qa_data.observational configs/2025-04-19/observational/causal_sufficient_distribution_mid.yaml > 0419_mid.out 2> 0419_mid.err &
# pid2=$!

# python3 -u -m qa_data.observational configs/2025-04-19/observational/causal_sufficient_distribution_large.yaml > 0419_large.out 2> 0419_large.err &
# pid3=$!

# wait $pid1
# echo "First job (PID $pid1) completed."

# wait $pid2
# echo "Second job (PID $pid2) completed."

# wait $pid3
# echo "Third job (PID $pid3) completed."

# echo "All jobs done."

# python3 -u -m qa_data.interventional configs/2025-05-21/interventional/causal_sufficient_distribution_small.yaml > 0521_small.out 2> 0521_small.err &
# pid1=$!
# python3 -u -m qa_data.interventional configs/2025-05-21/interventional/causal_sufficient_distribution_mid.yaml > 0521_mid.out 2> 0521_mid.err &
# pid2=$!
# python3 -u -m qa_data.interventional configs/2025-05-21/interventional/causal_sufficient_distribution_large.yaml > 0521_large.out 2> 0521_large.err &
# pid3=$!
# wait $pid1
# echo "First job (PID $pid1) completed."

# wait $pid2
# echo "Second job (PID $pid2) completed."

# wait $pid3
# echo "Third job (PID $pid3) completed."

# echo "All jobs done."

# python3 -u -m qa_data.observational configs/2025-05-21/observational/causal_sufficient_distribution_small.yaml > 0521_small.out 2> 0521_small.err &
# pid1=$!
# python3 -u -m qa_data.observational configs/2025-05-21/observational/causal_sufficient_distribution_mid.yaml > 0521_mid.out 2> 0521_mid.err &
# pid2=$!
# python3 -u -m qa_data.observational configs/2025-05-21/observational/causal_sufficient_distribution_large.yaml > 0521_large.out 2> 0521_large.err &
# pid3=$!
# wait $pid1
# echo "First job (PID $pid1) completed."

# wait $pid2
# echo "Second job (PID $pid2) completed."

# wait $pid3
# echo "Third job (PID $pid3) completed."

# echo "All jobs done."

# python3 -u -m qa_data.interventional configs/2025-06-03-2/interventional/causal_sufficient_distribution_small.yaml > 0603-2_small.out 2> 0603-2_small.err &
# pid1=$!
# python3 -u -m qa_data.interventional configs/2025-06-03-2/interventional/causal_sufficient_distribution_mid.yaml > 0603-2_mid.out 2> 0603-2_mid.err &
# pid2=$!
# python3 -u -m qa_data.interventional configs/2025-06-03-2/interventional/causal_sufficient_distribution_large.yaml > 0603-2_large.out 2> 0603-2_large.err &
# pid3=$!
# wait $pid1
# echo "First job (PID $pid1) completed."

# wait $pid2
# echo "Second job (PID $pid2) completed."

# wait $pid3
# echo "Third job (PID $pid3) completed."

# echo "All jobs done."

# python3 -u -m qa_data.interventional configs/2025-06-16/interventional/causal_sufficient_distribution_small.yaml > logs/0616_small.out 2> logs/0616_small.err &
# pid1=$!
# python3 -u -m qa_data.interventional configs/2025-06-16/interventional/causal_sufficient_distribution_mid.yaml > logs/0616_mid.out 2> logs/0616_mid.err &
# pid2=$!
# python3 -u -m qa_data.interventional configs/2025-06-16/interventional/causal_sufficient_distribution_large.yaml > logs/0616_large.out 2> logs/0616_large.err &
# pid3=$!
# wait $pid1
# echo "First job (PID $pid1) completed."

# wait $pid2
# echo "Second job (PID $pid2) completed."

# wait $pid3
# echo "Third job (PID $pid3) completed."

# echo "All jobs done."

# python3 -u -m qa_data.interventional configs/2025-06-17/interventional/causal_sufficient_distribution_small.yaml > logs/0617_small.out 2> logs/0617_small.err &
# pid1=$!
# python3 -u -m qa_data.interventional configs/2025-06-17/interventional/causal_sufficient_distribution_mid.yaml > logs/0617_mid.out 2> logs/0617_mid.err &
# pid2=$!
# python3 -u -m qa_data.interventional configs/2025-06-17/interventional/causal_sufficient_distribution_large.yaml > logs/0617_large.out 2> logs/0617_large.err &
# pid3=$!
# wait $pid1
# echo "First job (PID $pid1) completed."

# wait $pid2
# echo "Second job (PID $pid2) completed."

# wait $pid3
# echo "Third job (PID $pid3) completed."

# echo "All jobs done."


# python3 -u -m qa_data.interventional configs/2025-06-03-1/interventional/causal_sufficient_distribution_small.yaml  > logs/06031_small.out 2> logs/06031_small.err &
# pid1=$!
# python3 -u -m qa_data.interventional configs/2025-06-03-1/interventional/causal_sufficient_distribution_mid.yaml > logs/06031_mid.out 2> logs/06031_mid.err &
# pid2=$!
# python3 -u -m qa_data.interventional configs/2025-06-03-1/interventional/causal_sufficient_distribution_large.yaml > logs/06031_large.out 2> logs/06031_large.err &
# pid3=$!
# wait $pid1
# echo "First job (PID $pid1) completed."

# wait $pid2
# echo "Second job (PID $pid2) completed."

# wait $pid3
# echo "Third job (PID $pid3) completed."

# echo "All jobs done."

# python3 -u -m qa_data.interventional configs/2025-06-03-2/interventional/causal_sufficient_distribution_small.yaml > logs/06032_small.out 2> logs/06032_small.err &
# pid1=$!
# python3 -u -m qa_data.interventional configs/2025-06-03-2/interventional/causal_sufficient_distribution_mid.yaml > logs/06032_mid.out 2> logs/06032_mid.err &
# pid2=$!
# python3 -u -m qa_data.interventional configs/2025-06-03-2/interventional/causal_sufficient_distribution_large.yaml > logs/06032_large.out 2> logs/06032_large.err &
# pid3=$!
# wait $pid1
# echo "First job (PID $pid1) completed."

# wait $pid2
# echo "Second job (PID $pid2) completed."

# wait $pid3
# echo "Third job (PID $pid3) completed."

# echo "All jobs done."



python3 -u -m qa_data.interventional configs/2025-06-29/interventional/causal_sufficient_distribution_small.yaml > logs/0629_small.out 2> logs/0629_small.err &
pid1=$!
python3 -u -m qa_data.interventional configs/2025-06-29/interventional/causal_sufficient_distribution_mid.yaml > logs/0629_mid.out 2> logs/0629_mid.err &
pid2=$!
python3 -u -m qa_data.interventional configs/2025-06-29/interventional/causal_sufficient_distribution_large.yaml > logs/0629_large.out 2> logs/0629_large.err &
pid3=$!
wait $pid1
echo "First job (PID $pid1) completed."

wait $pid2
echo "Second job (PID $pid2) completed."

wait $pid3
echo "Third job (PID $pid3) completed."

echo "All jobs done."
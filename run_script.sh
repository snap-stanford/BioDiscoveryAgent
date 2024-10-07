
num_processes=10

for data_name in IL2 IFNG Carnevale22_Adenosine
do

  # # vanilla
  for ((i=0; i<0+$num_processes; i++)); do
    run_name="$i"
    python3 research_assistant.py --task perturb-genes --model claude-3-5-sonnet-20240620 --run_name "$run_name" --data_name $data_name --steps 10 --num_genes 128 --log_dir sonnet &
  done
  wait

  # # all tools
  for ((i=0; i<0+$num_processes; i++)); do
    run_name="$i"
    python3 research_assistant.py --task perturb-genes --model claude-3-5-sonnet-20240620 --run_name "$run_name" --data_name $data_name --steps 10 --num_genes 128 --log_dir sonnet_all --gene_search True --gene_search_diverse True --critique True --lit_review True &
  done
  wait
done

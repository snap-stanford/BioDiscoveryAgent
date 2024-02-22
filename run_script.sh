
#python research_assistant.py --task perturb-genes-brief --run_name exp1 --bio_task IL2 --steps 100 --num_genes 32 
#python research_assistant.py --task perturb-genes-brief --run_name exp2 --bio_task IL2 --steps 100 --num_genes 32 
#python research_assistant.py --task perturb-genes-brief --run_name exp3 --bio_task IL2 --steps 100 --num_genes 32

#python research_assistant.py --task perturb-genes-brief --run_name exp_gpt4_1 --use_gpt4 True --bio_task IL2 --steps 30 --num_genes 32
#python research_assistant.py --task perturb-genes-brief --run_name exp_gpt4_2 --use_gpt4 True --bio_task IL2 --steps 30 --num_genes 32
#python research_assistant.py --task perturb-genes-brief --run_name exp_gpt4_3 --use_gpt4 True --bio_task IL2 --steps 30 --num_genes 32

#python research_assistant.py --task perturb-genes-brief --run_name exp1 --bio_task IFNG --steps 100 --num_genes 32
#python research_assistant.py --task perturb-genes-brief --run_name exp2 --bio_task IFNG --steps 100 --num_genes 32
#python research_assistant.py --task perturb-genes-brief --run_name exp3 --bio_task IFNG --steps 100 --num_genes 32

#python research_assistant.py --task perturb-genes-brief --run_name exp1 --data_name Belk22 --steps 100 --num_genes 32
#python research_assistant.py --task perturb-genes-brief --run_name exp2 --data_name Belk22 --steps 100 --num_genes 32
#python research_assistant.py --task perturb-genes-brief --run_name exp3 --data_name Belk22 --steps 100 --num_genes 32



#python research_assistant.py --task perturb-genes-brief --model claude --run_name exp4 --data_name Scharenberg22 --steps 100 --num_genes 16
#python research_assistant.py --task perturb-genes-brief --model claude --run_name exp5 --data_name Scharenberg22 --steps 100 --num_genes 16
# python research_assistant.py --task perturb-genes-brief --model claude --run_name dummy_summary9 --data_name IFNG --steps 15 --num_genes 32 --log_dir logs_normal

num_processes=10
# IL2 Carnevale22_Adenosine Carnevale22_Cyclosporine Carnevale22_Tacrolimus Carnevale22_TGFb Steinhart_crispra_GD2_D22
for data_name in Steinhart_crispra_GD2_D22 Carnevale22_Adenosine Carnevale22_Cyclosporine Carnevale22_Tacrolimus Carnevale22_TGFb IL2 IFNG
do

  # # vanilla
  for ((i=0; i<0+$num_processes; i++)); do
    run_name="dummy_summary$i"
    folder="perturb-genes-brief$i"
    python research_assistant.py --folder_name "$folder" --task perturb-genes-brief --model claude-1 --run_name "$run_name" --data_name $data_name --steps 10 --num_genes 128 --log_dir v1 &
  done
  wait

  # # vanilla + feature
  for ((i=0; i<0+$num_processes; i++)); do
    run_name="dummy_summary$i"
    folder="perturb-genes-brief$i"
    python research_assistant.py --folder_name "$folder" --task perturb-genes-brief --model claude-1 --run_name "$run_name" --data_name $data_name --steps 10 --num_genes 128 --log_dir v1_feature --gene_search True &
  done
  wait

  # # vanilla + critique
  for ((i=0; i<0+$num_processes; i++)); do
    run_name="dummy_summary$i"
    folder="perturb-genes-brief$i"
    python research_assistant.py --folder_name "$folder" --task perturb-genes-brief --model claude-1 --run_name "$run_name" --data_name $data_name --steps 10 --num_genes 128 --log_dir v1_critique --critique True &
  done
  wait

  # # vanilla + arxiv search
  # # vanilla + arxiv search + feature + critique

  # # vanilla + obs only TODO
  # # vanilla + no obs TODO
done
# Loop to run the Python script in parallel
# for ((i=0; i<=0+$num_processes; i++)); do
#   run_name="dummy_summary$i"
#   folder="perturb-genes-brief$i"
#   python research_assistant.py --folder_name "$folder" --task perturb-genes-brief --model gpt-4 --run_name "$run_name" --data_name IFNG --steps 10 --num_genes 128 --log_dir gpt4_2 &
# done

# Wait for all background processes to finish

#python research_assistant.py --task perturb-genes-brief --model gpt3.5 --run_name gpt3.5_exp1 --data_name Belk22 --steps 100 --num_genes 32
#python research_assistant.py --task perturb-genes-brief --model gpt3.5 --run_name gpt3.5_exp2 --data_name Belk22 --steps 100 --num_genes 32
#python research_assistant.py --task perturb-genes-brief --model gpt3.5 --run_name gpt3.5_exp3 --data_name Belk22 --steps 100 --num_genes 32


#python research_assistant_gpt4.py --task perturb-genes-brief --use_gpt4 True --run_name exp_gpt4_fix_1 --data_name Belk22 --steps 100 --num_genes 32
#python research_assistant_gpt4.py --task perturb-genes-brief --use_gpt4 True --run_name exp_gpt4_2 --data_name Belk22 --steps 100 --num_genes 32
#python research_assistant_gpt4.py --task perturb-genes-brief --use_gpt4 True --run_name exp_gpt4_3 --data_name Belk22 --steps 100 --num_genes 32

#python research_assistant.py --task perturb-genes-brief --run_name exp1 --data_name Scharenberg22 --steps 100 --num_genes 32
#python research_assistant.py --task perturb-genes-brief --run_name exp2 --data_name Scharenberg22 --steps 100 --num_genes 32
#python research_assistant.py --task perturb-genes-brief --run_name exp3 --data_name Scharenberg22 --steps 100 --num_genes 32

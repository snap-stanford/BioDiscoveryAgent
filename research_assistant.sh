#/bin/bash

# time stamp
folder=research_assistant_logs_05_03
device=7

for i in {1..5}
do 
    task=cifar10-training
    ts=$(date +%Y%m%d_%H%M%S)
    run_name=claude-$task-$ts
    mkdir -p $folder/$run_name
    python=$(which python)

    python -u research_assistant.py --log_dir $folder --run_name $run_name --task $task --steps 30 --device $device --python $python > $folder/$run_name/log.log 2>&1
done

# for i in {1..5}
# do 
#     for task in cifar10-training speed-up bibtex-generation
#     do 

#         ts=$(date +%Y%m%d_%H%M%S)
#         run_name=claude-$task-$ts
#         mkdir -p $folder/$run_name
#         python=$(which python)

#         python -u research_assistant.py --log_dir $folder --run_name $run_name --task $task --steps 20 --device $device --python $python 

#         # ts=$(date +%Y%m%d_%H%M%S)
#         # run_name=gpt4-$task-$ts
#         # mkdir -p $folder/$run_name
#         # python=$(which python)

#         # python -u research_assistant.py --log_dir $folder --run_name $run_name --task $task --steps 20 --device $device --python $python --use_gpt4 True > $folder/$run_name/log 2>&1
#     done
# done
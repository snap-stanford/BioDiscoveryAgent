import numpy as np
import pandas as pd
import os
import glob
import argparse

def calc_stats(data: str, model: str, rounds: int, trials: int, essential: bool):
    topmovers = np.load(f'datasets/topmovers_{data}.npy', allow_pickle=True)
    # if data == "Horlbeck":
    #     temp = []
    #     for pair in topmovers:
    #         print(pair)
    #         newpair = f"{pair[0]}_{pair[1]}"
    #         temp.append(newpair)
    #     topmovers = temp
    accuracies = []
    for i in range(trials):
        if f"{model}_{data}" in os.listdir("/dfs/scratch0/andleerew/llms_results"):
            subdir = os.path.join("/dfs/scratch0/andleerew/llms_results", f"{model}_{data}/{str(i)}/sampled_genes_{rounds}.npy")
        elif f"{model}_{data}" in os.listdir("."):
            subdir = f'{model}_{data}/{str(i)}/sampled_genes_{rounds}.npy'
            # subdir = f'{model}_{data}/test/sampled_genes_{trials}.npy'
            # subdir = f'sonnet_IFNG_test3/sampled_genes_{trials}.npy'
        elif f"{model}_{data}" in os.listdir("/dfs/scratch0/andleerew/qianv2"):
            subdir = os.path.join("/dfs/scratch0/andleerew/qianv2", f"{model}_{data}/{str(i)}/sampled_genes_{rounds}.npy") 
        elif f"{model}_{data}" in os.listdir("/dfs/scratch0/andleerew"):
            subdir = os.path.join("/dfs/scratch0/andleerew", f"{model}_{data}/{str(i)}/sampled_genes_{rounds}.npy") 
        else:
            print(f"ERROR: Couldn't find file {model}_{data}/{str(i)}/sampled_genes_{rounds}.npy")
            continue
        # subdir = f"/dfs/scratch0/andleerew/qianv1/{model}_{data}/dummy_summary{str(i)}/sampled_genes_{rounds}.npy"
        if not os.path.exists(subdir):
            print(f"ERROR: Couldn't find file {subdir}")
            continue
        print(subdir)
        # subdir = f'{model}_{data}/test/sampled_genes_{trials}.npy'
        # subdir = f'{model}_{data}/{str(i)}/sampled_genes_{trials}.npy'
        # if not os.path.exists(subdir):
        #     print(f"{subdir} does not exist")
        #     continue
        # print(subdir)
        pred = np.load(subdir)
        # print(topmovers)
        if essential == 0:
            essential = pd.read_csv("CEGv2.txt", delimiter='\t')['GENE'].tolist()
            topmovers = list(set(topmovers) - set(essential))
            pred = list(set(pred) - set(essential))
        hits = list(set(pred).intersection(topmovers))
        if data == "Horlbeck":
            accuracies.append(len(hits))
        else: 
            accuracies.append(len(hits)/len(topmovers))
    print(accuracies)
    print(f"Model: {model}, Data: {data}, mean: {np.mean(accuracies)}, std: {np.std(accuracies)}")    
    # print(len(topmovers))
    # print("===== CLAUDE V1 STATS =====")
    # ref_acc = []
    # for i in range(10):
    #     # subdir = f'/dfs/user/yhr/AI_RA/research_assistant/logs/agent_log/IFNG/IFNG_exp{str(i)}/sampled_genes_5.npy'
    #     subdir = f"/dfs/scratch0/jianv/bio-logs/agent-final/v4_all3_again50_IFNG/dummy_summary{str(i)}/sampled_genes_5.npy"
    #     if not os.path.exists(subdir):
    #         print(f"{subdir} does not exist")
    #         continue
    #     pred = np.load(subdir)
    #     hits = list(set(pred).intersection(topmovers))
    #     ref_acc.append(len(hits)/len(topmovers))
    # print(f"mean: {np.mean(ref_acc)}, std: {np.std(ref_acc)}")    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--essential', type=int, default=1)
    args = parser.parse_args()
    calc_stats(args.dataset, args.model, args.rounds, args.trials, args.essential)


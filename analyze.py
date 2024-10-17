import numpy as np
import pandas as pd
import os
import glob
import argparse

def calc_stats(data: str, model: str, rounds: int, trials: int, essential: bool):
    topmovers = np.load(f'datasets/topmovers_{data}.npy', allow_pickle=True)
    if data == "Horlbeck":
        temp = []
        for pair in topmovers:
            print(pair)
            newpair = f"{pair[0]}_{pair[1]}"
            temp.append(newpair)
        topmovers = temp
    accuracies = []
    for i in range(trials):
        if f"{model}_{data}" in os.listdir("."):
            subdir = f'{model}_{data}/test/sampled_genes_{rounds}.npy'
        else:
            print(f"ERROR: Couldn't find file {model}_{data}/test/sampled_genes_{rounds}.npy")
            continue
        if not os.path.exists(subdir):
            print(f"ERROR: Couldn't find file {subdir}")
            continue
        print(subdir)
        pred = np.load(subdir)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--essential', type=int, default=1)
    args = parser.parse_args()
    calc_stats(args.dataset, args.model, args.rounds, args.trials, args.essential)


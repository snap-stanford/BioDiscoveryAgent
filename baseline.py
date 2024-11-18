from gene import *
import numpy as np
import argparse
import random

def baseline_sample(dataset: str, sample_size: int, pathways):
    genes = get_topk_genes_in_reactome(pathways, sampled = [], K=sample_size)
    if len(genes) < sample_size:
        print("trying KEGG enrichment")
        remaining = sample_size - len(genes)
        pathways = get_enrichment_KEGG_pathways(genes)
        candidates = get_topk_genes_in_pathways(pathways, genes, K=remaining)
        genes = genes + candidates
    np.save(f"baseline/{dataset}/sampled_genes_1.npy", genes)
    for i in range(1, 5):
        gene_sampled = list(np.load(f"baseline/{dataset}/sampled_genes_{str(i)}.npy"))
        pathways = get_enrichment(gene_sampled, database="Reactome_2022")
        path_ids = []
        for path in pathways:
            path_ids.append(path.split()[-1])
        genes = get_topk_genes_in_reactome(path_ids, gene_sampled, K=sample_size)
        if len(genes) < sample_size:
            print("trying KEGG enrichment")
            remaining = sample_size - len(genes)
            pathways = get_enrichment_KEGG_pathways(gene_sampled)
            candidates = get_topk_genes_in_pathways(pathways, gene_sampled + genes, K=remaining)
            genes = genes + candidates
        print(len(genes))
        all_sampled = gene_sampled + genes
        print(len(all_sampled))
        np.save(f"baseline/{dataset}/sampled_genes_{str(i+1)}.npy", all_sampled)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    pathways = []
    sample_size = 128
    if args.dataset == "IFNG":
        pathways = ["R-HSA-877300", "R-HSA-1280215"]
    elif args.dataset == "IL2":
        pathways = ["R-HSA-451927", "R-HSA-1280215"]
    elif args.dataset == "Carnevale22_Adenosine":
        pathways = ["R-HSA-389948", "R-HSA-202433", "R-HSA-389957"]
    elif args.dataset == "Steinhart_crispra_GD2_D22":
        pathways = ["R-HSA-451927", "R-HSA-110021", "R-HSA-165159"]
    elif args.dataset == "Scharenberg22":
        pathways = ["R-HSA-3229371", "R-HSA-6798163", "R-HSA-1632852"]
        sample_size = 32
    elif args.dataset == "Sanchez21_down":
        pathways = ["R-HSA-5683057", "R-HSA-381119", "R-HSA-983168"]
    baseline_sample(args.dataset, sample_size, pathways)
    
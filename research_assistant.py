import os
import anthropic
import json
import subprocess
import re
import datetime
import argparse
import shutil
import selectors
from tools import ALL_TOOLS, agent_loop
import tools
#from LLM import complete_text_openai, complete_text_claude

def read_task_prompt(json_file_path):
    """
    Reads the task prompt from a JSON file and returns the task description and measurement.

    Parameters:
    - json_file_path (str): Path to the JSON file.

    Returns:
    - task_description (str): Description of the task.
    - measurement (str): Measurement associated with the task.
    """

    with open(json_file_path, 'r') as f:
        prompt_data = json.load(f)

    task_description = prompt_data['Task']
    measurement = prompt_data['Measurement']

    return task_description, measurement

initial_prompt = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Reflection: Thoughts on previous results and next steps. 
2. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
3. Solution: Propose a list of predicted genes to test separated by commas in this format: 1. <Gene name 1>, 2. <Gene name 2> ...
Do not include any genes from this prompt (since they're already tested).

"""

initial_prompt_gene_search = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Reflection: Thoughts on previous results and next steps. 
2. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
3. Gene Search: Name a gene to search for 10 most similar genes based on features. Only include the gene name itself after "2. Gene Search:".
4. Solution: Propose a list of predicted genes to test separated by commas in this format: 1. <Gene name 1>, 2. <Gene name 2> ...
Do not include any genes from this prompt (since they're already tested).

"""

initial_prompt_topk = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Reflection: Thoughts on previous results and next steps. 
2. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
3. Correlated Genes: Name a gene to search for 10 most correlated genes based on Pearson's correlation. Only include the gene name itself after "2. Correlated Genes:".
4. Solution: Propose a list of predicted genes to test separated by commas in this format: 1. <Gene name 1>, 2. <Gene name 2> ...
Do not include any genes from this prompt (since they're already tested).

"""

initial_prompt_rna = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Reflection: Thoughts on previous results and next steps. 
2. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
3. Active Tissues: Name a gene to search for the top 10 tissues where this gene is active, based on transcripts per million. Only include the gene name itself after "2. Active Tissues:".
4. Solution: Propose a list of predicted genes to test separated by commas in this format: 1. <Gene name 1>, 2. <Gene name 2> ...
Do not include any genes from this prompt (since they're already tested).

"""

initial_prompt_pathways = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Reflection: Thoughts on previous results and next steps. 
2. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
3. Reactome Pathways: Name a gene to search for the associated biological pathways. Only include the gene name itself after "2. Reactome Pathways:".
4. Solution: Propose a list of predicted genes to test separated by commas in this format: 1. <Gene name 1>, 2. <Gene name 2> ...
Do not include any genes from this prompt (since they're already tested).

"""

## Norman
initial_prompt_pairs_norman = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
2. Solution: Propose a list of predicted pairs of genes to test separated by commas in this format: 1. <Gene name 1> + <Gene name 2>, 2. <Gene name 3> + <Gene name 4>, 3... 

ONLY CHOOSE FROM this gene list ['AHR', 'ARID1A', 'ARRDC3', 'ATL1', 'BCORL1', 'BPGM', 'CBARP', 'CBFA2T3', 'CBL', 'CDKN1A', 'CDKN1B', 'CDKN1C', 'CEBPA', 'CEBPB', 'CEBPE', 'CELF2', 'CITED1', 'CKS1B', 'CLDN6', 'CNN1', 'CNNM4', 'COL1A1', 'COL2A1', 'CSRNP1', 'DLX2', 'DUSP9', 'EGR1', 'ELMSAN1', 'ETS2', 'FEV', 'FOSB', 'FOXA1', 'FOXA3', 'FOXF1', 'FOXL2', 'FOXL2NB', 'FOXO4', 'GLB1L2','HES7', 'HK2', 'HNF4A', 'HOXA13', 'HOXB9', 'HOXC13', 'IER5L', 'IGDCC3','IKZF3', 'IRF1', 'JUN', 'KIF18B', 'KLF1', 'LHX1', 'LYL1', 'MAML2','MAP2K3', 'MAP2K6', 'MAPK1', 'MEIS1', 'MIDN', 'NIT1', 'OSR2', 'POU3F2','PRDM1', 'PRTG', 'PTPN1', 'PTPN12', 'PTPN13', 'PTPN9', 'RHOXF2B','RP5-862P8.2', 'RREB1', 'S1PR2', 'SAMD1', 'SET', 'SGK1', 'SLC38A2','SLC4A1', 'SLC6A9', 'SNAI1', 'SPI1', 'TBX2', 'TBX3', 'TMSB4X', 'TP73','TSC22D1', 'UBASH3A', 'UBASH3B', 'ZBTB1', 'ZBTB10', 'ZBTB25', 'ZC3HAV1','ZNF318']

"""

initial_prompt_pairs = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
2. Reasoning: Explanations of the reasoning behind all the proposed combinations.
3. Solution: Propose a list of predicted pairs of genes to test separated by commas in this format: 1. <Gene name 1> + <Gene name 2>, 2. <Gene name 3> + <Gene name 4>, 3... 

ONLY CHOOSE FROM THIS GENE LIST: ['TNFRSF9', 'ZAP70', 'LHX6', 'EMP3', 'CD27', 'EBF2', 'GRAP2', 'VPS29', 'CBLB', 'IL2RG', 'PLCG2', 'CD3E', 'FOXQ1', 'OTUD7A', 'LIME1', 'DEF6', 'RPL26', 'NMT1', 'NFKB2', 'SLC16A1', 'ZEB2', 'PIK3AP1', 'PI4KB', 'ITPKB', 'MUC21', 'RELA', 'IL9R', 'EIF3K', 'RIPK3', 'PSTPIP1', 'CD28', 'IL2', 'TRIM21', 'PLCG1', 'RNF40', 'MAP3K12', 'CPSF4', 'LAT2', 'CD247', 'IL1R1', 'FOXL2', 'FOSB', 'WT1', 'ARHGAP15', 'AKAP12', 'TRAF3IP2', 'CD3G', 'RPL35', 'VAV1', 'RAC2', 'MYB', 'IFNGR2', 'TSC1', 'MAP3K7', 'TNFRSF1B', 'GRAP', 'SHOC2', 'HELZ2', 'FOXL2NB', 'IRX4', 'FPR2', 'IL2RB', 'SNRPC', 'KIDINS220', 'EP400', 'RPL38', 'PSMD4', 'JAK1', 'INPPL1', 'PTPRC', 'RNF20', 'LCK', 'SPTLC2', 'CD2', 'IFNG', 'RPL19', 'MAP4K1', 'FOXF1', 'ARHGDIB', 'APOBEC3D', 'GCSAML', 'SLAMF6', 'LAT', 'FOXO4', 'EOMES', 'FOSL1', 'LTBR', 'STAT3', 'TRAF6', 'ANXA2R', 'OTUD7B', 'SRP68', 'TBX21', 'ITPKA', 'PDGFRA', 'BICDL2', 'CEACAM1', 'MCM2', 'APOL2', 'SRP19', 'RPS7', 'TAF13', 'GATA3', 'TNFRSF1A', 'EIF3D', 'CD5', 'MCM3AP', 'JMJD1C', 'CAD', 'SLA2', 'WAS', 'CDKN2C', 'MUC1', 'ITK', 'CD3D', 'EMP1', 'DGKZ', 'IKZF3', 'BRD9', 'DEPDC7', 'NRF1', 'HGS', 'MAK16', 'LCP2']
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="perturb-genes-brief-Horlbeck",
                        help="task name")
    parser.add_argument("--log_dir", type=str, default="logs", help="script")
    parser.add_argument("--folder_name", type=str, default="temp", help="temp folder name")

    parser.add_argument("--run_name", type=str, default="exp", help="script "
                                                                     "name")
    parser.add_argument("--data_name", type=str, default='Horlbeck',
                        help="dataset name")
    parser.add_argument("--steps", type=int, default=6, help="number of steps")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--model", type=str, default='claude-v1',
                                                help="LLM choice")
    parser.add_argument("--python", type=str, default="/lfs/turing2/0/qhwang/miniconda3/envs/llm/bin/python", help="python command")
    parser.add_argument("--continue_research", type=str, default=None, help="continue from a previous run")
    parser.add_argument("--interactive_interval", type=int, default=None, help="interactive interval")
    parser.add_argument("--enable_help", type=bool, default=False, help="enable help")
    parser.add_argument("--use_gpt4", type=bool, default=False, help="use gpt4")
    parser.add_argument("--num_genes", type=int, default=32, help="number of "
                                                                 "genes to sample per round")
    parser.add_argument("--manual_prepare", type=bool, default=False, help="use gpt4")
    parser.add_argument("--prompt_tries", type=int, default=20)
    parser.add_argument("--critique", type=bool, default=False, help="critique")
    parser.add_argument("--gene_search", type=bool, default=False, help="gene_search")
    parser.add_argument("--gene_search_diverse", type=bool, default=False, help="gene_search")
    parser.add_argument("--lit_review", type=bool, default=False, help="perform literature review")
    parser.add_argument("--topk", type=bool, default=False, help="top k correlated gene search")
    parser.add_argument("--rna", type=bool, default=False, help="top tpm RNA-seq")
    parser.add_argument("--pathways", type=bool, default=False, help="pathway search")
    parser.add_argument("--enrichment", type=bool, default=False, help="enrichment analysis")
    parser.add_argument("--reactome", type=bool, default=False, help="enrichment on Reactome")
    parser.add_argument("--combinatorial", type=bool, default=False, help="combinatorial")
    parser.add_argument("--use_single_gene", type=bool, default=False, help="combinatorial")
    parser.add_argument("--csv_path", type=str, default=".", help="path to save achilles.csv")
    args = parser.parse_args()

    tools.DEVICE = args.device
    tools.PYTHON = args.python

    print("DEVICE:", tools.DEVICE)
    print("PYTHON:", tools.PYTHON)


# , 'Edit Script (Direct)''Work On Subtask',
    all_tools = ['Reflection', 'Arxiv Search']
    #all_tools = ['Reflection', 'Read File',]
    #all_tools = [ 'Copy File', 'List Files', 'Read File', 'Reflection',
    # 'Final Answer','Inspect Script Lines', 'Edit Script (AI)', 'Undo Edit Script', 'Execute Script']
    if args.enable_help: 
        all_tools.append('Request Help')

    low_level_tools = all_tools
    high_level_tools = all_tools

    # research_problem = "Find the part of code that customizes the model in train.py"
    # folder_name = "add_metric"

    # Task and measurement information
    task_description, measurement = read_task_prompt(
                            './datasets/task_prompts/'+args.data_name+'.json')


    # research_problem = "Create a eval.py file to evaluate the model trained with train.py over VQA-RAD dataset. The code for VQA-RAD dataset is in dataset_RAD.py."
    # folder_name = "add_metric"
    if args.task == "debug":
        research_problem = "Given a training script on a dataset train.py, improve upon the current performance of the model with a simple change."
        benchmark_name = "cifar10_training"

    elif args.task == "cifar10-training":
        research_problem = "Given a training script on a dataset train.py, improve upon the current model performance (trained with current hyperparmeters in train.py) for more than 10%. The training epochs should be within 10 to save time."
        benchmark_name = "cifar10_training"

    elif args.task == "perturb-genes":
        research_problem = "You are running a series of experiments to " \
                           "identify genes whose perturbation would " \
                           "most impact Interferon-gamma production. Given a " \
                           "list of experimental outcomes following the perturbation of some set of genes, " \
                           "the goal is to predict a set of new genes " \
                           "that would lead to extremely positive or " \
                           "extremely negative values of this score."
        benchmark_name = "perturb-genes"

        instructions = "\n Based on these results and your knowledge of biology, " \
                       "predict the next {} genes I should experimentally " \
                       "test, i.e. genes that show a strong log " \
                       "fold change in INF-gamma (whether strongly positive or strongly negative) " \
                       "upon being knocked out. IFN-gamma is a cytokine produced " \
                       "by CD4+ and CD8+ T cells that induces additional T " \
                       "cells. It might be worth exploring co-essential " \
                       "genes. Use HGNC gene naming convention.  " \
                       "DO PREDICT GENES THAT HAVE ALREADY BEEN TESTED " \
                       "".format(args.num_genes)

    elif args.task == "perturb-genes-brief":
        research_problem = "I'm planning to run a genome-wide CRISPR screen " \
                           "to {}. There are 18,939 possible  genes to perturb and I can only " \
                           "perturb {} genes at a time. For each " \
                           "perturbation, I'm able to measure out {} which " \
                           "will be referred to as the score. I can " \
                           "only do a few rounds of experimentation.".format(
                            task_description,  args.num_genes, measurement)

        benchmark_name = "perturb-genes-brief"

        instructions = "\n Based on these results and " \
                           "prior knowledge of biology, make the best " \
                       "possible prediction of the " \
                           "first {} genes that I should test to maximize " \
                           "the score. Use HGNC gene naming convention." \
                           "DO NOT PREDICT GENES THAT HAVE ALREADY BEEN TESTED"\
                           "".format(args.num_genes)

    elif args.task == "perturb-genes-brief-NormanGI":
        research_problem = "I'm planning to run a genome-wide CRISPR screen " \
                           "to {}. There are 92 possible genes to perturb and I can only " \
                           "perturb {} gene pairs at a time. For each " \
                           "perturbation, I'm able to measure out {} which " \
                           "will be referred to as the score. I can " \
                           "only do a few rounds of experimentation.".format(
                            task_description,  args.num_genes, measurement)

        benchmark_name = "perturb-genes-brief"

        instructions = "\n Based on these results and " \
                           "prior knowledge of biology, make the best " \
                       "possible prediction of the " \
                           "first {} genes that I should test to maximize " \
                           "the score. Use HGNC gene naming convention." \
                           "DO NOT PREDICT GENES THAT HAVE ALREADY BEEN TESTED"\
                           "".format(args.num_genes)

    elif args.task == "perturb-genes-brief-Horlbeck":
        research_problem = "I am interested in {}. There are 450 genes from which pairs of " \
                            "genes must be chosen. I can only perturb {} gene pairs at a " \
                            "time. For each perturbation, I'm able to measure out {} which will " \
                            "be referred to as the score.".format(task_description, args.num_genes, measurement)

        benchmark_name = "perturb-genes-brief"

        instructions = "\n Based on these results and using your prior knowledge of biology,"\
                       "can you suggest {} other combinations that may also show a synergistic" \
                       "effect upon perturbation. DO NOT PREDICT GENE PAIRS THAT HAVE ALREADY" \
                       "BEEN TESTED. Hint: genetic interactions are often found between" \
                       "functionally related genes".format(args.num_genes)

    elif args.task == "speed-up":
        research_problem = "Given a inference script inference.py, execute it to see the current generation speed per token and then try to improve it with accelerate library. The script is run on a single A100 GPU. Before you give the final answer, please ask yourself if there is any other way to improve the speed."
        benchmark_name = "speed_up"

    elif args.task == "add-metric":
        research_problem = "Create a eval.py file to evaluate the model trained with train.py over VQA-RAD dataset. The code for VQA-RAD dataset is in dataset_RAD.py."
        benchmark_name = "add_metric"

    elif args.task == "literature-review":
        research_problem = """Create a literature_review.py file to create a large language model(LLM) based AI system for the task of literature review. The LLM can be accessed using anthropic API with API key in claude_api_key.txt. Use it as in claude_example.py.
        Given ANY query on a research topic, e.g. Language model hallucination detection, the system should 
            - Call arxiv API (with reference in arxiv_API_reference.txt) with appropriate queries to get a list of related papers.
            - Use the LLM to summarize these papers and generate a report. The report should include the following information: 1) the list of papers; 2) the summary of each paper; 3) the conclusion of the literature review."""
        benchmark_name = "literature_review"

    elif args.task == "fix-literature-review":
        research_problem = """Fix literature_review.py file to create a large language model(LLM) based AI system for the task of literature review. The LLM can be accesed using anthropic API with API key in claude_api_key.txt. Use it as in claude_example.py.
        Given ANY query on a research topic, e.g. Language model hallucination detection, the system should 
            - Call arxiv API (with reference in arxiv_API_reference.txt) with appropriate queries to get a list of related papers.
            - Use the LLM to summarize these papers and generate a report. The report should include the following information: 1) the list of papers; 2) the summary of each paper; 3) the conclusion of the literature review."""
        benchmark_name = "literature_review"
    elif args.task == "bibtex-generation":
        research_problem = """Create a bibtex_generation.py file that contains an LLM based AI system for the task of bibtex generation. The LLM should be accessed through API as in LLM_example.py.
        Given ANY paragraph, e.g. "We use the method proposed in PromptTuning to train the Transformer model...", the system should 
            - Use the LLM to find all phrases and claims in the paragraph that require citations, e.g. PromptTuning and Transformer
            - Use google scholar API (with reference in google_scholar_API_reference.txt) with appropriate queries to find proper references and get the bibtex entries for the papers.
            - Finally, use LLM to generate the original paragraph with bibtex entries, e.g. "We use the method proposed in \cite{lester-etal-2021-power} to train the Transformer \cite{...} model..., as well as the bibtext entries for the referred papers.
            """
        benchmark_name = "bibtex-generation"

    else:
        raise ValueError("task not supported")



    #######################################################
    #                                                     # 
    #            Prepare Environment                      # 
    #                                                     #
    #######################################################

    # default workspace folder name
    folder_name = args.folder_name
    current_history = None
    if args.manual_prepare:
        print(f"Please prepare the folder {folder_name} manually and then press enter to continue.")
        input()

    elif args.continue_research is None:
        pass
        
        # # remove the folder if it exists
        # if os.path.exists(folder_name):
        #     shutil.rmtree(folder_name)

        # os.mkdir(folder_name)

        # # copy the benchmarks folder to folder_name
        # if os.path.exists(os.path.join("benchmarks", benchmark_name)):
        #     shutil.copytree(os.path.join("benchmarks", benchmark_name), folder_name)

        # # init research_log.log
        # with open(os.path.join(folder_name, "research_log.log"), "w") as f:
        #     f.write("")

        # # init python files in the folder
        # for file_name in os.listdir(folder_name):
        #     if file_name.endswith(".bak"):
        #         os.rename(os.path.join(folder_name, file_name), os.path.join(folder_name, file_name[:-4]))
        
        # # init backup folder and remove all content if it exists
        # if os.path.exists(os.path.join(folder_name, "backup")):
        #     shutil.rmtree(os.path.join(folder_name, "backup"))
        # os.mkdir(os.path.join(folder_name, "backup"))

    else:
        # restore backup folder
        if os.path.exists(os.path.join(args.continue_research, "folder_backup")):
            folder_name = folder_name + "_continued"
            shutil.copytree(os.path.join(args.continue_research, "folder_backup"), folder_name)

        # restore current_history.json
        with open(os.path.join(args.continue_research, "current_history.json") , "r") as f:
            current_history = json.load(f)
        
        # restore research_log.log
        shutil.copyfile(os.path.join(args.continue_research, "research_log.log"), os.path.join(folder_name, "research_log.log"))

        # restore python files in the folder
        for file_name in os.listdir(folder_name):
            if file_name.endswith(".bak"):
                os.rename(os.path.join(args.continue_research, file_name[:-4]), os.path.join(folder_name, file_name[:-4]))
        


    #######################################################
    #                                                     # 
    #           Prepare for main agent loop               # 
    #                                                     #
    #######################################################

    
    if args.gene_search:   
        initial_prompt = initial_prompt_gene_search
    if args.combinatorial:
        initial_prompt = initial_prompt_pairs
    if args.topk:
        initial_prompt = initial_prompt_topk
    if args.rna:
        initial_prompt = initial_prompt_rna
    if args.pathways:
        initial_prompt = initial_prompt_pathways
    if current_history is None:
        current_history = {
            "tool_names" : high_level_tools,
            "low_level_tools": low_level_tools,
            "high_level_tools": high_level_tools,
            "research_problem" : research_problem,
            "initial_prompt": initial_prompt,
            "actions" : [],
            "observations" : [],
            "folder_name": folder_name,
            "instructions": instructions
        }

    # execute_script("script_name:literature_review.py", "no", "literature_review")
    # exit()

    
    log_dir = os.path.join(args.log_dir +'_'+ args.data_name, args.run_name)
    print("Log directory: ", log_dir)
    agent_loop(current_history, args.steps, args.use_gpt4, log_dir, args)


    #######################################################
    #                                                     # 
    #           Clean up Environment                      # 
    #                                                     #
    #######################################################


    # save current history
    # with open(os.path.join(args.log_dir, args.run_name,
    # f"current_history.json"), "w") as f:
    #    json.dump(current_history, f)

    # save research_log.log
    #with open(os.path.join(folder_name, "research_log.log"), "r") as f:
    #    research_log = f.read()
    #with open(os.path.join(args.log_dir, args.run_name,
    # f"research_log.log"), "w") as f:
    #    f.write(research_log)

    # save all python files in the folder
    #for file_name in os.listdir(folder_name):
    #    if file_name.endswith(".py"):
    #        shutil.copyfile(os.path.join(folder_name, file_name),
    #        os.path.join(args.log_dir, args.run_name, file_name))

    # save the whole folder
    #try:
    #    shutil.copytree(folder_name, os.path.join(args.log_dir,
    #    args.run_name, "folder_backup"))
    #except:
    #    print("Error saving backup folder")



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
from screen import read_task_prompt
import tools
#from LLM import complete_text_openai, complete_text_claude


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

## Norman
initial_prompt_pairs_norman = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
2. Solution: Propose a list of predicted pairs of genes to test separated by commas in this format: 1. <Gene name 1> + <Gene name 2>, 2. <Gene name 3> + <Gene name 4>, 3... 

ONLY CHOOSE FROM this gene list ['AHR', 'ARID1A', 'ARRDC3', 'ATL1', 'BCORL1', 'BPGM', 'CBARP', 'CBFA2T3', 'CBL', 'CDKN1A', 'CDKN1B', 'CDKN1C', 'CEBPA', 'CEBPB', 'CEBPE', 'CELF2', 'CITED1', 'CKS1B', 'CLDN6', 'CNN1', 'CNNM4', 'COL1A1', 'COL2A1', 'CSRNP1', 'DLX2', 'DUSP9', 'EGR1', 'ELMSAN1', 'ETS2', 'FEV', 'FOSB', 'FOXA1', 'FOXA3', 'FOXF1', 'FOXL2', 'FOXL2NB', 'FOXO4', 'GLB1L2','HES7', 'HK2', 'HNF4A', 'HOXA13', 'HOXB9', 'HOXC13', 'IER5L', 'IGDCC3','IKZF3', 'IRF1', 'JUN', 'KIF18B', 'KLF1', 'LHX1', 'LYL1', 'MAML2','MAP2K3', 'MAP2K6', 'MAPK1', 'MEIS1', 'MIDN', 'NIT1', 'OSR2', 'POU3F2','PRDM1', 'PRTG', 'PTPN1', 'PTPN12', 'PTPN13', 'PTPN9', 'RHOXF2B','RP5-862P8.2', 'RREB1', 'S1PR2', 'SAMD1', 'SET', 'SGK1', 'SLC38A2','SLC4A1', 'SLC6A9', 'SNAI1', 'SPI1', 'TBX2', 'TBX3', 'TMSB4X', 'TP73','TSC22D1', 'UBASH3A', 'UBASH3B', 'ZBTB1', 'ZBTB10', 'ZBTB25', 'ZC3HAV1','ZNF318']

"""

## Horlbeck
initial_prompt_pairs = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
2. Solution: Propose a list of predicted pairs of genes to test separated by commas in this format: 1. <Gene name 1> + <Gene name 2>, 2. <Gene name 3> + <Gene name 4>, 3... 

ONLY CHOOSE FROM THIS GENE LIST: ['AARS2', 'AATF', 'ABCB7', 'ACTL6A', 'ACTR10', 'ADAT2', 'ADPRM', 'AFG3L2', 'ANAPC13', 'ANKZF1', 'APOOL', 'ARGLU1', 'ARIH1', 'ARL2', 'ASCC3', 'ASNA1', 'ATP1A1', 'ATP5A1', 'ATP5F1', 'ATP5J2', 'ATP6AP1', 'ATP6V1G1', 'ATXN10', 'AURKA', 'BDP1', 'BNIP1', 'BOD1L1', 'BTF3', 'BUB1B', 'BUB3', 'C11orf30', 'C14orf178', 'C14orf2', 'C6orf203', 'C9orf114', 'CACTIN', 'CAMLG', 'CAPZB', 'CARS2', 'CARS', 'CCDC84', 'CCNK', 'CCT8', 'CDC16', 'CDC23', 'CDC26', 'CDC27', 'CDC40', 'CDC73', 'CDCA8', 'CDK12', 'CDK1', 'CENPK', 'CENPM', 'CENPP', 'CENPW', 'CEP152', 'CEP192', 'CEP63', 'CHMP2A', 'CIRH1A', 'CIT', 'CKAP5', 'CLSPN', 'CMAS', 'CNOT1', 'COA3', 'COG3', 'COPE', 'COPS2', 'COPS4', 'COPS5', 'COQ2', 'COX16', 'COX5A', 'COX6C', 'COX7C', 'CSTF3', 'CTNNBL1', 'CYB5B', 'DAP3', 'DARS', 'DCAF7', 'DCTN2', 'DCTN4', 'DDX18', 'DDX46', 'DHODH', 'DHX37', 'DICER1', 'DLST', 'DNA2', 'DNTTIP2', 'DONSON', 'DYNC1H1', 'EBNA1BP2', 'EEF1G', 'EFR3A', 'EIF2B5', 'EIF3C', 'EIF3D', 'EIF3H', 'EIF3M', 'EIF4A1', 'EIF4G1', 'EIF5B', 'EIF6', 'ELP4', 'EMC1', 'EMC2', 'EMC4', 'EMC6', 'EMC7', 'EMG1', 'ERH', 'EXOSC2', 'EXOSC4', 'EXOSC6', 'EXOSC9', 'FAM208A', 'FAM50A', 'FAM98B', 'FARSA', 'FARSB', 'FBXO28', 'FDPS', 'FH', 'FIP1L1', 'GEMIN2', 'GEMIN4', 'GFER', 'GFI1B', 'GINS4', 'GNL2', 'GNL3L', 'GNPAT', 'GOLT1B', 'GPN3', 'GTF2E1', 'GTF2H1', 'GTF2H4', 'GTF3C3', 'GTPBP4', 'HBS1L', 'HCCS', 'HSCB', 'HUS1', 'HYPK', 'IARS', 'ICT1', 'IMPDH2', 'INTS3', 'INTS5', 'KARS', 'KCTD10', 'KDM1A', 'KIAA1731', 'KIF11', 'KIF14', 'LARS2', 'LARS', 'LEO1', 'LIAS', 'LONP1', 'LUC7L3', 'Ltn1', 'MAPKAP1', 'MARS2', 'MARS', 'MASTL', 'MAT2A', 'MAX', 'MBTPS2', 'MCM3AP', 'MCM3', 'MCM4', 'MED14', 'MED17', 'MED1', 'MED22', 'MED23', 'MED24', 'MED28', 'MED9', 'METAP2', 'METTL17', 'MFAP1', 'MINOS1-NBL1', 'MINOS1', 'MPHOSPH6', 'MRP63', 'MRPL10', 'MRPL13', 'MRPL15', 'MRPL16', 'MRPL18', 'MRPL19', 'MRPL22', 'MRPL24', 'MRPL27', 'MRPL32', 'MRPL33', 'MRPL36', 'MRPL37', 'MRPL39', 'MRPL3', 'MRPL42', 'MRPL43', 'MRPL46', 'MRPL50', 'MRPS10', 'MRPS11', 'MRPS14', 'MRPS16', 'MRPS18A', 'MRPS18B', 'MRPS21', 'MRPS23', 'MRPS25', 'MRPS27', 'MRPS28', 'MRPS30', 'MRPS35', 'MRPS5', 'MRPS9', 'MTBP', 'MTERFD1', 'MTHFD2', 'MTOR', 'MTPAP', 'MZT1', 'NAA25', 'NAA50', 'NACA', 'NAMPT', 'NAT10', 'NCAPD2', 'NCBP1', 'NDUFA2', 'NDUFA8', 'NDUFA9', 'NDUFB1', 'NDUFB2', 'NDUFB4', 'NEMF', 'NFE2L1', 'NFYC', 'NIP7', 'NOC2L', 'NOL10', 'NOL8', 'NOL9', 'NOLC1', 'NOP58', 'NSMCE1', 'NSMCE4A', 'NUBP1', 'NUDC', 'NUP43', 'NUP54', 'NUP85', 'NUTF2', 'OGFOD1', 'OPA1', 'ORAOV1', 'PAFAH1B1', 'PAXBP1', 'PDCD7', 'PDRG1', 'PDSS2', 'PELO', 'PET112', 'PFAS', 'PGD', 'PGK1', 'PHF5A', 'PITRM1', 'PMF1', 'PMPCB', 'PMVK', 'PNISR', 'PNN', 'POLA1', 'POLD1', 'POLD3', 'POLE2', 'POLE', 'POLN', 'POLR2B', 'POLR2K', 'POLR3A', 'POLR3B', 'PPAT', 'PPCS', 'PPIE', 'PPP2R1A', 'PPP2R2A', 'PPWD1', 'PRIM2', 'PRMT5', 'PRPF18', 'PSMB6', 'PSMB7', 'PSMC1', 'PSMC2', 'PSMC6', 'PSMD12', 'PSMD1', 'PSMD4', 'PSMD6', 'PSPH', 'PTCD1', 'PTCD3', 'PTTG1', 'PWP1', 'QARS', 'RAD51', 'RAE1', 'RBBP8', 'RBFA', 'RBM17', 'RBM22', 'RBM25', 'REXO2', 'RFC4', 'RFC5', 'RINT1', 'RNF20', 'RNGTT', 'ROMO1', 'RPA3', 'RPF2', 'RPL10', 'RPL11', 'RPL13', 'RPL23', 'RPL24', 'RPL27', 'RPL28', 'RPL30', 'RPL32', 'RPL35', 'RPL36A', 'RPL3', 'RPL5', 'RPL8', 'RPLP1', 'RPLP2', 'RPP30', 'RPP38', 'RPS18', 'RPS27A', 'RPS28', 'RPSA', 'RPTOR', 'RRM1', 'RTF1', 'SAP18', 'SCO1', 'SCYL1', 'SDHC', 'SEC22B', 'SEH1L', 'SF3B1', 'SFXN1', 'SGOL1', 'SKA3', 'SKIV2L2', 'SKIV2L', 'SLC35A1', 'SLC39A9', 'SLC7A1', 'SLC7A6OS', 'SMC5', 'SMNDC1', 'SNAPC1', 'SNIP1', 'SNRNP70', 'SNRPB', 'SNRPC', 'SNRPD3', 'SNRPG', 'SPAG7', 'SPATA5', 'SRFBP1', 'SRP19', 'SRSF7', 'SSBP1', 'SSB', 'STXBP4', 'SYVN1', 'TAF1A', 'TAF1', 'TAF2', 'TAZ', 'TBC1D31', 'TBCB', 'TCF25', 'TCOF1', 'TFB1M', 'TGIF2', 'TIMELESS', 'TIMM22', 'TIMM23B', 'TIMM9', 'TIPIN', 'TMEM261', 'TOMM22', 'TRA2B', 'TRMT5', 'TRNAU1AP', 'TRPM7', 'TRRAP', 'TSEN2', 'TTC4', 'TTI2', 'TUBB', 'TUBGCP3', 'TUBGCP4', 'TUBGCP5', 'UBA2', 'UBA3', 'UMPS', 'UQCRB', 'USE1', 'USMG5', 'VPS29', 'VPS4A', 'VPS72', 'WARS2', 'WDR36', 'WDR61', 'WDR70', 'WDR77', 'WDR7', 'XPO1', 'XRN2', 'YTHDC1', 'ZCCHC9', 'ZMAT2', 'ZMAT5', 'ZNF511', 'ZNF574', 'ZNF598', 'ZNF830', 'ZNHIT6', 'ZNRD1', 'ZWINT']
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
    parser.add_argument("--lit_review", type=bool, default=False, help="perform literature review")
    parser.add_argument("--combinatorial", type=bool, default=True, help="combinatorial")
    
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
        
        # remove the folder if it exists
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)

        os.mkdir(folder_name)

        # copy the benchmarks folder to folder_name
        if os.path.exists(os.path.join("benchmarks", benchmark_name)):
            shutil.copytree(os.path.join("benchmarks", benchmark_name), folder_name)

        # init research_log.log
        with open(os.path.join(folder_name, "research_log.log"), "w") as f:
            f.write("")

        # init python files in the folder
        for file_name in os.listdir(folder_name):
            if file_name.endswith(".bak"):
                os.rename(os.path.join(folder_name, file_name), os.path.join(folder_name, file_name[:-4]))
        
        # init backup folder and remove all content if it exists
        if os.path.exists(os.path.join(folder_name, "backup")):
            shutil.rmtree(os.path.join(folder_name, "backup"))
        os.mkdir(os.path.join(folder_name, "backup"))

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




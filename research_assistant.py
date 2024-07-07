import os
import json
import argparse
from tools import GenePerturbAgent


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

## Combination task: Horlbeck
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

    parser.add_argument("--run_name", type=str, default="exp", help="script "
                                                                     "name")
    parser.add_argument("--data_name", type=str, default='Horlbeck',
                        help="dataset name")
    parser.add_argument("--steps", type=int, default=6, help="number of steps")
    parser.add_argument("--model", type=str, default='claude-v1',
                                                help="LLM choice")
    parser.add_argument("--continue_research", type=str, default=None, help="continue from a previous run")
    parser.add_argument("--interactive_interval", type=int, default=None, help="interactive interval")
    parser.add_argument("--enable_help", type=bool, default=False, help="enable help")
    parser.add_argument("--use_gpt4", type=bool, default=False, help="use gpt4")
    parser.add_argument("--num_genes", type=int, default=32, help="number of "
                                                                 "genes to sample per round")
    parser.add_argument("--manual_prepare", type=bool, default=False, help="use gpt4")
    parser.add_argument("--prompt_tries", type=int, default=20)
    parser.add_argument("--critique", type=bool, default=False, help="AI critic")
    parser.add_argument("--gene_search", type=bool, default=False, help="gene search")
    parser.add_argument("--gene_search_diverse", type=bool, default=False, help="gene search using diversity mode instead of similarity mode")
    parser.add_argument("--lit_review", type=bool, default=False, help="perform literature review")
    parser.add_argument("--combinatorial", type=bool, default=False, help="combinatorial")
    
    args = parser.parse_args()

    # Task and measurement information
    task_description, measurement = read_task_prompt(
                            './datasets/task_prompts/'+args.data_name+'.json')

    if args.task == "perturb-genes":
        research_problem = "I'm planning to run a CRISPR screen " \
                           "to {}. There are 18,939 possible  genes to perturb and I can only " \
                           "perturb {} genes at a time. For each " \
                           "perturbation, I'm able to measure out {} which " \
                           "will be referred to as the score. I can " \
                           "only do a few rounds of experimentation.".format(
                            task_description,  args.num_genes, measurement)

        benchmark_name = "perturb-genes"

        instructions = "\n Based on these results and " \
                           "prior knowledge of biology, make the best " \
                       "possible prediction of the " \
                           "first {} genes that I should test to maximize " \
                           "the score. Use HGNC gene naming convention." \
                           "DO NOT PREDICT GENES THAT HAVE ALREADY BEEN TESTED"\
                           "".format(args.num_genes)

    elif args.task == "perturb-genes-Horlbeck":
        research_problem = "I am interested in {}. There are 450 genes from which pairs of " \
                            "genes must be chosen. I can only perturb {} gene pairs at a " \
                            "time. For each perturbation, I'm able to measure out {} which will " \
                            "be referred to as the score.".format(task_description, args.num_genes, measurement)

        benchmark_name = "perturb-genes"

        instructions = "\n Based on these results and using your prior knowledge of biology,"\
                       "can you suggest {} other combinations that may also show a synergistic" \
                       "effect upon perturbation. DO NOT PREDICT GENE PAIRS THAT HAVE ALREADY" \
                       "BEEN TESTED. Hint: genetic interactions are often found between" \
                       "functionally related genes".format(args.num_genes)

    else:
        raise ValueError("task not supported")



    #######################################################
    #                                                     # 
    #           Prepare for main agent loop               # 
    #                                                     #
    #######################################################

    
    if args.gene_search:   
        initial_prompt = initial_prompt_gene_search
    if args.combinatorial:
        initial_prompt = initial_prompt_pairs
            
    current_history = {
        "research_problem" : research_problem,
        "initial_prompt": initial_prompt,
        "actions" : [],
        "observations" : [],
        "instructions": instructions
    }

    
    log_dir = os.path.join(args.log_dir +'_'+ args.data_name, args.run_name)
    print("Log directory: ", log_dir)
    experiment_agent = GenePerturbAgent(current_history, args.steps, args.use_gpt4, log_dir, args)
    experiment_agent.run_experiment()
    #agent_loop(current_history, args.steps, args.use_gpt4, log_dir, args)




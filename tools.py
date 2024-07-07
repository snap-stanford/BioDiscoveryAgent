import os
import re
from LLM import complete_text
import anthropic
import numpy as np
import pandas as pd
from tqdm import tqdm
from get_lit_review import get_lit_review

def parse_action_input(s, entries):
    s = s.split("{")[1].split("}")[0].strip()
    pattern = ""
    for e in entries:
        pattern += f'"{e}":([\s\S]*),\s*'
    pattern = pattern[:-4]
    result = re.search(pattern, s, re.MULTILINE)
    if result is None:
        raise Exception("Invalid: " + s)
    # stripe each entry
    return [r.strip().strip('\"') for r in result.groups()]
  
    
def gene_search_f(gene_name, gene_search_diverse, **kwargs):
    print(gene_name)
    df = pd.read_csv("/dfs/user/yhr/AI_RA/research_assistant/datasets/features/achilles.csv")
    df = df.rename(lambda x : x.split(" (")[0], axis='columns')
    df.drop(columns=["DepMap_ID"], inplace=True)
    df.dropna(inplace=True, axis='rows')
    if gene_name not in df.columns:
        return f"Gene {gene_name} not found"
    if gene_search_diverse:
        return ", ".join((df[gene_name].dot(df) / (np.linalg.norm(df, axis=0) * np.linalg.norm(df[gene_name]))).sort_values(ascending=True)[:50].index.tolist())
    else:
        return ", ".join((df[gene_name].dot(df) / (np.linalg.norm(df, axis=0) * np.linalg.norm(df[gene_name]))).sort_values(ascending=False)[:11].index.tolist()[1:])
    

def parse_entries(s, entries):
    pattern = ""
    for e in entries:
        e = e.replace("[", "\[").replace("]", "\]")
        pattern += f"{e}:([\s\S]*)"
    result = re.search(pattern, s, re.MULTILINE)
    if result is None:
        raise Exception("Invalid: " + s)
    # stripe each entry
    parsed = [r for r in result.groups()]
    return {e: parsed[idx]  for idx, e in enumerate(entries)}


def print_action(entries):
    return "".join([ k + ": " + v for k,v in  entries.items()])


def summarize_remaining_genes(all_genes, summary_size=20, bs=1000):

    blocks = [all_genes[i:i + bs] for i in range(0, len(all_genes), bs)]
    abridged_list = []
    for idx, b in tqdm(enumerate(blocks), total=len(blocks)):
        start_gene_number = bs * idx + 1
        end_gene_number = bs * idx + 1 + bs

        prompt = f"""
                The full gene list is too long. Given this (partial) observation 
                from gene {start_gene_number} to gene {end_gene_number}: 
                ``` 
                {b}
                ```
                Generate a shorter list containing {summary_size} 
                of these genes. Try to prioritize genes belonging to 
                distinct pathways such that different biological 
                processes and functions are comprehensively represented. 
                Only print out the genes separated by commans. Do not 
                include any additional information or explanation. Do 
                not include any gene that is guessed rather than 
                directly present in the list.
                """
        completion = complete_text(prompt, model="claude-1", log_file=None)
        abridged_list.append(completion)

    abridged_list = ','.join(abridged_list)
    abridged_list = abridged_list.split(',')
    abridged_list = [x.strip(' ') for x in abridged_list]

    return abridged_list


def gene_choices_prompt(prompt, num_genes_pick, remaining_genes):

    index = prompt.find("Please add")
    if index != -1:
        prompt = prompt[:index]
    else:
        prompt = ''

    prompt += " Please add {} more genes to this list from {}." \
                    "\n Remember not to include any previously tested " \
                    "genes. Begin this list with the word 'Solution:' " \
                "".format(num_genes_pick, remaining_genes)

    return prompt
    
    
def process_valid_output(gene_next_sample, curr_sample, gene_sampled,
                         dropped_genes, args):

        new_genes_pred = list(set(gene_next_sample) -
                              set(gene_sampled) -
                              set(curr_sample))
        print('New genes predicted:', len(new_genes_pred))
        curr_sample = curr_sample + new_genes_pred

        new_prompt = ''
        if len(curr_sample) < args.num_genes:
            new_prompt += "\n You have so far predicted {} out of the " \
                      "required {} genes for this round. These " \
                      "were: \n".format(len(curr_sample), args.num_genes)
            new_prompt += str(curr_sample)
            new_prompt += "\n Please add {} more genes to this list. " \
                      "Remember not to include previously " \
                      "tested genes including: \n ".format(args.num_genes - len(
                curr_sample))
            new_prompt += str(list(dropped_genes))
            return curr_sample, new_prompt

        else:
            return curr_sample, None

class GenePerturbAgent(object):
    def __init__(self, current_history, steps, use_gpt4, log_dir, args):
        self.current_history = current_history
        self.steps = steps
        self.use_gpt4 = False  # as per the original code
        self.log_dir = log_dir
        self.log_file = None
        self.args = args
        self.research_problem = current_history["research_problem"]
        self.valid_format_entries = ["Solution"]
        self.hits_history = []
        self.lit_review_summary = ""
        self.ground_truth, self.all_hit_genes = self.load_datasets()
        self.measured_genes = self.ground_truth.index.values
        self.gene_sampled = []
        self.initialize_log_dir()
        self.curr_step = 0

    def load_datasets(self):
        ground_truth_path = f'./datasets/ground_truth_{self.args.data_name}.csv'
        hit_genes_path = f'./datasets/topmovers_{self.args.data_name}.npy'

        if not self.args.combinatorial:
            ground_truth = pd.read_csv(ground_truth_path, index_col=0)
            all_hit_genes = np.load(hit_genes_path)
        else:
            import ast
            ground_truth = pd.read_csv(ground_truth_path)
            ground_truth['Gene_pairs'] = ground_truth['Gene_pairs'].apply(ast.literal_eval)
            ground_truth.set_index('Gene_pairs', inplace=True)
            all_hit_genes = np.load(hit_genes_path)
            all_hit_genes = [tuple(l) for l in all_hit_genes]

        return ground_truth, all_hit_genes

    def initialize_log_dir(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def write_log(self, message):
        with open(os.path.join(self.log_dir, "main_log"), "a", 1) as f:
            f.write(message + "\n")

    def generate_prompt(self, gene_readout=None, hits=None):
        prompt = f'Step {self.curr_step}\n '
        prompt += self.current_history["initial_prompt"].format(research_problem=self.research_problem)
        prompt += "\nNow let's start!\n\n"
        if self.curr_step != 0 and gene_readout is not None:
            prompt += self.append_experiment_results(gene_readout, hits)
        prompt += self.current_history["instructions"]
        return prompt

    def append_experiment_results(self, gene_readout, hits):
        if len(gene_readout) < 1500:
            return (
                f"\n This is not your first round. All tested genes and their measured log fold change are: \n"
                f"{gene_readout.drop(hits).to_string()}\n"
                f"You have successfully identified {len(hits)} hits so far over all experiment cycles! The results for the hits are: \n"
                f"{self.ground_truth.loc[hits].to_string()}"
            )
        else:
            negative_summary = self.summarize_results(gene_readout.drop(hits), "negative")
            positive_summary = self.summarize_results(self.ground_truth.loc[hits], "positive")
            return (
                f"\n This is not your first round. The summary of all tested genes and their measured log fold change are: \n"
                f"{negative_summary}\n"
                f"The summary of these hits is the following: {positive_summary}\n"
                f"Keep this in mind while choosing genes to be perturbed for the next round as they should also have similar properties."
            )

    def summarize_results(self, data, result_type):
        summary_prompt = f"{self.research_problem}\n Till now, these are all tested genes that are not hits along with their scores: \n{data.to_string()}"
        summary_prompt += "\n Summarize this in a few lines to find some common pattern in these which will aid in the next steps of experimental design to maximize your cumulative hits."
        log_file = os.path.join(self.log_dir, f"step_{self.curr_step}_log_{result_type}_sum.log")
        return complete_text(summary_prompt, model="claude-1", log_file=log_file)

    def perform_lit_review(self, lit_review_prompt):
        lit_review_prompt += f"\nYou might have already some literature review information as provided below. Try to gather information which is not repetitive to what you have and finally helps the most in solving the research problem at hand. \n {self.lit_review_summary} \n "
        self.lit_review_summary += get_lit_review(str(lit_review_prompt), model=self.args.model, max_number=4)

    def save_sampled_genes(self, gene_sampled, curr_sample):
        all_sampled_so_far = gene_sampled + curr_sample[:self.args.num_genes]
        np.save(os.path.join(self.log_dir, f'sampled_genes_{self.curr_step + 1}.npy'), all_sampled_so_far)

    def process_completion(self, prompt, gene_sampled, curr_sample, log_file):
        
        prompt_try = str(prompt)
        for itr in range(self.args.prompt_tries):
            if self.use_gpt4:
                completion = complete_text_gpt4(prompt_try, stop_sequences=["Observation:"], log_file=log_file)
            else:
                completion = complete_text(prompt_try, model=self.args.model, log_file=log_file)
                if "Gene Search:" in completion:
                    completion_pre = completion.split("4. Solution:")[0]
                    completion_pre += "Gene Search Result:" + gene_search_f(completion_pre.split("Gene Search:")[1].strip(), self.args.gene_search_diverse)
                    completion_post = complete_text(prompt_try + completion_pre + "\n\n3. Solution:", model=self.args.model, log_file=log_file)
                    completion = completion_pre + "\n\n4. Solution:" + completion_post

            try:
                entries = parse_entries(completion, [e.strip() for e in self.valid_format_entries])
                valid_format = True
            except:
                valid_format = False

            if not valid_format:
                print(itr, 'Invalid output')
                prompt_update = ''
            else:
                pred_genes = [p.strip(' \n[]') for p in entries['Solution'].replace("\n", ",").split(',')]
                pred_genes = [p.split('.')[-1].strip(' ') for p in pred_genes]
                if self.args.combinatorial:
                    pred_genes = [tuple(sorted(s.split(" + "))) for s in pred_genes]

                gene_next_sample = list(set(pred_genes).intersection(set(self.measured_genes)))
                dropped_genes = set(pred_genes) - set(gene_next_sample)
                curr_sample, prompt_update = process_valid_output(gene_next_sample, curr_sample, gene_sampled, dropped_genes, self.args)

                if prompt_update is None:
                    self.save_sampled_genes(gene_sampled, curr_sample)
                    break

                if itr >= self.args.prompt_tries - 4:
                    if genes_remain_summary is None:
                        num_genes_pick = self.args.num_genes - len(curr_sample)
                        genes_remain = list(set(self.measured_genes).difference(set(gene_sampled)))
                        genes_remain_summary = summarize_remaining_genes(genes_remain)
                    else:
                        genes_remain_summary = list(set(genes_remain_summary).difference(set(curr_sample)))
                    prompt_update = gene_choices_prompt(prompt_update, num_genes_pick, genes_remain_summary)

            if itr == self.args.prompt_tries - 1:
                np.save(os.path.join(self.log_dir, f'sampled_genes_{self.curr_step + 1}.npy'), gene_sampled)
            else:
                prompt_try = prompt + prompt_update

    def run_experiment(self):
        self.write_log("================================Start=============================")
        self.write_log(self.current_history["initial_prompt"].format(research_problem=self.research_problem))
        
        for self.curr_step in range(self.steps):
            if self.curr_step != 0:
                self.gene_sampled = list(np.load(os.path.join(self.log_dir, f'sampled_genes_{self.curr_step}.npy')))
                if self.args.combinatorial:
                    self.gene_sampled = [tuple(l) for l in self.gene_sampled]
                gene_readout = self.ground_truth.loc[self.gene_sampled]
                hits = list(set(self.gene_sampled).intersection(self.all_hit_genes))
                print(self.curr_step, 'Number of cumulative hits:', str(len(hits)))

            prompt = self.generate_prompt(gene_readout if self.curr_step != 0 else None, 
                                          hits if self.curr_step != 0 else None)
            log_file = os.path.join(self.log_dir, f"step_{self.curr_step}_log.log")
            curr_sample = []
            genes_remain_summary = None

            if self.curr_step < 5 and self.args.lit_review:
                self.perform_lit_review(prompt.split("Always respond")[0])
            self.process_completion(prompt, self.gene_sampled, curr_sample, log_file)

            if not self.args.critique:
                continue

            critique_prompt = self.generate_critique_prompt(curr_sample)
            self.process_completion(prompt, self.gene_sampled, curr_sample)

    def generate_critique_prompt(self, curr_sample):
        prompt_c = f"""You are a scientist working on problems in drug discovery.
Research Problem: {self.research_problem}
"""
        if self.curr_step != 0:
            prompt_c += (
                f"\n All tested genes so far and their measured log fold change are: \n{self.ground_truth.drop(self.hits).to_string()}"
                f"\n The results for the hits are: \n{self.ground_truth.loc[self.hits].to_string()}"
            )
        prompt_c += (
            f"\n\nNow for the next round of experiment your students are planning on testing the following genes: \n{str(curr_sample[:self.args.num_genes])}"
            f"""\n\nAs an advisor, please critique this plan and suggest some changes to it. Use this format:
1. Critique: include all relevant details of the critique.
2. Updated Solution: Give an updated selection of {self.args.num_genes} genes based on the critique separated by commas in this format:: 1. <Gene name 1>, 2. <Gene name 2> ...
Please do not critique/make changes if there is no need to make a change.
"""
        )
        return prompt_c


            

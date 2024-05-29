import os
import re
from LLM import complete_text, complete_text_claude
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


def agent_loop(current_history, steps, use_gpt4, log_dir, args):
    valid_format_entires = ["Solution"]
    
    use_gpt4 = False


    research_problem = current_history["research_problem"]
    if not args.combinatorial:
        ground_truth = pd.read_csv('./datasets/ground_truth_' + args.data_name + '.csv',
                                index_col=0)
        all_hit_genes = np.load('./datasets/topmovers_'+ args.data_name + '.npy')
    else:
        import ast  # ast.literal_eval safely evaluates a string as a Python literal

        ground_truth = pd.read_csv('./datasets/ground_truth_' + args.data_name + '.csv')
        # Convert the string representation of the tuples back to actual tuples
        ground_truth['Gene_pairs'] = ground_truth['Gene_pairs'].apply(ast.literal_eval)
        ground_truth.set_index('Gene_pairs', inplace=True)
        
        all_hit_genes = np.load('./datasets/topmovers_'+ args.data_name + '.npy')
        all_hit_genes = [tuple(l) for l in all_hit_genes]


    measured_genes = ground_truth.index.values
    gene_sampled = []
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, "main_log") , "w", 1) as f:
        f.write("================================Start=============================\n")
        last_steps = 3
        f.write(current_history["initial_prompt"].format(research_problem=research_problem) + "\n")
        hits_history = []
        lit_review_summary = ""
        for curr_step in range(steps):
            
            #curr_step = len(current_history["actions"])

            if curr_step !=0:
                ## Add experimental result from last run to prompt
                gene_sampled = list(np.load(log_dir + '/sampled_genes_'+str(
                            curr_step)+'.npy'))
                if args.combinatorial:
                    gene_sampled = [tuple(l) for l in gene_sampled]
                gene_readout = ground_truth.loc[gene_sampled]

                ## Get list of hits
                hits = list(set(gene_sampled).intersection(all_hit_genes))
                print(curr_step, 'Number of cumulative hits:', str(len(hits)))


            prompt = 'Step {}\n '.format(curr_step)
            prompt += current_history["initial_prompt"].format(research_problem=research_problem)
            if args.lit_review:
                lit_review_prompt = current_history["initial_prompt"].format(research_problem=research_problem).split("Always respond")[0]

            prompt += "\nNow let's start!\n\n"
            
            if curr_step == 0:
                pass
                
            else:
                if len(gene_readout) < 1500:
                    ## Append experiment results to the current prompt
                    prompt += "\n This is not your first round. All tested genes and " \
                            "their measured log fold change are: \n"\
                            + gene_readout.drop(hits).to_string()

                    prompt +=  "\n You have successfully identified {} hits so " \
                                "far over all experiment cycles! The results for the " \
                                "hits are: \n".format(len(hits)) + \
                                ground_truth.loc[hits].to_string()
                                
                    if args.lit_review:
                        lit_review_prompt +=  "\n You have successfully identified {} hits so " \
                            "far over all experiment cycles! The results for the " \
                            "hits are: \n".format(len(hits)) + \
                            ground_truth.loc[hits].to_string()

                    hits_history.append(len(hits))
                else:
                    # summarize the results
                    non_hit_sum_prompt = research_problem + "\n Till now, these are all tested genes that are not hits along with their scores: \n" + gene_readout.drop(hits).to_string()
                    non_hit_sum_prompt += "\n Summarize this in a few lines to find some common pattern in these which will aid in the next steps of experimental design to maximize your cumulative hits."
                    sum_log_file = os.path.join(log_dir , f"step_{curr_step}_log_neg_sum.log")
                    negative_examples_summary = complete_text(non_hit_sum_prompt, model = "claude-1", log_file = sum_log_file)

                    
                    hit_sum_prompt = research_problem + "\n Till now, you have identified the following genes as hits along with their scores: \n" + ground_truth.loc[hits].to_string()
                    hit_sum_prompt += "\n Summarize this in a few lines to find some common pattern in these which will aid in the next steps of experimental design to maximize your cumulative hits."
                    sum_log_file = os.path.join(log_dir , f"step_{curr_step}_log_pos_sum.log")
                    positive_examples_summary = complete_text(hit_sum_prompt, model = "claude-1", log_file = sum_log_file)

                    prompt += "\n This is not your first round. The summary of all tested genes and " \
                            "their measured log fold change are: \n" + negative_examples_summary
                    prompt += "\n The summary of these hits is the following: " + positive_examples_summary
                    prompt += "\n Keep this in mind while choosing genes to be perturbed for the next round as they should also have similar properties."

                    prompt += "\n Till now, the progression of the number of cumulative hits is as follows: " + ", ".join(map(str, hits_history))
                    prompt += "\n If you see the number of hits not increasing a lot over the past few rounds, rethink your design strategy to try to maximize these. One possibility could be that you are exploiting only one mode in the distribution and you might want to try some very different types of genes in order to find some different interesting possible pathways." 
                    prompt +=  "\n You have successfully identified {} hits so " \
                                "far over all experiment cycles! You will not be shown the results until " \
                                "the end of all the rounds, so design your strategy accordingly. \n".format(len(hits))
                    hits_history.append(len(hits))

                prompt += current_history["instructions"]



            # prompting
            f.write("Step " + str(curr_step) + ":\n")

            if curr_step < 5 and args.lit_review:
                print("starting literature review")
                lit_review_prompt += f"\nYou might have already some literature review information as provided below. Try to gather information which is not repititive to what you have and finally helps the most in solving the research problem at hand. \n {lit_review_summary} \n "
                lit_review_summary += get_lit_review(str(lit_review_prompt), model=args.model, max_number=4)
                print("finished literature review")
                prompt += f"\n You have done some literature review till now and have the following information at your disposal which you may use to make your predictions: \n {lit_review_summary}"

            log_file = os.path.join(log_dir , f"step_{curr_step}_log.log")
            prompt_try = str(prompt)

            curr_sample = []
            genes_remain_summary = None

            ## Help GPT count and not repeat genes
            for itr in range(args.prompt_tries):
                if use_gpt4:
                    import pdb; pdb.set_trace()
                    completion = complete_text_gpt4(prompt_try, stop_sequences=[
                                        "Observation:"], log_file=log_file)
                else:
                    
                    completion = complete_text(prompt_try, model = args.model, log_file=log_file)

                    if "Gene Search:" in completion:
                        completion_pre = completion.split("4. Solution:")[0] 
                        # execute action gene search
                        completion_pre = completion_pre + "Gene Search Result:" + gene_search_f(completion_pre.split("Gene Search:")[1].strip(), args.gene_search_diverse)
                        
                        
                        completion_post = complete_text(prompt_try + anthropic.AI_PROMPT +  completion_pre + "\n\n3. Solution:", model = args.model, ai_prompt = "", log_file=log_file)
                        completion = completion_pre + "\n\n4. Solution:" + completion_post

                # parse the action and action input
                try:
                    entries = parse_entries(completion,
                                        [e.strip() for e in
                                         valid_format_entires])
                    valid_format = True
                except:
                    valid_format = False

                if not valid_format:
                    print(itr, 'Invalid output')
                    prompt_ubpdate = '' # this will remove prompt_update from the prompt!!

                else:
                    ## Save predicted gene list
                    pred_genes = entries['Solution'].replace("\n", ",").split(',')
                    pred_genes = [p.strip(' \n[]') for p in pred_genes]
                    pred_genes = [p.split('.')[-1].strip(' ') for p in
                                  pred_genes]
                    
                    if args.combinatorial:
                        pred_genes = [tuple(sorted(s.split(" + "))) for s in pred_genes]

                    gene_next_sample = list(set(pred_genes).intersection(set(
                        measured_genes)))
                    print('Dropped genes:',
                          len(pred_genes) - len(gene_next_sample))
                    dropped_genes = set(pred_genes) - set(gene_next_sample)
                    curr_sample, prompt_update = \
                        process_valid_output(gene_next_sample, curr_sample,
                                             gene_sampled, dropped_genes, args)

                    if prompt_update is None:
                        all_sampled_so_far = gene_sampled + curr_sample[:args.num_genes]
                        np.save(log_dir + '/sampled_genes_' + str(curr_step + 1) +
                            '.npy', all_sampled_so_far)
                        break
                    
                    if itr >= args.prompt_tries - 4:
                        
                        if genes_remain_summary is None:
                            # Start choosing from gene list instead of random sample
                            num_genes_pick = args.num_genes - len(curr_sample)
                            genes_remain = list(set(measured_genes).difference(set(gene_sampled)))
                            genes_remain_summary = summarize_remaining_genes(genes_remain)
                        else:
                            genes_remain_summary = list(set(
                                genes_remain_summary).difference(set(curr_sample)))
                        prompt_update = gene_choices_prompt(prompt_update,
                                            num_genes_pick, genes_remain_summary)
                        
                if itr == args.prompt_tries-1:
                    np.save(log_dir + '/sampled_genes_' + str(curr_step + 1) +
                            '.npy', gene_sampled)
                else:
                    prompt_try = prompt + prompt_update
            
            if not args.critique:
                continue
            log_file = os.path.join(log_dir , f"step_{curr_step}_critique_log.log")
            prompt_c = f"""You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}
            
"""
            if curr_step == 0:
                pass
                
            else:
                prompt_c += "\n All tested genes so far and " \
                            "their measured log fold change are: \n"\
                            + gene_readout.drop(hits).to_string()

                prompt_c +=  "\n The results for the hits are: \n".format(len(hits)) + \
                                ground_truth.loc[hits].to_string()
            
            prompt_c += "\n\nNow for the next round of experiment your students are planning on testing the following genes: \n" + str(curr_sample[:args.num_genes])
            prompt_c += f"""\n\nAs an advisor, please critique this plan and suggest some changes to it. Use this format: 
1. Critique: include all relevant details of the critique.  
2. Updated Solution: Give an updated selection of {args.num_genes} genes based on the critique separated by commas in this format:: 1. <Gene name 1>, 2. <Gene name 2> ... \n

Please do not critique/make changes if there is no need to make a change.

"""
            
            prompt_try = str(prompt_c)
            curr_sample = []
            ## Help GPT count and not repeat genes
            for itr in range(args.prompt_tries):
                if use_gpt4:
                    import pdb; pdb.set_trace()
                    completion = complete_text_gpt4(prompt_try, stop_sequences=[
                                        "Observation:"], log_file=log_file)
                else:
                    completion = complete_text(prompt_try, model = args.model, log_file=log_file)

                # parse the action and action input
                try:
                    entries = parse_entries(completion,
                                        [e.strip() for e in valid_format_entires])
                    valid_format = True
                except:
                    valid_format = False

                if not valid_format:
                    print(itr, 'Invalid output')
                    prompt_update = ''

                else:
                    ## Save predicted gene list
                    pred_genes = entries['Solution'].split(',')
                    pred_genes = [p.strip(' \n[]') for p in pred_genes]
                    pred_genes = [p.split('.')[-1].strip(' ') for p in
                                  pred_genes]
                    gene_next_sample = list(set(pred_genes).intersection(set(
                        measured_genes)))
                    print('Dropped genes:',
                          len(pred_genes) - len(gene_next_sample))
                    dropped_genes = set(pred_genes) - set(gene_next_sample)
                    curr_sample, prompt_update = \
                        process_valid_output(gene_next_sample, curr_sample,
                                             gene_sampled, dropped_genes, args)

                    if prompt_update is None:
                        all_sampled_so_far = gene_sampled + curr_sample[:args.num_genes]
                        np.save(log_dir + '/sampled_genes_' + str(curr_step + 1) +
                            '.npy', all_sampled_so_far)
                        break
                    
                    if itr >= args.prompt_tries - 4:
                        
                        if genes_remain_summary is None:
                            # Start choosing from gene list instead of random sample
                            num_genes_pick = args.num_genes - len(curr_sample)
                            genes_remain = list(set(measured_genes).difference(set(gene_sampled)))
                            genes_remain_summary = summarize_remaining_genes(genes_remain)
                        else:
                            genes_remain_summary = list(set(
                                genes_remain_summary).difference(set(curr_sample)))
                        prompt_update = gene_choices_prompt(prompt_update,
                                            num_genes_pick, genes_remain_summary)

                if itr == args.prompt_tries-1:
                    np.save(log_dir + '/sampled_genes_' + str(curr_step + 1) +
                            '.npy', gene_sampled)
                else:
                    prompt_try = prompt_c + prompt_update

            

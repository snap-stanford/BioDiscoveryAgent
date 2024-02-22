import json
import os
import re
import subprocess
from LLM import complete_text, complete_text_claude
import selectors
import datetime
import shutil
import glob
import copy
import anthropic
import difflib
import numpy as np
import pandas as pd
from tqdm import tqdm
DEVICE = -1
PYTHON = "python"
USE_GPT4_EDIT_FILE = False
#from get_lit_review import get_lit_review

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


def research_log(action_input, invalid_action_error, folder_name = ".", **kwargs):
    try:
        content, = parse_action_input(action_input, ["content"])
    except:
        return invalid_action_error


    # if action == "read":
    #     return open(os.path.join(folder_name,"research_log.log")).read()
    # elif action == "write":
    #     if content is None:
    #         return "Research Log write action requires a second content argument"
    with open(os.path.join(folder_name,"research_log.log"), "a") as f:
        f.write(content+"\n")
    return open(os.path.join(folder_name,"research_log.log")).read()
    # else:
    #     return "Invalid operation for Research Log. Please use one of \"read\", \"write\""

def list_files(action_input, invalid_action_error, folder_name = ".", **kwargs):
    try:
        path, = parse_action_input(action_input, ["dir_path"])
    except:
        return invalid_action_error
    return subprocess.check_output(["ls", "-F", os.path.join(folder_name,path)]).decode("utf-8")
    # elif operation == "cd":
    #     return subprocess.check_output(["cd", os.path.join(folder_name,path)]).decode("utf-8")
    # elif operation == "pwd":
    #     return subprocess.check_output(["pwd", os.path.join(folder_name,path)]).decode("utf-8")
    # else:
    #     return "Invalid operation" 
def copy_file(action_input, invalid_action_error, folder_name = ".", **kwargs):
    try:
        source, destination = parse_action_input(action_input, ["source", "destination"])
    except:
        return invalid_action_error
    shutil.copyfile(os.path.join(folder_name,source), os.path.join(folder_name,destination))
    return f"File {source} copied to {destination}"

def understand_file(action_input, invalid_action_error, folder_name = ".", **kwargs):
    try:
        file_name, things_to_look_for = parse_action_input(action_input, ["file_name","things_to_look_for"])
    except:
        return invalid_action_error
    try:
        lines = open(os.path.join(folder_name,file_name)).readlines()
    except:
        return f"Error: cannot find script {file_name}"
    # zip lines with line id 
    # lines = [f"{i+1}: {l}" for i, l in enumerate(lines)]
    # group by 200 lines
    blocks = ["".join(lines[i:i+200]) for i in range(0, len(lines), 200)]

    descriptions  = []
    for idx, b in enumerate(blocks):
        start_line_number = 200*idx+1
        end_line_number = 200*idx+1 + len(b.split("\n"))
        prompt = f"""Given this (partial) file from line {start_line_number} to line {end_line_number}: 

``` 
{b}
```

Here is a detailed description on what to look for and what should returned: {things_to_look_for}

The description should short and also reference crtical lines in the script relevant to what is being looked for. Only describe what is objectively confirmed by the file content. Do not include guessed numbers. If you cannot find the answer to certain parts of the request, you should say "In this segment, I cannot find ...".
"""

        completion = complete_text_claude(prompt, log_file=kwargs["log_file"]+f"_{idx}")
        descriptions.append(completion)
    if len(descriptions) == 1:
        return descriptions[0]
    else:
        descriptions = "\n\n".join(["Segment {idx}: \n\n" + s for s in descriptions])
        prompt = f"""Given the relevant observations for each segments of a file, summarize to get a cohesive description of the entire file on what to look for and what should returned: {things_to_look_for}

{descriptions}
"""

        completion = complete_text_claude(prompt, log_file=kwargs["log_file"])

        return completion


def inspect_script_lines(action_input, invalid_action_error, folder_name = ".", **kwargs):
    try:
        script_name, start_line_number, end_line_number = parse_action_input(action_input, ["script_name", "start_line_number", "end_line_number"])
    except:
        return invalid_action_error
    try:
        start_line_number = int(start_line_number)
        end_line_number = int(end_line_number)
    except:
        return "Error: start_line_number and end_line_number must be integers"
    if end_line_number - start_line_number > 100:
        return "Error: the number of lines to display is limited to 100 lines"
    try:
        lines = open(os.path.join(folder_name,script_name)).readlines()
    except:
        return f"Error: cannot find script {script_name}"
    # zip lines with line id 
    # lines = [f"{i+1}: {l}" for i, l in enumerate(lines)]
    content = "".join(lines[max(int(start_line_number)-1, 0):int(end_line_number)])
    return content


def edit_script_direct(action_input, invalid_action_error, folder_name = ".", **kwargs):
    try:
        script_name, start_line_number, end_line_number, edited_content = parse_action_input(action_input, ["script_name", "replace_start_line_number", "replace_end_line_number", "edited_content"])
    except:
        return invalid_action_error
    try: 
        content = open(os.path.join(folder_name,script_name)).read() 
    except:
        return f"Error: the file {script_name} does not exist"
    try:
        start_line_number = int(start_line_number)
        end_line_number = int(end_line_number)
    except:
        return "Error: start_line_number and end_line_number must be integers"
    lines = content.splitlines()
    edited_lines = edited_content.splitlines()
    # edited_lines = [ line.split(":", 1)[1] for line in   edited_content.splitlines()]
    new_lines = lines[:int(start_line_number)-1] + edited_lines + lines[int(end_line_number):]
    new_content = "\n".join(new_lines)


    # backup all old file with prefix script_name
    backup_name = os.path.join(folder_name,"backup", f"{script_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    shutil.copyfile(os.path.join(folder_name,script_name), backup_name)

    
    with open(os.path.join(folder_name,script_name), "w") as f:
        f.write(new_content)

    # new_content = "\n".join([f"{i+1}: {l}" for i, l in enumerate(new_lines)])
    return "Here is the new script, please check if the edit is correct and desirable:\n\n" + new_content     

def edit_script(action_input, invalid_action_error, folder_name = ".", **kwargs):
    #TODO: handle long file editing
    try:
        script_name, instruction, save_name = parse_action_input(action_input, ["script_name", "edit_instruction", "save_name"])
    except:
        return invalid_action_error
    try: 
        content = open(os.path.join(folder_name,script_name)).read() 
    except:
        return f"Error: the file {script_name} does not exist"

    # lined_content = "\n".join([f"{i+1}: {l}" for i, l in enumerate(content.splitlines())])

    prompt = f"""Given this python script:

```python 
{content}
```

Edit the script by following the instruction:
{instruction}

Provide the full code after the edit, making no other changes. Start the python code with "```python". 
    
"""

    # Do not edit the part marked with DO NOT EDIT, and also do not introduce new DO NOT EDIT marks. If you need to edit that part to fulfill the edit isntruction, return an error message that explains this, starting with "Error:"
    if USE_GPT4_EDIT_FILE:
        completion =complete_text_openai(prompt, log_file=kwargs["log_file"])
    else:
        completion = complete_text_claude(prompt, log_file=kwargs["log_file"])

    # detect error message
    # if "Error:" in completion:
    #     return completion.split("Error:")[1].strip()

    # parse out the new content between ```python and ```
    new_content = completion.split("```python")[1].split("```")[0].strip()
    # new_content = "\n".join([l.split(": ", 1)[1] for l in new_content.splitlines()])

    # backup all old file with prefix script_name
    backup_name = os.path.join(folder_name,"backup", f"{script_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    shutil.copyfile(os.path.join(folder_name,script_name), backup_name)


    with open(os.path.join(folder_name,save_name), "w") as f:
        f.write(new_content)

    # new_lines = new_content.splitlines()
    # new_content = "\n".join([f"{i+1}: {l}" for i, l in enumerate(new_lines)])
    diff = list(difflib.unified_diff(content.splitlines(keepends=True), new_content.splitlines(keepends=True)))
    diff = "".join(diff)

    return f"The edited file is saved to {save_name}. Here is the diff, please check if the edit is correct and desirable:\n\n" + diff



def undo_edit_script(action_input, invalid_action_error, folder_name = ".", **kwargs):
    try:
        script_name, = parse_action_input(action_input, ["script_name"])
    except:
        return invalid_action_error
    backup_files = glob.glob(os.path.join(folder_name,"backup", f"{script_name}_*"))
    if len(backup_files) == 0:
        return f"Error: cannot undo edit for {script_name}"
    backup_files.sort()
    backup_file = backup_files[-1]
    shutil.copyfile(backup_file, os.path.join(folder_name,script_name))
    # delete the backup file
    os.remove(backup_file)

    new_content = open(os.path.join(folder_name,script_name)).read()
    # new_lines = new_content.splitlines()
    # new_content = "\n".join([f"{i}: {l}" for i, l in enumerate(new_lines)])
    return f"Content of {script_name} after undo the most recent edit:\n" + new_content

def run_experiment():

    pass

    return

def gene_search(action_input, folder_name = ".", **kwargs):
    try:
        gene_name = parse_action_input(action_input, ["gene_name"])[0].strip()
    except:
        return "Gene search failed either due to parsing, try again and follow the schema given to you on how to use this tool."
    import pandas as pd
    import numpy as np
    return gene_search_f(gene_name, folder_name, **kwargs)
    
def gene_search_f(gene_name, folder_name = ".", **kwargs):
    print(gene_name)
    df = pd.read_csv("/dfs/user/yhr/AI_RA/research_assistant/datasets/features/achilles.csv")
    df = df.rename(lambda x : x.split(" (")[0], axis='columns')
    df.drop(columns=["DepMap_ID"], inplace=True)
    df.dropna(inplace=True, axis='rows')
    if gene_name not in df.columns:
        return f"Gene {gene_name} not found"
    return ", ".join((df[gene_name].dot(df) / (np.linalg.norm(df, axis=0) * np.linalg.norm(df[gene_name]))).sort_values(ascending=False)[:11].index.tolist()[1:])
    
    
def arxiv_search(action_input, max_papers = 5, folder_name = ".", **kwargs):
    
    try:
        query = parse_action_input(action_input, ["script_name"])[0].strip()
    except:
        return "Arxiv search failed either due to parsing, try again and follow the schema given to you on how to use this tool."

    import arxiv
    search = arxiv.Search(
        query = query,
        id_list = [],
        max_results = max_papers,
        SortCriterion = SortCriterion.Relevance,
        SortOrder = SortOrder.Descending
    )
    
    observation = ""
    for result in arxiv.Client().results(search):
        observation += "\n" + result.title + "\n\n" + result.summary + "\n"

    return observation
    

def execute_script(action_input, invalid_action_error, folder_name = ".", **kwargs):
    # TODO: handle long output
    try:
        script_name = parse_action_input(action_input, ["script_name"])[0].strip()
    except:
        return invalid_action_error

    script_path = os.path.join(".",script_name)
    cmd = f"CUDA_VISIBLE_DEVICES={DEVICE} {PYTHON} -u {script_path}"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=folder_name)

    stdout_lines = []
    stderr_lines = []

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)

    while process.poll() is None and selector.get_map():
        events = selector.select(timeout=1)

        for key, _ in events:
            line = key.fileobj.readline()
            if key.fileobj == process.stdout:
                print("STDOUT:", line, end =" ")
                stdout_lines.append(line)
            else:
                print("STDERR:", line, end =" ")
                stderr_lines.append(line)

    for line in process.stdout:
        line = line
        print("STDOUT:", line, end =" ")
        stdout_lines.append(line)
    for line in process.stderr:
        line = line
        print("STDERR:", line, end =" ")
        stderr_lines.append(line)

    return_code = process.returncode

    if return_code != 0:
        return "".join(stderr_lines)
    else:
        return "".join(stdout_lines)


def request_help(action_input, invalid_action_error, folder_name = ".", **kwargs):
    try:
        request, = parse_action_input(action_input, ["request"])
    except:
        return invalid_action_error

    return input(f"AI is requesting help: {request}\n")

def reflection(action_input, invalid_action_error, folder_name = ".", research_problem = "", **kwargs):
    try:
        things_to_reflect_on, = parse_action_input(action_input, ["things_to_reflect_on"])
    except:
        return invalid_action_error

    research_log_content = open(os.path.join(folder_name, "research_log.log")).read()

    prompt = f"""We are trying to solve this research problem: {research_problem}

Your current research log:
```
{research_log_content}
```

Reflect on this: {things_to_reflect_on} 

Give an answer in natural language paragraphs as truthfully as possible. 

"""
    reflection = complete_text_claude(prompt, log_file=kwargs["log_file"])
    return f"Reflection: {reflection}\n"


def summarize_action_and_observation(action, observation, **kwargs):

    prompt = f"""Given your action and the observation: 
{action} 
[Observation]:
```
{observation}
```

Summarize your action and the observation in this format:
[Reasoning]: Summarize the reasoning behind the action
[Action]: Summarize all relevant details of the action objectively
[Observation]: Summarize all relevant details in the observation objectively

Do not include additional information or suggestions.
"""

    summary = "[Reasoning]:" + complete_text_claude(prompt, log_file=kwargs["log_file"]).split("[Reasoning]:")[1]
    return summary



def retrieval_from_research_log(folder_name, research_problem, current_plan, **kwargs):
    # TODO: sliding window/ vector retrieval
    research_log_content = open(os.path.join(folder_name, "research_log.log")).read()
    prompt = f"""We are trying to solve this research problem: {research_problem}

Your current Research Plan and Status

{current_plan}
    
Your current research log:
```
{research_log_content}
```

Concisely summarize and list all relevant information from the research log that will be helpful for future step in this format:
"""

    retrieval = complete_text_claude(prompt, log_file=kwargs["log_file"])

    return retrieval


def update_plan_and_status(summary_of_last_step, thought_and_action, folder_name, research_problem, current_plan, **kwargs):
    
    research_log_content = open(os.path.join(folder_name, "research_log.log")).read()
    prompt = f"""We are trying to solve this research problem: {research_problem}

Current Plan and Status:
```
{current_plan}
```
    
Given a summary of what you did in the last step: 
{summary_of_last_step}

And what you are doing in this step:
{thought_and_action}

Update the plan and status. The plan and status should be concise but informative. Do not include any result that is guessed rather than directly confirmed by the observation.

"""

    plan = complete_text_claude(prompt, log_file=kwargs["log_file"]).split("```\n")[1].split("\n```")[0]

    return plan

def work_on_subtask(action_input, invalid_action_error, folder_name = ".", **kwargs):
    try:
        subtask, = parse_action_input(action_input, ["subtask"])
    except:
        return invalid_action_error

    current_history = copy.deepcopy(kwargs["current_history"])
    current_history["actions"] = []
    current_history["observations"] = []
    current_history["tool_names"] = current_history["low_level_tools"]
    current_history["research_problem"] += f"\n\nCurrently, we decided to work on subtask: {subtask}\n"
    current_history["instructions"] = low_evel_instructions
    
    return agent_loop(current_history, steps = 20, use_gpt4 = False, log_file = kwargs["log_file"], args = kwargs["args"])


def construct_tools_prompt(tool_names):
    tools_prompt = ""
    tools = {}
    for tool_name in tool_names:
        tool = ALL_TOOLS[tool_name]
        tools[tool_name] = tool
        tools_prompt += f"""- {tool_name}:
    {tool["description"]}
    Usage:
    ```
    {tool["usage"]}
    Observation: [{tool["return"]}]
    ```
        """.strip() + "\n\n"
    return tools_prompt, tools

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

def split_outside_parentheses(s, delimiter=','):
    parts = []
    current = []
    level = 0
    # Go through each character in the string
    for char in s:
        if char == '(':
            level += 1
        elif char == ')':
            level -= 1
        # If we find the delimiter and we are not inside parentheses
        if char == delimiter and level == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    # Add the last part
    parts.append(''.join(current).strip())
    return parts

def agent_loop(current_history, steps, use_gpt4, log_dir, args):
    valid_format_entires = ["Solution"]
    # valid_format_entires = ["[Reflection]", "[Research Plan and Status]","[Thought]", "[Action]","[Action Input]"]
    
    current_plan = "Empty"
    #summary_of_last_step = "Empty"
    #relevant_history = ""
    use_gpt4 = False


    folder_name = current_history["folder_name"] 
    tool_names = current_history["tool_names"]
    research_problem = current_history["research_problem"]
    # research_problem = "Your task is to identify predict genes which important for some task. You will get some observations telling which the scores of the genes chosen by you and some of them will be hits. Your tasks is maximize hits over various rounds by taking into account the observations and prior knowledge. "
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
        f.write("Enabled Tools:" + str(tool_names) + "\n") 
        tools_prompt, tools = construct_tools_prompt(tool_names)
        f.write("================================Start=============================\n")
        last_steps = 3
        # research_log_content = open(os.path.join(folder_name, "research_log.log")).read()
        f.write(current_history["initial_prompt"].format(tools_prompt=tools_prompt, tool_names=tool_names,  research_problem=research_problem) + "\n")
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

                # construct prompt for the current step
                #research_log_content = open(os.path.join(folder_name,
                # "research_log.log")).read()

            prompt = 'Step {}\n '.format(curr_step)
            prompt += current_history["initial_prompt"].format(
                                            tools_prompt=tools_prompt,
                                            tool_names=tool_names,
                                            research_problem=research_problem)
            if args.lit_review:
                lit_review_prompt = current_history["initial_prompt"].format(
                                            tools_prompt=tools_prompt,
                                            tool_names=tool_names,
                                            research_problem=research_problem).split("Always respond")[0]


            if curr_step > last_steps:
                # prompt += "\nWe have already made some progress in this. Let's continue! \n\n"

                # f.write("Research Log Update:\n" + summary)



                # prompting for retrieval
                log_file = os.path.join(log_dir , f"step_{curr_step}_log_retrieval.log")
                relevant_history = retrieval_from_research_log( folder_name, research_problem, current_plan, log_file=log_file)

                prompt += ''
                #prompt += f"""
                #Here is a summary of relevant actions and observations you have done:
                    #```
                #{relevant_history}
                #```

                #Here are the exact several steps you have done most recently (up to 3 steps):

                # """
            else:
                prompt += "\nNow let's start!\n\n"
            # if curr_step > last_steps:
            #     prompt += "Step " + str(curr_step-last_steps) + ":\n"
            # else:
            #     prompt += "Step 0:\n"

            if curr_step == 0:
                pass
                #prompt += current_history["instructions"]
                #prompt += "\n To get you started, here is a list of " \
                #          "tested genes and their measured log fold change
                #          in INF-γ: \n" +\
                #          gene_readout.to_string()

                #prompt += "\n Out of these, we call {} genes hits,
                # since they " \
                #        "showed high log fold change values".format(len(hits))\
                #          + ground_truth.loc[hits].to_string()

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
                        completion_pre = completion_pre + "Gene Search Result:" + gene_search_f(completion_pre.split("Gene Search:")[1].strip())
                        
                        
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

            """
            # research_log_content = open(os.path.join(folder_name, "research_log.log")).read()
            new_research_log_content = rg.strip("```") + "\n\n"
            # if "<previous research log>" in new_research_log_content:
            #     new_research_log_content = new_research_log_content.replace("<previous research log>", research_log_content)
            entries["Research Plan"] = new_research_log_content

            current_history["actions"].append(entries)
            f.write(anthropic.AI_PROMPT + "\n" + print_action(current_history["actions"][-1]) + "\nObservation:\n")

            with open(os.path.join(log_dir, "display_step"), "w", 1) as f_t:
                f_t.write("Step " + str(curr_step) + ":\n")
                f_t.write(anthropic.AI_PROMPT + "\n" + print_action(current_history["actions"][-1]) + "\nObservation:\n")

            entries["Research Plan"] = new_research_log_content.replace("**", "")
            # with open(os.path.join(folder_name, "research_log.log"), "w") as fr:
            #     fr.write(new_research_log_content)


            # parse the completion
            #observation = ''
            #current_history["observations"].append(observation)

            #f.write("\n```\n" + current_history["observations"][-1] +
            # "\n```\n\n")
            #with open(os.path.join(log_dir, "display_step"), "a",
            # 1) as f_t:
            #    f_t.write("\n```\n" + current_history["observations"][
            #    -1] + "\n```\n\n")

            ## prompting to summarize previous action and observation
            #log_file = os.path.join(log_dir , f"step
            # _{curr_step}_log_summary.log")
            #summary_of_last_step = summarize_action_and_observation(
            # current_history["actions"][-1], current_history["observations"][-1], log_file=log_file)
            # update research log
            #with open(os.path.join(folder_name, "research_log.log"),
            # "a") as f_r:
            #    f_r.write("\n\nStep " + str(curr_step) + ":\n" +
            #    summary_of_last_step + "\n")

            #if args.interactive_interval is not None and (curr_step + 1) %
                # args.interactive_interval == 0:
            #    human_input = input("Please enter your advice: ")
            #    current_history["observations"][-1] += "\n" +
                #    anthropic.HUMAN_PROMPT + "\n" + human_input + "\n"
            #    f.write("\n" + anthropic.HUMAN_PROMPT + "\n" + human_input + "\n")
            """





ALL_TOOLS = {

        "Research Log": {
            "description": "Append to the research log to keeps track of 1) the high level plan 2) the research progress so far 3) important understandings based on recent observations 4) new plans and so on. Its current content will always be displayed at the beginning.",
            "usage": """ 
    Action: Research Log
    Action Input: {
        "content": [a string within 500 character limit]
    }""".strip(),
            "return": "The observation will be the updated content of research log",
            "function": research_log
        },
        "List Files": {
            "description": "Use this to navigate the file system.",
            "usage": """
    Action: List Files
    Action Input: {
        "dir_path": [a valid relative path to a directory]
    }""".strip(),
            "return": "The observation will be a list of files and folders in dir_path or current directory is dir_path is empty, or an error message if dir_path is invalid.",
            "function": list_files
        },
        "Read File": {
            "description": "Use this to read the whole file and understand certain aspects. You should provide detailed description on what to look for and what should be returned. To get a better understanding of the file, you can use Inspect Script Lines action to inspect specific part of the file.",
            "usage": """
    Action: Read File
    Action Input: {
        "file_name": [a valid file name with relative path to current directory if needed],
        "things_to_look_for": [a detailed description on what to look for and what should returned]
    }""".strip(),
            "return": "The observation will be a description of relevant content and lines in the file. If the file does not exist, the observation will be an error message.",
            "function": understand_file
        },
        "Inspect Script Lines": {
            "description": "Use this to inspect specific part of a python script precisely, or the full content of a short script. The number of lines to display is limited to 100 lines. This is especially helpful when debugging.",
            "usage": """
    Action: Inspect Script Lines
    Action Input: {
        "script_name": [a valid python script name with relative path to current directory if needed],
        "start_line_number": [a valid line number],
        "end_line_number": [a valid line number]
    }""".strip(),
            "return": "The observation will be the content of the script between start_line_number and end_line_number . If the script does not exist, the observation will be an error message.",
            "function": inspect_script_lines
        },
        "Edit Script (AI)": {
            "description": "Use this to do a relatively large but cohesive edit over a python script. Instead of editing the script directly, you should describe the edit instruction so that another AI can help you do this.",
            "usage": """
    Action: Edit Script (AI)
    Action Input: {
        "script_name": [a valid python script name with relative path to current directory if needed],
        "edit_instruction": [a detailed step by step description on how to edit it.],
        "save_name": [a valid file name with relative path to current directory if needed]
    }""".strip(),
            "return": "The observation will be the edited content of the script. If the script does not exist, the observation will be an error message. You should always double check whether the edit is correct. If it is far from correct, you can use the Undo Edit Script action to undo the edit.",
            "function": edit_script
        },
        "Edit Script (Direct)": {
            "description": "Use this to edit the python script precisely for up to 100 lines. Only use this when debugging and be careful with indentation.",
            "usage": """
    Action: Edit Script (Direct)
    Action Input: {
        "script_name": [a valid python script name with relative path to current directory if needed],
        "replace_start_line_number": [a valid line number],
        "replace_end_line_number": [a valid line number],
        "edited_content": [valid python code to replace the content between start_line_number and end_line_number with proper indentations in 4 spaces e.g. "\n#Define the neural network model\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()"]\n    }"]""".strip(),
            "return": "The observation will be the edited content of the script. If the script does not exist, the observation will be an error message. You should always double check whether the edit is correct. If it is far from correct, you can use the Undo Edit Script action to undo the edit.",
            "function": edit_script_direct
        },
        "Undo Edit Script": {
            "description": "Use this to undo the last edit of the python script.",
            "usage": """
    Action: Undo Edit Script
    Action Input: {
        "script_name": [a valid python script name with relative path to current directory if needed]
    }""".strip(),
            "return": "The observation will be the content of the script before the last edit. If the script does not exist, the observation will be an error message.",
            "function": undo_edit_script
        },
        "Execute Script": {
            "description": "Use this to execute the python script.",
            "usage": """
    Action: Execute Script
    Action Input: {
        "script_name": [a valid python script name with relative path to current directory if needed]
    }""".strip(),
            "return": "The observation will be output of the script or errors.",
            "function": execute_script
        },
    "Request Help": {
            "description": "Use this to request help from human. Use this only when the provided tools and files are not enough for accomplishing necessary steps, such as requesting API reference or installing a library. So you should check through the provided tools and files first.",
            "usage": """
    Action: Request Help
    Action Input: {
        "request": [a detailed description on what to do]
    }""".strip(),
            "return": "The observation will be the response from human.",
            "function": request_help
        },
    "Reflection": {
            "description": "Use this to look over all the past steps and reflect. You should provide detailed description on what to reflect on and what should be returned.",
            "usage": """
    Action: Reflection
    Action Input: {
        "things_to_reflect_on": [a detailed description on what to reflect on and what should be returned]
    }""".strip(),
            "return": "The observation will be a the reflection.",
            "function": reflection
    },
    "Final Answer": {
            "description": "Use this to provide the final answer to the current task.",
            "usage": """
    Action: Final Answer
    Action Input: {
        "final_answer": [a detailed description on the final answer]
    }""".strip(),
            "return": "The observation will be empty.",
            "function": None # should link to evaluation
    },
    "Work On Subtask": {
            "description": "Use this to work on a subtask of the current task, such as debugging. You should provide detailed description on what to work on and what should be returned.",
            "usage": """
    Action: Work On Subtask
    Action Input: {
        "subtask": [a detailed description on what to work on and what should be returned]
    }""".strip(),
            "return": "The observation will be a the result of working on the subtask.",
            "function": work_on_subtask
    },
    "Copy File": {
            "description": "Use this to copy a file to a new location with a new name.",
            "usage": """
    Action: Copy File
    Action Input: {
        "source": [a valid file name with relative path to current directory if needed],
        "destination": [a valid file name with relative path to current directory if needed]
    }""".strip(),
            "return": "The observation will be empty.",
            "function": copy_file
    },
    "Arxiv Search": {
            "description": "Use this to query arxiv and get a summary of the top few paper abstracts as retrieved by your query .",
            "usage": """
    Action: Arxiv Search
    Action Input: {
        "query": [a query string to be search on arxiv for],
    }""".strip(),
            "return": "The observation will be empty.",
            "function": arxiv_search
    },
    "Gene Search": {
            "description": "Use this to search for 100 most similar genes based on feature.",
            "usage": """
    Action: Gene Search
    Action Input: {
        "gene_name": [a gene name],
    }""".strip(),
            "return": "The 10 most similar genes sorted in similarity.",
            "function": gene_search
    },

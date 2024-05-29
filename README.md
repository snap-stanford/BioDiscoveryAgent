# BioDiscoveryAgent

research_assistant.py is the main file. 

## Installation

Install required packages using the following command:
```
pip install -r requirements.txt
```
Claude API key is required for running the code. Please visit the [Anthropic website](https://docs.anthropic.com/en/docs/getting-access-to-claude) 
for more information

## Tasks

1. IFNG
2. IL2
3. Carnevale
4. Scharenberg
   
## Commands

For using all tools on IFNG dataset:

```
python research_assistant.py  --task perturb-genes-brief --model claude-1 --run_name test --data_name IFNG --steps 5 --num_genes 128 --log_dir v1
```
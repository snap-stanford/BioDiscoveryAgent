# BioDiscoveryAgent

BioDiscoveryAgent is an AI agent for closed loop design of 
biological experiments. BioDiscoveryAgent designs genetic perturbation experiments 
using only an LLM (Claude v1) paired with a suite of tools (literature 
search, gene search, AI critique).

![BioDiscoveryAgent](assets/icon.pdf)

## Installation

Install required packages using the following command:
```
pip install -r requirements.txt
```
Claude API key is required for running the code. Please visit the [Anthropic website](https://docs.anthropic.com/en/docs/getting-access-to-claude) 
for more information

## Datasets

1. IFNG
2. IL2
3. Carnevale
4. Scharenberg
   
## Commands

To run BioDiscoveryAgent with all tools on the IFNG dataset:

```
python research_assistant.py  --task perturb-genes-brief --model claude-1 --run_name test --data_name IFNG --steps 5 --num_genes 128 --log_dir v1
```

## Preprocessing your own dataset

To preprocess your own dataset, please follow the instructions in the [preprocessing notebook](notebooks/Preprocessing.ipynb)
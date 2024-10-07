from pydantic import BaseModel, Field
from langchain.tools import BaseTool, tool

from typing import Type, List, Dict
from collections import Counter
import requests
import pandas as pd
import numpy as np
import json
import re

# from bioagentos.utils import get_gene_id, execute_graphql_query, ID

class GeneInput(BaseModel):
    gene_name: str = Field(description="gene name, e.g. BRCA1")

class GeneTopKInput(BaseModel):
    gene_name: str = Field(description="gene name, e.g. BRCA1")
    K: int = Field(description="number of top results to return")

class GeneListTopKInput(BaseModel):
    genes: List[str] = Field(description="list of gene names")
    K: int = Field(description="number of top results to return")

@tool("get_gene_info_from_opentarget", args_schema=GeneInput, return_direct=True)
def get_gene_info_from_opentarget(gene_name: str) -> str:
    """Given a gene name, returns detailed gene information including symbol, 
    biotype, description, chromosome, TSS, start, end, forward strand, exons. 
    Source: open target genetics."""
    gene_id = get_gene_id(gene_name, ID.ENSEMBL)
    if gene_id is None:
        return "Error: Not a valid gene."
    
    query = """
    query getGeneInfo($geneId: String!) {
        geneInfo(geneId: $geneId) {
            id
            symbol
            bioType
            description
            chromosome
            tss
            start
            end
            fwdStrand
            exons
        }
    }
    """
    variables = {"geneId": gene_id}
    
    try:
        data = execute_graphql_query(query, variables)
        gene = data['data']['geneInfo']
        output = f"""
        Gene ID: {gene['id']}
        Symbol: {gene['symbol']}
        Biotype: {gene['bioType']}
        Description: {gene['description']}
        Chromosome: {gene['chromosome']}
        TSS: {gene['tss']}
        Start: {gene['start']}
        End: {gene['end']}
        Forward Strand: {gene['fwdStrand']}
        Exons: {', '.join(map(str, gene['exons']))}
        """
        return output.strip()
    except Exception as e:
        return {"error": str(e)}

@tool("get_gene_to_reactome_pathways", args_schema=GeneInput, return_direct=True)
def get_gene_to_reactome_pathways(gene_name: str) -> List[str]:
    """
    Use this to get the list of reactome pathways this gene relates to.
    Input: one valid gene name
    """
    base_url = "https://mygene.info/v3"
    query_url = f"{base_url}/query"
    params = {
        "q": gene_name,
        "scopes": "symbol",
        "fields": "name,symbol,entrezgene,ensembl.gene,pathway",
        "species": "human"
    }
    response = requests.get(query_url, params=params)
    result = response.json()
    
    if not result['hits']:
        return [f"No annotations found for {gene_name}."]
    
    annotations = result['hits'][0]
    if 'pathway' in annotations and 'reactome' in annotations['pathway']:
        reactome_pathways = [f"{i['id']}: {i['name']}" for i in annotations['pathway']['reactome']]
        return reactome_pathways
    
    return ["No reactome pathways found for this gene."]

@tool("get_gene_pLI", args_schema=GeneInput, return_direct=True)
def get_gene_pLI(gene_name: str) -> str:
    """
    Use this to get the query gene's pLI (probability of being Loss-of-function Intolerant).
    Input: one valid gene name
    """
    constraint = pd.read_csv('/dfs/user/kexinh/agent_data/forweb_cleaned_exac_r03_march16_z_data_pLI_CNV-final.txt', sep='\s+')
    gene2pli = dict(constraint[['gene', 'pLI']].values)
    
    gene_symbol = gene_name.strip().strip("```")
    try:
        pli_score = gene2pli[gene_symbol]
        if pli_score > 0.9:
            return f"pLI score is {pli_score} and this gene is associated with severe phenotypes in the haploinsufficient state"
        elif pli_score < 0.1:
            return f"pLI score is {pli_score} and this gene is not haploinsufficient"
        else:
            return f"pLI score is {pli_score}"
    except KeyError:
        return "pLI score is not found in this EXAC database"

# TODO: @yusuf
'''
@tool("get_similar_genes", args_schema=GeneInput, return_direct=True)
def get_similar_genes(gene_name, data_file, **kwargs):
    pass
'''

# @tool("get_rna_seq", args_schema=GeneTopKInput, return_direct=True)
def get_rna_seq(gene_name: str, K: int = 10) -> str:
    """
    Given a gene name, this tool will return the max K transcripts-per-million (TPM) per tissue from the RNA-seq expression.
    """
    try:
        import gget
        # Fetch RNA-seq data using gget
        data = gget.archs4(gene_name, which="tissue")
        
        if data.empty:
            return f"No RNA-seq data found for the gene {gene_name}."
        
        # Create a readable output string
        readable_output = f"RNA-seq expression data for {gene_name}:\n"
        for index, row in data.iterrows():
            if index < K:
                tissue = row['id']
                min_tpm = row['min']
                q1_tpm = row['q1']
                median_tpm = row['median']
                q3_tpm = row['q3']
                max_tpm = row['max']
                readable_output += (
                    f"\nTissue: {tissue}\n"
                    #f"  - Min TPM: {min_tpm}\n"
                    #f"  - Q1 TPM: {q1_tpm}\n"
                    f"  - Median TPM: {median_tpm}\n"
                    #f"  - Q3 TPM: {q3_tpm}\n"
                    #f"  - Max TPM: {max_tpm}\n"
                )
            else:
                break
        
        return readable_output
    
    except Exception as e:
        return f"An error occurred: {e}"

# @tool("get_top_k_correlated_genes", args_schema=GeneTopKInput, return_direct=True)
def get_top_k_correlated_genes(gene_name: str, K=10):
    """
    Fetches the top K most correlated genes for a given gene of interest using ARCHS4.
    """
    import gget
    species='human'
    ensembl=False
    save=False
    json_output=True
    verbose=True
    # Constructing the command options based on the input parameters
    options = {
        'species': species,
        'json': json_output,
        'verbose': verbose,
        'ensembl': ensembl,
        'save': save
    }
    
    # Fetching the top correlated genes
    correlation_table = gget.archs4(gene_name, **options)
    
    # Filtering the top K correlated genes
    top_k_genes = correlation_table[:K]
    # Formatting the output as a readable string
    output_str = f"Top {K} genes most correlated with {gene_name}:\n\n"
    for idx, row in enumerate(top_k_genes):
        output_str += str(idx+1) + '. Gene: ' + str(row['gene_symbol']) + \
        ', Pearson Correlation: ' + str(row['pearson_correlation'])[:5] + "\n"
    return output_str



def get_enrichment(genes, top_k=10, database="ontology"):
    """
    Perform enrichment analysis for a list of genes and format the results.

    Parameters:
    genes (list of str): List of gene symbols to analyze.
    top_k (int, optional): Number of top pathways to return. Default is 10.
    database (str, optional): Name of the database to use for enrichment analysis. Default is "ontology".

    Returns:
    str: Formatted string of the top K enrichment results.
    """
    import gget
    df = gget.enrichr(genes, database=database)
    df = df.head(top_k)
    output_str = ""
    paths = []
    for idx, row in df.iterrows():
        output_str += (
            f"Rank: {row['rank']}\n"
            f"Path Name: {row['path_name']}\n"
            f"P-value: {row['p_val']:.2e}\n"
            f"Z-score: {row['z_score']:.6f}\n"
            f"Combined Score: {row['combined_score']:.6f}\n"
            f"Overlapping Genes: {', '.join(row['overlapping_genes'])}\n"
            f"Adjusted P-value: {row['adj_p_val']:.2e}\n"
            f"Database: {row['database']}\n"
            "----------------------------------------\n"
        )
        paths.append(row['path_name'])
    return paths
    return output_str

def filter_valid_gene_names(gene_name):
    valid_genes = set()
    gene_pattern = re.compile(r'^[A-Z0-9]+$')
    potential_genes = re.split(r'[ ,:()-]+', gene_name)
    for gene in potential_genes:
        if gene_pattern.match(gene):
           valid_genes.add(gene) 
    return valid_genes

def get_genes_in_reactome_pathway(pathway_id):
    # Step 1: Get the Reactome pathway ID for the given pathway name
    pathway_url = f"https://reactome.org/ContentService/data/participants/{pathway_id}/participatingPhysicalEntities"
    pathway_response = requests.get(pathway_url)
        
    if pathway_response.status_code != 200 or not pathway_response.json():
        print(f"No detailed information found for the pathway ID '{pathway_id}'.")
        
        # Step 3: Parse the response to find genes
    participants = pathway_response.json()
    genes = set()
    for participant in participants:
        if 'name' in participant:
            for entry in participant["name"]:
                cands = filter_valid_gene_names(entry)
                genes.update(list(cands))
    if not genes:
        return f"No genes found for the pathway ID '{pathway_id}'."
    return genes

def get_genes_in_kegg_pathway(pathway_name):
    # First, get the KEGG pathway ID for the given pathway name
    search_url = f"http://rest.kegg.jp/find/pathway/{pathway_name}"
    response = requests.get(search_url)
    
    if response.status_code != 200 or not response.text:
        return f"No pathway found with the name '{pathway_name}'."
    
    # Parse the first pathway ID from the search results
    pathway_info = response.text.split('\n')[0]
    pathway_id = pathway_info.split()[0].split(":")[1]
    # print(f"pathway id: {pathway_id}")
    # Now, retrieve the pathway's gene list using the pathway ID
    pathway_id = pathway_id.replace("map", "hsa")
    pathway_url = f"http://rest.kegg.jp/get/{pathway_id}"
    pathway_response = requests.get(pathway_url)
    if pathway_response.status_code != 200 or not pathway_response.text:
        return f"No detailed information found for the pathway ID '{pathway_id}'."
    # Step 2: Parse the response to find genes
    genes = set()
    lines = pathway_response.text.split('\n')
    in_gene_section = False
    
    for line in lines:
        if line.startswith("GENE"):
            in_gene_section = True
        elif in_gene_section:
            if line.startswith(" "):  # Continuation of the gene section
                gene_info = line.strip().split()
                if len(gene_info) >= 2:
                    gene_symbol = gene_info[1].rstrip(';')
                    genes.add(gene_symbol)
            else:
                break  # Exit the gene section
    
    if not genes:
        return f"No genes found for the pathway ID '{pathway_id}'."
    
    return genes

def get_topk_genes_in_pathways(pathways, sampled, K=30):
    gene_counter = Counter()
    for pathway in pathways:
        genes = get_genes_in_kegg_pathway(pathway)
        if isinstance(genes, set):
            filtered = genes - set(sampled)
            gene_counter.update(filtered)
    new_genes = gene_counter.most_common(K)
    output = []
    for gene in new_genes:
        g = gene[0]
        output.append(g)
    return output

def get_topk_genes_in_reactome(pathways, sampled, K=30):
    gene_counter = Counter()
    for pathway in pathways:
        genes = get_genes_in_reactome_pathway(pathway)
        if isinstance(genes, set):
            filtered = genes - set(sampled)
            gene_counter.update(filtered)
    new_genes = gene_counter.most_common(K)
    output = []
    for gene in new_genes:
        g = gene[0].rstrip(';')
        output.append(g)
    return output

@tool("get_enrichment_biological_processes", args_schema=GeneListTopKInput, return_direct=True)
def get_enrichment_biological_processes(genes, K=10):
    """
    Perform enrichment analysis for biological processes and format the results.
    """
    return get_enrichment(genes, K, database="ontology")

# @tool("get_enrichment_KEGG_pathways", args_schema=GeneListTopKInput, return_direct=True)
def get_enrichment_KEGG_pathways(genes, K=10):
    """
    Perform enrichment analysis for KEGG pathways and format the results.
    """
    return get_enrichment(genes, K, database="pathway")

@tool("get_enrichment_transcription", args_schema=GeneListTopKInput, return_direct=True)
def get_enrichment_transcription(genes, K=10):
    """
    Perform enrichment analysis for transcription factors and format the results.
    """
    return get_enrichment(genes, K, database="transcription")

@tool("get_enrichment_diseases_drugs_GWAS_catalog", args_schema=GeneListTopKInput, return_direct=True)
def get_enrichment_diseases_drugs_GWAS_catalog(genes, K=10):
    """
    Perform enrichment analysis for diseases and drugs using GWAS catalog and format the results.
    """
    return get_enrichment(genes, K, database="diseases_drugs")

@tool("get_enrichment_celltypes", args_schema=GeneListTopKInput, return_direct=True)
def get_enrichment_celltypes(genes, K=10):
    """
    Perform enrichment analysis for cell types and format the results.
    """
    return get_enrichment(genes, K, database="celltypes")

@tool("get_enrichment_kinase_interactions", args_schema=GeneListTopKInput, return_direct=True)
def get_enrichment_kinase_interactions(genes, K=10):
    """
    Perform enrichment analysis for kinase interactions and format the results.
    """
    return get_enrichment(genes, K, database="kinase_interactions")

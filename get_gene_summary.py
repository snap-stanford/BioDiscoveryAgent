import requests
from bs4 import BeautifulSoup
import html2text
import mygene
mg = mygene.MyGeneInfo()
from LLM import complete_text_claude
import anthropic
import concurrent.futures

parts_to_remove = [
    "##  Summary\n",
    "NEW",
    'Try the newGene table',
    'Try the newTranscript table',
    '**',
    "\nGo to the top of the page Help\n"
]

def rough_text_from_gene_name(gene_number):
    
    # get url
    url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_number}"
    # Send a GET request to the URL
    summary_text = ''
    soup = None
    try:
        response = requests.get(url, timeout=30)
    except requests.exceptions.Timeout:
        print('time out')
        return((summary_text,soup))
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the "summary" tab content by inspecting the page's structure
        summary_tab = soup.find('div', {'class': 'rprt-section gene-summary'})

        # Check if the "summary" tab content is found
        if summary_tab:
            # Convert the HTML to plain text using html2text
            html_to_text = html2text.HTML2Text()
            html_to_text.ignore_links = True  # Ignore hyperlinks

            # Extract the plain text from the "summary" tab
            summary_text = html_to_text.handle(str(summary_tab))
            # Remove the specified parts from the original text
            for part in parts_to_remove:
                summary_text = summary_text.replace(part, ' ')
                # Replace '\n' with a space
            summary_text = summary_text.replace('\n', ' ')

            # Reduce multiple spaces into one space
            summary_text = ' '.join(summary_text.split())
            # Print or save the extracted text
        else:
            print("Summary tab not found on the page.")
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    return((summary_text,soup))        

gene_name_to_summary_page = {}

import concurrent.futures

def summarize_gene_table(data, research_problem):
    data['Summary'] = ""

    # gene_name_to_summary_page = {}  # Cache for gene descriptions

    def summarize_gene(gene_name):
        # nonlocal gene_name_to_summary_page

        if gene_name not in gene_name_to_summary_page:
            cd_24_name = mg.querymany(gene_name, scopes='symbol', species='human')
            gene_name_to_tax_id = {}
            for result in cd_24_name:
                if "_id" in result and "query" in result:
                    gene_name_to_tax_id[result['symbol']] = result['_id']

            for gene_name, page_id in sorted(gene_name_to_tax_id.items()):
                if gene_name not in gene_name_to_summary_page:
                    parsed_text, unparsed_html = rough_text_from_gene_name(page_id)
                    gene_name_to_summary_page[gene_name] = parsed_text

        parsed_text = gene_name_to_summary_page[gene_name]
        summarize_prompt = '''You are given the following research problem: {} \n\n
For the gene {}, provided is the ncbi description of the gene properties: {} \n\n
Your task is to summarize the gene description in strictly less than 1-2 lines which you think is useful to solve the provided research problem.
        '''.format(research_problem, gene_name, parsed_text)

        return gene_name, complete_text_claude(summarize_prompt, stop_sequences=[anthropic.HUMAN_PROMPT], log_file=None)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(summarize_gene, gene_name) for gene_name in data.index]

    for future in concurrent.futures.as_completed(futures):
        gene_name, summary = future.result()
        data.at[gene_name, 'Summary'] = summary

    return data

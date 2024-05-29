from LLM import complete_text, complete_text_claude
import arxiv
import scholarly
from pymed import PubMed
from urllib import request
from bs4 import BeautifulSoup

DEFAULT_URL = {
    'biorxiv':
    'https://www.biorxiv.org/search/{}%20numresults%3A25%20sort%3Arelevance-rank'
}

class MyBiorxivRetriever():
    def __init__(self, search_engine='biorxiv', search_url=None):
        self.search_engine = search_engine
        self.search_url = search_url or DEFAULT_URL[search_engine]
        return

    def _get_article_content(self,
                             page_soup,
                             exclude=[
                                 'abstract', 'ack', 'fn-group', 'ref-list'
                             ]):
        article = page_soup.find("div", {'class': 'article'})
        article_txt = ""
        if article is not None:
            for section in article.children:
                if section.has_attr('class') and any(
                        [ex in section.get('class') for ex in exclude]):
                    continue
                article_txt += section.get_text(' ')

        return article_txt

    def _get_all_links(self, page_soup, max_number, base_url="https://www.biorxiv.org"):
        links = []
        for link in page_soup.find_all(
                "a", {"class": "highwire-cite-linked-title"})[:max_number]:
            uri = link.get('href')
            links.append({'title': link.text, 'biorxiv_url': base_url + uri})

        return links

    def _get_papers_list_biorxiv(self, query, max_number):
        papers = []
        url = self.search_url.format(query)
        page_html = request.urlopen(url).read().decode("utf-8")
        page_soup = BeautifulSoup(page_html, "lxml")
        links = self._get_all_links(page_soup, max_number)
        papers.extend(links)
        return papers
    
    def query_short(self, query, max_number, metadata=True, full_text=True):
        query = query.replace(' ', '%20')

        if self.search_engine == 'biorxiv':
            papers = self._get_papers_list_biorxiv(query, max_number)
        else:
            raise Exception('None implemeted search engine: {}'.format(
                self.search_engine))

        return papers

    def query_entire_papers(self, papers):

        for paper in papers:
            biorxiv_url = paper['biorxiv_url'] + '.full'
            page_html = request.urlopen(biorxiv_url).read().decode("utf-8")
            page_soup = BeautifulSoup(page_html, "lxml")

            abstract = page_soup.find("div", {
                'class': 'abstract'
            })
            if abstract is not None:
                paper['abstract'] = abstract.get_text(' ')
            else:
                paper['abstract'] = ''

            article_txt = self._get_article_content(page_soup)
            paper['full_text'] = article_txt

        return papers
    


def understand_file(lines, things_to_look_for, model):

    blocks = ["".join(lines[i:i+2000]) for i in range(0, len(lines), 2000)]

    descriptions  = []
    for idx, b in enumerate(blocks):
        start_line_number = 2000*idx+1
        end_line_number = 2000*idx+1 + len(b.split("\n"))
        prompt = f"""Given this (partial) file from line {start_line_number} to line {end_line_number}: 

``` 
{b}
```

Here is a detailed description on what to look for and what should returned: {things_to_look_for}

The description should short and also reference crtical lines in the script relevant to what is being looked for. Only describe what is objectively confirmed by the file content. Do not include guessed numbers. If you cannot find the answer to certain parts of the request, you should say "In this segment, I cannot find ...".
"""

        completion = complete_text_claude(prompt, model = model, log_file=None)
        descriptions.append(completion)
    if len(descriptions) == 1:
        return descriptions[0]
    else:
        descriptions = "\n\n".join(["Segment {idx}: \n\n" + s for s in descriptions])
        prompt = f"""Given the relevant observations for each segments of a file, summarize to get a cohesive description of the entire file on what to look for and what should returned: {things_to_look_for}

{descriptions}
"""

        completion = complete_text(prompt, model = model, log_file=None)

        return completion
    
    
def what_to_query(current_prompt, model):
    
    prompt = '''
You are an expert at literature review. You are given the current state of the research problem and some previously done research: \n \n
{}
Your task is to come up with a one-line very focussed query without any additonal terms surrounding it to search relevant papers which you think would help the most in making progress on the provided research problem. 
    '''.format(current_prompt)
    
    print(current_prompt)
    query = complete_text(prompt=prompt, model = model, log_file='paper_search.log')
    return query

def biorxiv_search(query, max_number, folder_name = ".", **kwargs):
    
    br = MyBiorxivRetriever()
    papers = br.query_short(query, max_number)
    #implement some logic here to let the LLM choose which papers it wants to read from a huge set of just titles
    papers_full = br.query_entire_papers(papers)
    return papers

def arxiv_search(query, max_papers, folder_name = ".", **kwargs):
    
    client = arxiv.Client()

    search = arxiv.Search(
        query = query,
        id_list = [],
        max_results = max_papers,
        sort_by = arxiv.SortCriterion.Relevance,
    )
    
    observation = ""
    for paper in client.results(search):
        observation += "\n" + paper.title + "\n\n" + paper.summary + "\n"

    return observation

def scholar_search(query, max_papers , folder_name = ".", **kwargs):
    
    search_query = scholarly.search_pubs(query)
    scholarly.pprint(next(search_query))

    search = arxiv.Search(
        query = query,
        id_list = [],
        max_results = max_papers,
        SortCriterion = arxiv.SortCriterion.Relevance,
        SortOrder = arxiv.SortOrder.Descending
    )
    
    observation = ""
    for result in arxiv.Client().results(search):
        observation += "\n" + result.title + "\n\n" + result.summary + "\n"

    return observation


def paperqa_search(query, max_papers, folder = '.', **kwargs):
    
    from paper_scrapper import paper_scrapper
    import paperqa
    
    papers = paperscraper.search_papers(query, limit = max_papers)
    docs = paperqa.Docs()
    for path,data in papers.items():
        try:
            docs.add(path)
        except ValueError as e:
            print('Could not read', path, e)

    answer = docs.query(query)
    return answer

def get_lit_review(prompt, model, max_number):

    lit_review = ""
    import time
    import random
    timer = random.randint(0,1e6)
    pubmed = PubMed(tool="MyTool", email=f"{timer}@email.address")
    pubmed._rateLimit = 11

    # return lit_review
    while True:
        query = what_to_query(prompt, model)
        print(query)
        papers = list(pubmed.query(query, max_results=max_number)) 
        while True:
            try:
                papers = list(pubmed.query(query, max_results=max_number))
                break
            except:
                print("Rate limit reached. Waiting for 10 seconds")
                time.sleep(10)
            
        
        for i, paper in enumerate(papers):
            lit_review += '\n' + paper.title + '\n'
            # lit_review += paper['biorxiv_url'] + '\n'
            prompt_for_summary = str(paper.title) + '\n' +  str(paper.abstract) + '\n' +  str(paper.methods) + '\n' + str(paper.conclusions) + '\n' + str(paper.results) + '\n'
            summarized_paper = understand_file(prompt_for_summary, f"general information that points to genes used for the protein production, some potential pathways, or anything else that will help make progress based on the current state of the research as given by: {prompt}", model)
            lit_review += summarized_paper + '\n \n'
        if len(lit_review)>10:
            break
    
    return str(lit_review)
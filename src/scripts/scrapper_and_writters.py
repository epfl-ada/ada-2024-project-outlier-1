import requests
from bs4 import BeautifulSoup
import csv
from tqdm import tqdm
from time import sleep



def scrape_wikipedia_articles(article_titles):
    """
    Scrapes the content of a list of Wikipedia articles and returns the links those articles contain.
    
    Parameters:
    article_titles (list): A list of Wikipedia article titles to scrape.
    
    Returns:
    dict: A dictionary where the keys are the article titles and the values are the links those articles contain.
    """
    article_links = {}
    article_names = []
    
    for title in tqdm(article_titles, desc="Scraping Wikipedia articles", unit="article"):
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        response = requests.get(url)
        
        sleep(0.1) # Be polite to Wikipedia servers
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            article_links[title] = []
            article_names.append(title)
            
            
            # Find the direct link to the current article title
            for link in soup.find_all("a", href=lambda href: href and href.startswith("/wiki/")):
                list_href = link.get("href").splitlines()
                for href in list_href:
                    if href.split('/')[-1] in article_titles and href.split('/')[-1] != title:
                        article_links[title].append(href.split('/')[-1])
        else:
            print(f"Error scraping {title}: {response.status_code}")
    
    return article_links, article_names



def export_dict_links_to_csv(article_links, output_file):
    """
    Exports the direct links to the Wikipedia articles to a CSV file.
    
    Parameters:
    article_links (dict): A dictionary where the keys are the article titles and the values are the direct links to those articles.
    output_file (str): The path to the output CSV file.
    """
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["linkSource", "linkTarget"])
        
        for source, targets in article_links.items():
            if len(targets) > 0:
                for target in targets:
                    writer.writerow([source, target])


def export_df_links_to_csv(df, output_file):
    """
    Exports the links from a DataFrame to a CSV file with columns "linkSource" and "linkTarget".
    
    Parameters:
    df (pandas.DataFrame): A DataFrame containing the links. It should have columns that can be renamed to "linkSource" and "linkTarget".
    output_file (str): The path to the output CSV file.
    """
    # Ensure the DataFrame has the correct column names
    df = df.rename(columns={df.columns[0]: "linkSource", df.columns[1]: "linkTarget"})
    
    # Select only the required columns in case there are more
    df = df[["linkSource", "linkTarget"]]
    
    # Export to CSV
    df.to_csv(output_file, index=False)


def export_articles_to_csv(articles, output_file):
    """
    Exports the Wikipedia articles names to a CSV file.
    
    Parameters:
    article_links (dict): A dictionary where the keys are the article titles and the values are the direct links to those articles.
    output_file (str): The path to the output CSV file.
    """
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["article"])
        
        for article in articles:
            writer.writerow([article])



def export_categories_to_csv(article_categories, output_file):
    """
    Exports the Wikipedia articles's given categories to a CSV file.
    
    Parameters:
    article_links (dict): A dictionary where the keys are the article titles and the values are the direct links to those articles.
    output_file (str): The path to the output CSV file.
    """
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["article", "category"])
        
        for article, category in article_categories.items():
            writer.writerow([article, category])



def write_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['# Hierarchical categories of all articles.'])
        writer.writerow(['# Many articles have more than one category. Some articles have no category.'])
        writer.writerow(['# Article names are URL-encoded; e.g., in Java they can be decoded using java.net.URLDecoder.decode(articleName, "UTF-8").'])
        writer.writerow(['# FORMAT:   article   category'])
        writer.writerow(['#'])
        writer.writerows(data)
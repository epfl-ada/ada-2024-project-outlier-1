import pandas as pd
import numpy as np

# loading all functions in the script
__all__ = ['loading_articles_links', 'articles_in_common', 'load_spm_2007']


def loading_articles_links(year, raw = False): 
    """
    Loading the links and article names list for 2007 or 2024.

    Args:
        year (str): year of the desired wikipedia's version. Can either be 2007 or 2024.
        raw (bool, optional): Put to true if want to work with the unprocessed data (only for 2024). Defaults to False.

    Returns:
        article_names (DataFrame): list of all Wikipedia's article names of the desired year
        links (DataFrame): all pair of links found in the network created by wikipedia's articles.
    """
    # raise a value error if the date specified is neither 2007 nor 2024
    #assert year == "2007" or year == 2024
    
    DATA_PATH = 'data/' + year + '/'
    if (year == '2007') :
        article_names = pd.read_csv(DATA_PATH + 'articles.tsv', sep='\t', comment='#', names=['article_'+ year])
        links = pd.read_csv(DATA_PATH + 'links.tsv', sep='\t', comment='#', names=['linkSource', 'linkTarget'] )
    
    if (year == '2024') :
        if raw:
            DATA_PATH += 'raw_'
        article_names = pd.read_csv(DATA_PATH + 'articles2024.csv', skiprows=2, names=['article_'+ year])
        links = pd.read_csv(DATA_PATH + 'links2024.csv', skiprows=1, names=['linkSource', 'linkTarget'] )
    
    return article_names, links

def articles_in_common(list1, list2) :
    '''
    Returns list of articles names that appear in both list1 and list2
    '''

    return set(list1['linkSource']).intersection(list2['linkSource'])

def load_spm_2007():
    '''
    Load the shortest path matrix from the file shortest-path-distance-matrix.txt given in the original wikispeedia dataset
    '''

    filename = 'data/2007/shortest-path-distance-matrix.txt'
    matrix = []
    
    with open(filename, 'r') as file:
        for line in file:
            # Skip lines that start with '#', as they are comments
            if line.startswith('#') or line.startswith('\n'):
                continue

            # Convert each character in the line to an integer or None for unreachable paths
            row = [(int(char) if char != '_' else float('inf')) for char in line.strip()]
            matrix.append(row)
    
    return np.array(matrix)
import pandas as pd
import numpy as np
import os as os

# loading all functions in the script
__all__ = ['loading_articles_links', 'loading_paths', 'loading_cleaned_categories', 'load_spm_2007']


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
        article_names = pd.read_csv(os.path.join(DATA_PATH , 'articles.tsv'), sep='\t', comment='#', names=['article_'+ year])
        links = pd.read_csv(os.path.join(DATA_PATH, 'links.tsv'), sep='\t', comment='#', names=['linkSource', 'linkTarget'] )
    
    if (year == '2024') :
        if raw:
            article_names = pd.read_csv(os.path.join(DATA_PATH, 'raw_articles2024.csv'), skiprows=[0,1], names=['article_'+ year])
        else:
            article_names = pd.read_csv(os.path.join(DATA_PATH, 'articles2024.csv'), skiprows=[0,1], names=['article_'+ year])
        links = pd.read_csv(os.path.join(DATA_PATH, 'links2024.csv'), skiprows=[0,1], names=['linkSource', 'linkTarget'] )
    
    return article_names, links

def loading_paths(): 
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
    
    DATA_PATH = 'data/2007/'
    path_finished = pd.read_csv(os.path.join(DATA_PATH, 'paths_finished.tsv'), sep='\t', comment='#', names=['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'rating'])
    path_unfinished = pd.read_csv(os.path.join(DATA_PATH, 'paths_unfinished.tsv'), sep='\t', comment='#', names=['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'target', 'type'])


    def find_suitable_end(x):
        '''
            Return the last article in the path that is not a way back denoted by '<'.
        '''
        i=1
        while x[-i]=='<':
            i+=1
        return x[-i]

    def prepare_df(df):
        # drop broken path (NaN)
        df = df.dropna(subset='path')
        # convert path in list of str
        df['path'] = df.path.str.split(';')

        # create new columns for the start and last visited article 
        df['start'] = df.path.map(lambda x: x[0])
        df['end'] = df.path.map(find_suitable_end)
        return df
    
    path_finished = prepare_df(path_finished)
    path_unfinished = prepare_df(path_unfinished)
    
    return path_finished, path_unfinished


def loading_cleaned_categories(): 
    """
    Loading the categories.

    Returns:
        categories (DataFrame): DataFrame of all Wikipedia's article names and their categories cleaned
    """
    
    DATA_PATH = 'data/2007/'
    categories = pd.read_csv(os.path.join(DATA_PATH, 'categories.tsv'), sep='\t', comment='#', names=['article', 'category'])
    # convert str to list by splitting on '.'
    categories.category = categories.category.str.split('.')
    # remove "subject" (starts by this everywhere)
    categories.category = categories.category.apply(lambda x: x[1:])
    categories['main_category'] = categories.category.apply(lambda x: x[0])
    return categories




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
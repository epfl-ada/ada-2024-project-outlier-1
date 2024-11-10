import pandas as pd
import numpy as np

# loading all functions in the script
__all__ = ['loading_articles_links', 'articles_in_common']


def loading_articles_links(year): 
    '''
    Loading the links and article names list for 2007 or 2024
    '''
    if (year == '2007') :
        article_names = pd.read_csv('data/'+year+'/articles.tsv',sep='\t', comment='#', names=['article_'+year])
        links = pd.read_csv('data/'+year+'/links.tsv',sep='\t', comment='#', names=['linkSource', 'linkTarget'] )
    
    if (year == '2024') :
        article_names = pd.read_csv('data/'+year+'/articles2024.csv')
        links = pd.read_csv('data/'+year+'/links2024.csv')
    
    return article_names, links

def articles_in_common(list1, list2) :
    '''
    Returns list of articles names that appear in both list1 and list2
    '''

    return set(list1['linkSource']).intersection(list2['linkSource'])
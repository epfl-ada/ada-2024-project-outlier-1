import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind 
import seaborn as sns
import networkx as nx

__all__ = ['plot_average_links_per_page','creating_graph',
           'computing_shortest_path_matrix',
           'computing_difference_spm',
           'plotting_difference_heatmap',
           'get_sankey_data']


def plot_average_links_per_page(links2007, links2024) :
    '''
    bar plot of the mean number of links per pages, and compute independent t test 
    '''
    summary2007 = links2007.groupby(by='linkSource')
    summary2024 = links2024.groupby(by='linkSource')

    summary2007_df = summary2007.apply(lambda x: x, include_groups=False)
    summary2024_df = summary2024.apply(lambda x: x, include_groups=False)

    ### T test to find wheter those two means are significantly different : 
    # Null Hyp : the two means are not significantly different 
    # Alt hyp : they are significantly different 

    TtestResult, pvalue = ttest_ind(summary2007.count(), summary2024.count())
    print('T test p value :', pvalue)

    # p value = 1.04 e-305 << alpha = 0.05, the two distributions have significantly different means
    fig = plt.figure(figsize=(8,4))

    plt.subplot(1, 2, 1)
    plt.bar(x=['2007', '2024'], height= [summary2007_df.shape[0], summary2024_df.shape[0]])
    plt.ylabel('Total number of hyperlinks')
    plt.title('Difference in total number of hyperlinks')


    plt.subplot(1, 2, 2)
    plt.bar(x=['2007', '2024'], height= [np.mean(summary2007.count()), np.mean(summary2024.count())])
    plt.ylabel('Average number of links / page')
    plt.title('Mean number of links per page')

    plt.tight_layout()

    plt.show()

def creating_graph(links_list, articles_list) :
    '''
    Creating a directed graph from the articles list, with edges from the links list 
    '''
    G = nx.DiGraph()
    G.add_nodes_from(np.unique(articles_list))
    G.add_edges_from(links_list.to_numpy())
    return G

def computing_shortest_path_matrix(G, articles_list) :
    '''
    Computing the shortest path matix according to the Floyd Warshall Algorithm
    Warning : this takes approximately 5 to 10 min to run.
    ''' 
    return nx.floyd_warshall_numpy(G, nodelist= np.unique(articles_list.iloc[:,0]))

def computing_difference_spm(spm1, spm2):
    ''' 
    Function to compare the two shortest path matrix. Infinite values are replaced by 0 to avoid having inf to 4 hops being considered as inf. 
    Returns the difference of spm2 - spm1
    '''
    spm1 = np.where(spm1 == float('inf'), 0., spm1)
    spm2 = np.where(spm2 == float('inf'), 0., spm2)
    return spm2 - spm1

def plotting_difference_heatmap(spm1, spm2) :
    '''
    Visualise the difference in shortest paths using a heatmap.
    Red : the path is shorter in spm2 than spm1
    Blue : the path is longer in spm2 than spm1
    '''
    data = computing_difference_spm(spm1,spm2)

    sns.heatmap(data, vmin=-9, vmax=9, cmap='vlag')
    plt.title('Difference in shortest path')



def get_sankey_data(df, categories, type_data, suffix_fn='1'):
    """
    Produce formatted datafile to create a sankey diagram on https://app.flourish.studio/. 

    Args:
        df (pandas.DataFrame): dataframe containing the data
        categories (pandas.DataFrame): dataframe containing the categories of each article
        type_data (str): f for finished paths df, unf for unfinished paths df and links for links df

    Returns:
        distrib (np.array of float): distribution of the number of links/paths between categories
        tot_links (int): total number of links/paths
    """

    if type_data=='f' or type_data=='unf':
        col1 = 'source_cat'
        col2 = 'target_cat'
        # to be done in data treatment
        # df['source_cat'] = df_cat.main_category.loc[df.start].values
        # df['target_cat'] = df_cat.main_category.loc[df.end].values
        # if type_data=='unf':
        #     df['target_cat'] = df_cat.main_category.loc[df.target].values
    elif type_data=='links':
        col1 = 'catSource'
        col2 = 'catTarget'
    else:
        print('type_data parameter unrecognized, return -1')
        return -1
    


    grouped_by_cat = df.groupby([col1, col2])#.apply(lambda x: x).drop(columns=['source_cat', 'target_cat']).sort_index()
    # grouped_by_cat

    cat = list(categories.main_category.value_counts().sort_index().index)

    # main_cat
    source = []
    target = []
    value = []
    for (scat, ecat), group in grouped_by_cat:
        source.append(scat)
        target.append(ecat)
        value.append(len(group))
    sankey_data_finished = pd.DataFrame(zip(source, target, value), columns=['source', 'target', 'val'])
    sankey_data_finished.to_csv(f'./data/sankey/sankey_data_{type_data}_v'+suffix_fn+'.csv', index=False)
    distrib = np.array(value)/sum(value)
    tot_links = sum(value)
    return distrib, tot_links
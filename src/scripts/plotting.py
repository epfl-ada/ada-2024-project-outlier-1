import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind 
import seaborn as sns
import networkx as nx
from matplotlib.patches import Patch


__all__ = ['plot_average_links_per_page',
           'plot_difference_links_article',
           'creating_graph',
           'computing_shortest_path_matrix',
           'computing_difference_spm',
           'plotting_difference_heatmap']


def plot_average_links_per_page(links2007, links2024) :
    '''
    bar plot of the mean number of links per pages, and compute independent t test 
    '''
    summary2007 = links2007.groupby(by='linkSource')
    summary2024 = links2024.groupby(by='linkSource')

    ### T test to find wheter those two means are significantly different : 
    # Null Hyp : the two means are not significantly different 
    # Alt hyp : they are significantly different 

    TtestResult, pvalue = ttest_ind(summary2007.count(), summary2024.count())
    print('T test p value :', pvalue)

    # p value = 1.04 e-305 << alpha = 0.05, the two distributions have significantly different means
    fig = plt.figure(figsize=(8,4))

    plt.subplot(1, 2, 1)
    ax = sns.histplot(summary2007.count()['linkTarget'], kde=True, stat='density', color='orange', label='2007')
    ax = sns.histplot(summary2024.count()['linkTarget'], kde=True, stat='density', color='blue', label='2024')
    ax.set(title='Distribution of the number of links per article', xlabel='number of links', ylabel='pages')
    plt.ylabel('Total number of hyperlinks')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(x=['2007', '2024'], height= [np.mean(summary2007.count()), np.mean(summary2024.count())])
    plt.ylabel('Average number of links / page')
    plt.title('Mean number of links per page')

    plt.tight_layout()
    plt.show()

def plot_difference_links_article(links2007, links2024) :
    '''
    Plotting differences in links count per articles in 2004 vs 2007 
    Above zero : there are more links in 2024 
    Below zero : there are more links in 2007
    '''
    
    # Aggregate counts of links per linkSource for each year
    count_2007 = links2007.groupby(by='linkSource').size()
    count_2024 = links2024.groupby(by='linkSource').size()

    # we sort 2007 articles in decreasing order of number of links per article
    sorted_values_2007 = count_2007.sort_values(ascending=False)
    # we sort 2024 in the same order as 2007 to be able to compare them 
    sorted_values_2024 = count_2024.reindex(sorted_values_2007.index)

    difference= sorted_values_2024 - sorted_values_2007
    colors = [ 'lightseagreen' if val>0 else 'coral' for val in difference]

    fig = plt.figure(figsize=(10,6))
    ax = sns.barplot(x= difference.index, y = difference, palette= colors)
    ax.set_xticklabels([])

    ax.set(title='Difference in Number of links per article', 
        xlabel='Source Articles', ylabel='#links/article in 2024 - 2007')

    ax.set_ylim([-100,250])
    ax.get_xaxis().set_visible(False)

    legend_elements = [
        Patch(facecolor='lightseagreen', edgecolor='black', label='More links in 2024'),
        Patch(facecolor='coral', edgecolor='black', label='Less links in 2024')
    ]
    plt.legend(handles=legend_elements, title='Legend', loc='upper right')

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
    plt.xlabel('articles list')
    plt.ylabel('articles list')
    plt.title('Difference in shortest path')
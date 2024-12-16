import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind 
import seaborn as sns
import networkx as nx
from matplotlib.patches import Patch
import random

import matplotlib as mpl
from collections import defaultdict
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import log10


__all__ = ['plot_average_links_per_page',
           'plot_difference_links_article',
           'creating_graph',
           'computing_shortest_path_matrix',
           'computing_difference_spm',
           'plotting_difference_heatmap',
           'computing_mean_shortest_path',
           'plot_degree_distribution',
           'plot_distribution_path_length',
           'plot_pagerank',
           'get_sankey_data',
           'get_multistep_sankey_data',
           'plotly_save_to_html',
           'plot_heatmap',
           'plot_heatmap_differences',
           'colorscale_cmap',
           'get_palette_cat',
           'plot_cat_pie_chart']


def plot_average_links_per_page(links2007, links2024, articles, graph_based=False) :
    '''
    bar plot of the mean number of links per pages, and compute independent t test 
    Graph based approach can be used when graph_based is set to True to take isolated nodes into account 
    Otherwise just computes the distribution based on the links list
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

    if(graph_based) : 
        ### creating Graphs 
        G_2007 = creating_graph(links2007, articles)
        G_2024 = creating_graph(links2024, articles)

        plt.subplot(1, 2, 1)
        plot_degree_distribution(G_2007, G_2024)
    else :
        # Distributions only using the number of links 

        plt.subplot(1, 2, 1)
        ax = sns.histplot(summary2007.count()['linkTarget'], kde=True, stat='density', color='orange', label='2007')
        ax = sns.histplot(summary2024.count()['linkTarget'], kde=True, stat='density', color='blue', label='2024')
        ax.set(title='Distribution of the number of links per article', xlabel='number of links', ylabel='number of pages')
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
 
    ax.set_ylim([-100,250])

    for tick in ax.get_xticklines():
        tick.set_visible(False)

    for label in ax.get_xticklabels():
        label.set_visible(False)

    ax.set_xlabel("Source Articles")

    ax.set(title='Difference in Number of links per article', 
        xlabel='Source Articles', ylabel='#links/article in 2024 - 2007')

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
    Function to compare the two shortest path matrix. Infinite values are replaced by 10 (maximum shortest path value) 
    to avoid having inf to 4 hops being considered as inf. 
    Returns the difference of spm2 - spm1
    '''
    spm1 = np.where(spm1 == float('inf'), 10, spm1)
    spm2 = np.where(spm2 == float('inf'), 10, spm2)
    return spm2 - spm1

def plotting_difference_heatmap(spm1, spm2) :
    '''
    Visualise the difference in shortest paths using a heatmap.
    Red : the path is shorter in spm2 than spm1
    Blue : the path is longer in spm2 than spm1
    '''
    data = computing_difference_spm(spm1,spm2)

    sns.heatmap(data, vmin=-9, vmax=9, cmap='icefire')
    plt.xlabel('articles list')
    plt.ylabel('articles list')
    plt.title('Difference in shortest path')

def computing_mean_shortest_path(spm) : 
    spm_values = [x if x != np.inf else 0 for x in spm]
    return np.mean(spm_values)

def plot_pagerank(G, year, threshold_value=0.005) :
    '''
    plot the network of the top threshold_value% nodes based on pagerank centrality using Plotly
    '''
    # computing pagerank
    pagerank = nx.pagerank(G)
    # Determine the threshold for top 20% of PageRank values
    threshold = sorted(pagerank.values(), reverse=True)[int(len(pagerank) * threshold_value)]

    # Filter nodes based on the threshold
    filtered_nodes = [node for node, score in pagerank.items() if score > threshold]

    # Create a subgraph with only the filtered nodes
    G_filtered = G.subgraph(filtered_nodes)

    # Get positions for nodes using a spring layout
    pos = nx.kamada_kawai_layout(G_filtered)

    # Extract node and edge data for Plotly
    node_x = []
    node_y = []
    node_size = []
    node_color = []
    node_text = []

    for node in G_filtered.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_size.append(pagerank[node] * 10000)  # Scale node size by PageRank
        node_color.append(pagerank[node])  # Use PageRank as color
        node_text.append(f"{node}<br>PageRank: {pagerank[node]:.4f}")

    edge_x = []
    edge_y = []

    for edge in G_filtered.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)  # Break between edges
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    # Create the edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    # Create the node trace with text
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",  # Display both markers and text
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="PageRank"),
            line_width=2
        ),
        text=node_text,  # Node names
        textposition="middle center",  # Position text above the nodes
        textfont=dict(size=10, color="black"),  # Font customization
        hoverinfo="text"
    )

    # Create the Plotly figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Interactive Network Graph for "+year
                        +" (Top "+str(threshold_value*100)+"% by PageRank)",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=0, l=0, r=0, t=40),
                        annotations=[dict(
                            text="Node size and color proportional to PageRank",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.005,
                            y=-0.002
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))

    return fig

def get_sankey_data(df, categories, type_data, get_stats=False, suffix_fn='1'):
    """
    Produce formatted datafile to create a sankey diagram on https://app.flourish.studio/ or a heatmap. 

    Args:
        df (pandas.DataFrame): dataframe containing the data
        categories (pandas.DataFrame): dataframe containing the categories of each article
        type_data (str): f for finished paths df, unf for unfinished paths df and links for links df

    Returns:
        distrib (np.array of float): distribution of the number of links/paths between categories
        tot_links (int): total number of links/paths
    """

    if type_data=='f' or type_data=='unf' or type_data=='links':
        col1 = 'catSource'
        col2 = 'catTarget'
    else:
        print('get_sankey_data function: type_data parameter unrecognized, return -1')
        return -1
    
    cats = sorted(list(categories.main_category.unique()))

    source = []
    target = []
    value = []
    to_itself = defaultdict(float)

    if get_stats:
        means = defaultdict(float)
        maxs = defaultdict(float)
        mins = defaultdict(float)
        links_per_cat = defaultdict(float)
    
    for scat in cats:
        links_scat = 0
        group = df.loc[df[col1]==scat]
        tmp = []
        for ecat in cats:
            ggroup = group.loc[group[col2]==ecat]
            source.append(scat)
            target.append(ecat)
            value.append(len(ggroup))
            tmp.append(len(ggroup))
            if scat == ecat:
                to_itself[scat] = len(ggroup)
        if get_stats:
            tmp = np.array(tmp)
            links_scat = sum(tmp)
            to_itself[scat] = to_itself[scat]/links_scat
            links_per_cat[scat] = links_scat
            means[scat] = np.mean(tmp/links_scat)
            maxs[scat] = np.max(tmp/links_scat)
            mins[scat] = np.min(tmp/links_scat)

    
   

    tot_links = sum(value)
    distrib = np.array(value)/tot_links
    sankey_data_finished = pd.DataFrame(zip(source, target, value, distrib*100), columns=['source', 'target', 'val', 'percentage'])
    sankey_data_finished.to_csv(f'./data/sankey/sankey_data_{type_data}_v'+suffix_fn+'.csv', index=False)

    if get_stats:
        return distrib, tot_links, {'mean': means, 'min': mins, 'max': maxs, 'to_itself': to_itself, 'links_per_scats':links_per_cat}
    else:        
        return distrib, tot_links



def get_multistep_sankey_data(df, categories, get_stats=False, suffix_fn='1'):
    """
    Produce formatted datafile to create a sankey diagram with 2 steps on https://app.flourish.studio/. 

    Args:
        df (pandas.DataFrame): dataframe containing the data
        categories (pandas.DataFrame): dataframe containing the categories of each article

    Returns:
        distrib (np.array of float): distribution of the number of links/paths between categories
        tot_links (int): total number of links/paths
    """
    
    cats = sorted(list(categories.main_category.unique()))

    # main_cat
    source = []
    target = []
    value_start2end = []
    value_end2target = []
    step_from = []
    step_to = []

    
    for scat in cats:
        group = df.loc[df.catSource == scat]
        for ecat in cats:
            egroup = group.loc[group.catEnd == ecat]
            source.append(scat)
            target.append(ecat)
            value_start2end.append(len(egroup))
            step_from.append(0)
            step_to.append(1)



    for ecat in cats:
        group = df.loc[df.catEnd == ecat]
        for tcat in cats:
            tgroup = group.loc[group.catTarget == tcat]
            source.append(ecat)
            target.append(tcat)
            value_end2target.append(len(tgroup))
            step_from.append(1)
            step_to.append(2)


    tot_links_start2end = sum(value_start2end)
    tot_links_end2target = sum(value_end2target)
    distrib_start2end = np.array(value_start2end)/tot_links_start2end
    distrib_end2target = np.array(value_end2target)/tot_links_end2target
    distrib = np.concatenate((distrib_start2end, distrib_end2target))
    sankey_data_finished = pd.DataFrame(zip(source, target, value_start2end+value_end2target, distrib*100, step_from, step_to), columns=['source', 'target', 'val', 'percentage', 'step_from', 'step_to'])
    sankey_data_finished.to_csv(f'./data/sankey/sankey_data_multistep_v'+suffix_fn+'.csv', index=False)
      
    return distrib_start2end, distrib_end2target, tot_links_start2end, tot_links_end2target


def plotly_save_to_html(fig, fn):
    pio.show(fig)
    pie_html = pio.to_html(fig)

    with open(fn+'.html', 'w') as f:
        f.write(pie_html)




def plot_heatmap(vals, names, num_links, type_plot, vmin=0, vmax=0, gamma=0.47, stats=False):
    fig = go.Figure()

    if type_plot=='links':
        fn = 'links_categories'
        title = 'Category flows'
        xlabel = 'Link in'
        ylabel = 'Link out'
    elif type_plot=='unf_start': 
        fn = f'categories_unfinished_paths_start2end'
        title = f'Categories of start and end articles for unfinished paths'
        xlabel = 'End article category'
        ylabel = 'Source article category'
    elif type_plot=='unf_target': 
        fn = f'categories_unfinished_paths_end2target'
        title = f'Categories of end and target articles for unfinished paths'
        xlabel = 'Target article category'
        ylabel = 'End article category'
    else:
        fn = f'categories_{type_plot}inished_paths_start2target'
        title = f'Categories of start and target articles for {type_plot}inished paths'
        xlabel = 'Target article category'
        ylabel = 'Source article category'


    all_to_cat_perc = np.array([np.sum(vals*100, axis=0)]*len(vals))
    cat_to_all_perc = np.array([ [i]*len(vals) for i in np.sum(vals*100, axis=1)])

    all_to_cat_counts = np.array([np.sum(vals*num_links, axis=0)]*len(vals))
    cat_to_all_counts = np.array([ [i]*len(vals) for i in np.sum(vals*num_links, axis=1)])

    fig.add_trace(go.Heatmap(
        z = vals*100,
        x = names,
        y = names,
        customdata = np.dstack((np.round(vals*num_links,0), all_to_cat_perc, all_to_cat_counts, cat_to_all_perc, cat_to_all_counts)),
        hovertemplate = "* → %{x}: %{customdata[1]:0.3f}%, %{customdata[2]} counts <br>" +
            "%{y} → *: %{customdata[3]:0.3f}%, %{customdata[4]} counts <br>" +
            "%{y} → %{x}: %{z:0.3f}%, %{customdata[0]} counts <extra></extra>",
        hoverlabel_font_size = 18,
        colorscale = colorscale_cmap('plasma', vals*100, gamma, vmin, vmax),
    ))
    
    fig.update_layout(
        width = 800,
        height = 800,
        font_size = 18,
        yaxis_scaleanchor="x",
        title = dict(
            text = title,
            xanchor = 'center',
            x = 0.5)
    )

    fig.update_xaxes(
        title_text = xlabel
    )
    
    fig.update_yaxes(
        title_text = ylabel
    )

    if vmax!=0:
        fig.data[0].update(zmin=0, zmax=vmax)

    plotly_save_to_html(fig, fn)

    if stats:
        return dict(zip(names, np.sum(vals*100, axis=0))), dict(zip(names, np.sum(vals*100, axis=1)))



def plot_heatmap_differences(distrib1, distrib2, names, tot_links_1, vmin=0, vmax=0, gamma=0):
    fig = go.Figure()

    fn = f'categories_differences'
    title = f'Difference in distribution of start and target articles<br>categories between finished and unfinished paths'
    xlabel = 'Target article category'
    ylabel = 'Source article category'

    distrib1[distrib1==0] = 1/tot_links_1
    vals = ((distrib1-distrib2)/distrib1)

    all_to_cat_perc = np.array([np.sum(vals*100, axis=0)]*len(vals))
    cat_to_all_perc = np.array([ [i]*len(vals) for i in np.sum(vals*100, axis=1)])

    all_to_cat_counts = np.array([np.sum(vals, axis=0)]*len(vals))
    cat_to_all_counts = np.array([ [i]*len(vals) for i in np.sum(vals, axis=1)])
    

    fig.add_trace(go.Heatmap(
        z = vals*100,
        x = names,
        y = names,
        customdata = np.dstack((all_to_cat_perc, cat_to_all_perc)),
        hovertemplate = "* → %{x}: %{customdata[0]:0.3f}% <br>" +
            "%{y} → *: %{customdata[1]:0.3f}% <br>" +
            "%{y} → %{x}: %{z:0.3f}%  <extra></extra>",
        hoverlabel_font_size = 18,
        colorscale='RdBu',
        zmid=0
        # colorscale = colorscale_cmap('plasma', vals*100, gamma, vmin, vmax),
    ))
    
    fig.update_layout(
        width = 800,
        height = 800,
        font_size = 18,
        yaxis_scaleanchor="x",
        title = dict(
            text = title,
            xanchor = 'center',
            x = 0.5)
    )

    fig.update_xaxes(
        title_text = xlabel
    )
    
    fig.update_yaxes(
        title_text = ylabel
    )

    if vmax!=0:
        fig.data[0].update(zmin=0, zmax=vmax)

    plotly_save_to_html(fig, fn)



def colorscale_cmap(cmap_name, data, gamma, vmin, vmax):
    """
    Generate a Plotly-compatible colorscale in an appropriate scale.

    Parameters:
        cmap_name (str): name of the Matplotlib colormap (e.g., 'plasma').
        data (array-like): data array to compute the color scale range
        gamma (float): change colormap scale
        vmin (float): min. value for the colorscale
        vmax (float): max. value for the colorscale
    Returns:
        colorscale (list): Colorscale for use in the colorscale attribute of go.Heatmap.
    """
    if np.min(data) < 0:
        raise ValueError("Data must contain only positive values for an appropriate scale.")
    
    cmap = mpl.colormaps[cmap_name]
    
    if vmin==0 and vmax==0:
        vmin, vmax = np.min(data), np.max(data)
    norm = mpl.colors.PowerNorm(gamma, vmin=vmin, vmax=vmax)
    
    num_colors = 100
    values = np.linspace(vmin, vmax, num_colors)
    colors = [cmap(norm(v))[:3] for v in values]  # Extract RGB values
    
    # Create a Plotly-compatible colorscale
    colorscale = [
        [i / (num_colors - 1), f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})']
        for i, (r, g, b) in enumerate(colors)
    ]
    
    return colorscale


def get_palette_cat():
    fancy_palette = {"Art": "#daa520",
                    "Business_Studies": "#000000",
                    "Countries": "#e31a1c",
                    "Design_and_Technology": "#b2df8a",
                    "Religion": "#fdbf6f",
                    "Everyday_life": "#cab2d6",
                    "Geography": "#fb9a99",
                    "People": "#b15928",
                    "Language_and_literature": "#33a02c",
                    "IT": "#ff7f00",
                    "History": "#eeee55",
                    "Mathematics": "#6a3d9a",
                    "Music": "#888888",
                    "Citizenship": "#a6cee3",
                    "Science": "#1f78b4"}
    return fancy_palette



def plot_degree_distribution(G_2007, G_2024):
    """
    Plots distribution of the degrees of the nodes, using a graph based approach
    """
    degree_sequence_2007 = sorted((d for n, d in G_2007.degree()), reverse=True)
    degree_sequence_2024 = sorted((d for n, d in G_2024.degree()), reverse=True)

    ax = sns.histplot(degree_sequence_2024,
                    kde=True, stat='density', color='blue', label='2024')
    ax = sns.histplot(degree_sequence_2007,
                    kde=True, stat='density', color='orange', label='2007')

    ax.set(title='Distribution of the degrees of nodes', xlabel='degree', ylabel='fraction of nodes')
    plt.legend()
    

def plot_distribution_path_length(G_2007, G_2024, n_samples=100, start_hops=0, n_hops=10):
    """
    Plots the distribution of reachable nodes within a given number of hops in a graph, 
    based on a sample of nodes and comapre between 2007 and 2024 

    Parameters:
        - G_2007 (networkx.Graph): The 2007 graph to analyze.
        - G_2024 (networkx.Graph): The 2024 graph to analyze.
        - n_samples (int, optional): The number of nodes to sample.
        - start_hops (int, optional): The starting number of hops.
        - n_hops (int, optional): The maximum number of hops.
    Returns:
        - None
    """

    assert start_hops <= n_hops
    sampled_nodes = random.sample(list(G_2007.nodes) ,n_samples)
    G_2007_sampled = G_2007.subgraph(sampled_nodes)
    
    sampled_nodes = random.sample(list(G_2024.nodes) ,n_samples)
    G_2024_sampled = G_2024.subgraph(sampled_nodes)

    average_reachable_nodes_2007 = {}
    average_reachable_nodes_2024 = {}


    # to see the intersection of the average reachable nodes at 6 hops
    intersection_hops_2007 = 0
    intersection_hops_2024 = 0


    for i in range(start_hops,n_hops+1):
        average_reachable_nodes_2007[i] = np.mean([len(nx.single_source_shortest_path_length(G_2007, node, cutoff=i)) for node in G_2007_sampled.nodes()])
        average_reachable_nodes_2024[i] = np.mean([len(nx.single_source_shortest_path_length(G_2024, node, cutoff=i)) for node in G_2024_sampled.nodes()])

        if i == 6:
            intersection_hops_2007 = average_reachable_nodes_2007[i]
            
        if i == 5:
            intersection_hops_2024 = average_reachable_nodes_2024[i]

    average_reachable_nodes_2007 = average_reachable_nodes_2007.items()
    x,y_2007 = zip(*average_reachable_nodes_2007)

    average_reachable_nodes_2024 = average_reachable_nodes_2024.items()
    x,y_2024 = zip(*average_reachable_nodes_2024)

    sns.set_theme()

    plt.plot(x,y_2007)
    plt.plot(x,y_2024)

    plt.xlabel('Number of hops')
    plt.ylabel('Average of node reachable')
    plt.title('Average number of reachable nodes')
    plt.axhline(y=intersection_hops_2007, color='r', linestyle='--')
    plt.axvline(x=6, color='r', linestyle='--')

    plt.axhline(y=intersection_hops_2024, color='orange', linestyle='--')
    plt.axvline(x=5, color='orange', linestyle='--')

    plt.legend(['2007', '2024'])


def plot_cat_pie_chart(categories):
    counts = categories.main_category.value_counts()
    fancy_palette = get_palette_cat()
    categories['count'] = categories.main_category.apply(lambda x: counts.loc[x])

    fig = go.Figure(layout=go.Layout(width=800, height=500))

    fig.add_trace(go.Pie(
                        labels=counts.index,
                        values=counts.values,
                        marker_colors=counts.index.map(fancy_palette).fillna("white"),
                        showlegend=False,
                        texttemplate=['%{label}<br>%{percent}' if counts.loc[c]/sum(counts)>0.05 else '%{label} %{percent}' for c in counts.index]
                        )
    )            
            
    fig.update_traces(textposition=['inside' if counts.loc[c]/sum(counts)>0.05 else 'outside' for c in counts.index], textinfo='percent+label')
    fig.update_layout(font_size = 18,
                    title = dict({'text': f"Main categories", 'x': 0.5, 'xanchor': 'center'}),
                    margin=dict(l=20, r=20, t=50, b=20))
                    
    plotly_save_to_html(fig, 'pie_cat')
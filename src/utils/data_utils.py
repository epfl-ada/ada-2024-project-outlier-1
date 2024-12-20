import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import chi2_contingency
import sklearn.metrics as metrics

__all__ = ['delete_duplicates_cat',
           'art_not_in_cat',
           'articles_in_common',
           'cleaned_paths',
           'compute_stats_games',
           'chi2_contingency_test',
           'prepare_all_games',
           'get_relevant_metrics']

def delete_duplicates_cat(df_cat, relationships):
    """
    Choose a main category when an article is classified in differents main categories.

    Args:
        df_cat (pandas.DataFrame): dataframe containing the article names, the full and main category they are linked to
        relationships (list of tuples): arbitrary partial ordering between the categories.

    Returns:
        df_cat (pandas.DataFrame): dataframe containing the article names, the full and main category they are linked to, 
                                   wihtout duplicated articles anymore
        categories_duplicates (pandas.DataFrame): dataframe organized as df_cat with articles that are still assigned
                                                  to 2 differents categories (should be empty here)
    """
    # loop on all ordering relationships defined
    for cat, subcat in relationships:
        # create/update a df with the articles assigned in several main categories
        categories_duplicates = df_cat.loc[df_cat.article.duplicated(keep=False)]

        # find the ones with the main category of interest
        art = categories_duplicates.article.loc[categories_duplicates.main_category==cat].values

        # drop the ones with the main category of interest and sub category of interest
        df_cat = df_cat.drop(index=categories_duplicates.loc[(categories_duplicates.article.isin(art)) & (categories_duplicates.main_category==subcat)].index)
        
    
    categories_duplicates = df_cat.loc[df_cat.article.duplicated(keep=False)]
    return df_cat, categories_duplicates


def art_not_in_cat(df, df_cat, col, index=False):
    if index:
        target_in_cat = df[col].isin(df_cat.index)
    else:
        target_in_cat = df[col].isin(df_cat.article)
    return df[col][np.where(~target_in_cat)[0]].unique()


def articles_in_common(list1, list2) :
    '''
    Returns list of articles names that appear in both list1 and list2
    '''

    return set(list1['linkSource']).intersection(list2['linkSource'])


def cleaned_paths(df_f, df_unf, df_cat, art_to_remove, verbose=True):
    # df_cat = df_cat.set_index('article', drop=True)
    
    init_rows_f = df_f.shape[0]
    init_rows_unf = df_unf.shape[0]
    
    df_f = df_f.drop(df_f[df_f.start.isin(art_to_remove)].index)
    df_f = df_f.drop(df_f[df_f.end.isin(art_to_remove)].index)
    df_f['catSource'] = df_cat.main_category.loc[df_f.start].values
    df_f['catTarget'] = df_cat.main_category.loc[df_f.end].values    
    df_f['length'] = df_f.path.str.len()-1

    df_unf = df_unf.drop(df_unf[df_unf.start.isin(art_to_remove)].index)
    df_unf = df_unf.drop(df_unf[df_unf.target.isin(art_to_remove)].index)
    df_unf['catSource'] = df_cat.main_category.loc[df_unf.start].values
    df_unf['catTarget'] = df_cat.main_category.loc[df_unf.target].values
    df_unf['catEnd'] = df_cat.main_category.loc[df_unf.end].values
    df_unf['length'] = df_unf.path.str.len()-1

    df_f_cleaned = df_f.loc[df_f.length>0]
    df_unf_cleaned = df_unf.loc[df_unf.length>0]

    if verbose:
        # let's check how much data has been discarded
        print('After cleaning articles not in categories.tsv')
        print(f'Initial number of rows in path finished: {init_rows_f}, current number of rows: {df_f.shape[0]}, percentage of loss: {(init_rows_f-df_f.shape[0])/init_rows_f*100:.2f}%')
        print(f'Initial number of rows in path unfinished: {init_rows_unf}, current number of rows: {df_unf.shape[0]}, percentage of loss: {(init_rows_unf-df_unf.shape[0])/init_rows_unf*100:.2f}%')
        print('After cleaning articles not in categories.tsv + remove paths with length 0')
        print(f'Initial number of rows in path finished: {init_rows_f}, current number of rows: {df_f_cleaned.shape[0]}, percentage of loss: {(init_rows_f-df_f_cleaned.shape[0])/init_rows_f*100:.2f}%')
        print(f'Initial number of rows in path unfinished: {init_rows_unf}, current number of rows: {df_unf_cleaned.shape[0]}, percentage of loss: {(init_rows_unf-df_unf_cleaned.shape[0])/init_rows_unf*100:.2f}%')
    
    
    return df_f, df_unf, df_f_cleaned, df_unf_cleaned


def compute_stats_games(df, type_path):
    df_stats = pd.DataFrame(df.index.value_counts().sort_index())
    df_stats.index.names = ['start', 'target']
    df_stats[f'avg_{type_path}_path'] = df['path'].str.len().groupby(level=[0,1]).mean()
    df_stats[f'std_{type_path}_path'] = df['path'].str.len().groupby(level=[0,1]).std()
    df_stats[f'sem_{type_path}_path'] = df['path'].str.len().groupby(level=[0,1]).sem()
    df_stats[f'med_{type_path}_path'] = df['path'].str.len().groupby(level=[0,1]).median()
    df_stats[f'q25_{type_path}_path'] = df['path'].str.len().groupby(level=[0,1]).quantile(0.25)
    df_stats[f'q75_{type_path}_path'] = df['path'].str.len().groupby(level=[0,1]).quantile(0.75)
    df_stats[f'q10_{type_path}_path'] = df['path'].str.len().groupby(level=[0,1]).quantile(0.10)
    df_stats[f'q90_{type_path}_path'] = df['path'].str.len().groupby(level=[0,1]).quantile(0.90)
    df_stats[f'min_{type_path}_path'] = df['path'].str.len().groupby(level=[0,1]).min()
    df_stats[f'max_{type_path}_path'] = df['path'].str.len().groupby(level=[0,1]).max()
    
    return df_stats

def chi2_contingency_test(counts1, counts2):
    test = chi2_contingency(np.vstack((counts1, counts2)))
    return test.pvalue, test.statistic


def prepare_all_games(path_f, path_unf, cats, links, arts):

    def parse_game_serie(game):
        split = game.split(', ')
        start = split[0].strip()
        end = split[1].strip()
        return start, end

    def number_links_to_target(game, counts):
        _, end = parse_game_serie(game)
        try:
            n = counts.loc[end].values[0]
        except KeyError as ke:
            n = 0
        return n

    def find_cat(art):
        return cats.loc[art].main_category

    def shortest_path_from_graph(G, game):
        start, end = parse_game_serie(game)
        try:
            x = nx.shortest_path_length(G, source=start, target=end)
        except:
            x = 0
        return x

    path_finished_super_reduced = path_f[['start', 'end', 'path']].copy(deep=True).rename(columns={'end':'target'})
    path_unfinished_super_reduced = path_unf[['start', 'target', 'path']].copy(deep=True)

    path_finished_super_reduced['finished?'] = 1
    path_finished_super_reduced['length'] = path_finished_super_reduced.path.str.len()-1
    path_unfinished_super_reduced['finished?'] = 0
    path_unfinished_super_reduced['length'] = path_unfinished_super_reduced.path.str.len()-1

    all_games = pd.concat([path_finished_super_reduced, path_unfinished_super_reduced], axis=0, ignore_index=True)

    # makes things easier when we want only unique occurence of games
    all_games['game'] = all_games.apply(lambda x: x.start+', '+x.target, axis=1)


    countLinkToTarget = links[['linkTarget', 'linkSource']].groupby('linkTarget').count()

    G_2007 = nx.DiGraph()
    G_2007.add_nodes_from(np.unique(arts))
    G_2007.add_edges_from(links[['linkTarget', 'linkSource']].to_numpy())


    shortestPath_lst = [shortest_path_from_graph(G_2007, g) for g in all_games.game.unique()]
    shortestPath = pd.DataFrame({'shortestPath': shortestPath_lst}, index=all_games.game.unique())

    all_games['links_to_target'] = all_games.game.apply(number_links_to_target, counts=countLinkToTarget)
    all_games = all_games.loc[all_games['links_to_target']!=0]
    all_games['shortest_path'] = all_games.game.apply(lambda x: shortestPath.loc[x].values[0])
    all_games['catSource'] = all_games.start.apply(lambda x: find_cat(x))
    all_games['catTarget'] = all_games.target.apply(lambda x: find_cat(x))

    size_init = all_games.shape[0]
    print(f'Before removing path of length 0: win percentage={all_games["finished?"].mean()}, number of games={size_init}')
    all_games = all_games.loc[all_games.length>0]
    print(f'After removing path of length 0 : win percentage={all_games["finished?"].mean()}, number of games={all_games.shape[0]}')
    print(f'Percentage of games discarded: {(size_init-all_games.shape[0])/size_init*100:.2f}%')

    return all_games

def get_fp_fn_tp_tn(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    mat = metrics.confusion_matrix(y_true, y_pred)

    tp = mat[1,1] 
    tn = mat[0,0]
    fp = mat[0,1]
    fn = mat[1,0]

    return fp, fn, tp, tn

def precision_neg(y_true, y_pred):
    fp, fn, tp, tn = get_fp_fn_tp_tn(y_true, y_pred)
    try:
        pn = tn / (tn+fn)
    except:
        pn = 0
    return pn

def recall_neg(y_true, y_pred):
    fp, fn, tp, tn = get_fp_fn_tp_tn(y_true, y_pred)
    try:
        rn = tn / (tn+fp)
    except:
        rn = 0
    return rn

def f1_score_neg(y_true, y_pred):
    pn = precision_neg(y_true, y_pred)
    rn = recall_neg(y_true, y_pred)

    try:
        f1 = 2*pn*rn/(pn+rn)
    except:
        f1 = 0
    return f1


def get_relevant_metrics(ytrue, pred_proba, n=101):
    try:
        tmp = [[y[1]>i for y in pred_proba] for i in np.linspace(0, 1, n)]
    except:
        tmp = [[y>i for y in pred_proba] for i in np.linspace(0, 1, n)]
    
    acc = [metrics.accuracy_score(ytrue, t) for t in tmp]
    prec = [metrics.precision_score(ytrue, t) for t in tmp]
    rec = [metrics.recall_score(ytrue, t) for t in tmp]
    f1 = [metrics.f1_score(ytrue, t) for t in tmp]

    acc_neg = [metrics.accuracy_score(ytrue, t) for t in tmp]
    prec_neg = [precision_neg(ytrue, t) for t in tmp]
    rec_neg = [recall_neg(ytrue, t) for t in tmp]
    f1_neg = [f1_score_neg(ytrue, t) for t in tmp]

    balanced_acc = [metrics.balanced_accuracy_score(ytrue, t) for t in tmp]
    avg_f1 = 0.5*(np.array(f1)+np.array(f1_neg)) 
    
    return balanced_acc, avg_f1, prec, rec, prec_neg, rec_neg

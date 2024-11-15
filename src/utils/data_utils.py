import pandas as pd
import numpy as np

__all__ = ['delete_duplicates_cat', 'art_not_in_cat', 'articles_in_common', 'paths_with_cat_only', 'compute_stats_games']

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


def paths_with_cat_only(df_f, df_unf, df_cat, art_to_remove, verbose=True):
    # df_cat = df_cat.set_index('article', drop=True)
    
    init_rows_f = df_f.shape[0]
    init_rows_unf = df_unf.shape[0]
    
    df_f = df_f.drop(df_f[df_f.start.isin(art_to_remove)].index)
    df_f = df_f.drop(df_f[df_f.end.isin(art_to_remove)].index)
    df_f['source_cat'] = df_cat.main_category.loc[df_f.start].values
    df_f['target_cat'] = df_cat.main_category.loc[df_f.end].values    
    # df_f['source_cat'] = df_cat.main_category.loc[df_f.start == df_cat.article].values
    # df_f['target_cat'] = df_cat.main_category.loc[df_f.end == df_cat.article].values
    
    df_unf = df_unf.drop(df_unf[df_unf.start.isin(art_to_remove)].index)
    df_unf = df_unf.drop(df_unf[df_unf.target.isin(art_to_remove)].index)
    df_unf['source_cat'] = df_cat.main_category.loc[df_unf.start].values
    df_unf['target_cat'] = df_cat.main_category.loc[df_unf.target].values
    # df_unf['source_cat'] = df_cat.main_category.loc[df_unf.start == df_cat.article].values
    # df_unf['target_cat'] = df_cat.main_category.loc[df_unf.target == df_cat.article].values

    if verbose:
        # let's check how much data has been discarded
        print(f'Initial number of rows in path finished: {init_rows_f}, current number of rows: {df_f.shape[0]}, percentage of loss: {(init_rows_f-df_f.shape[0])/init_rows_f*100:.2f}%')
        print(f'Initial number of rows in path unfinished: {init_rows_unf}, current number of rows: {df_unf.shape[0]}, percentage of loss: {(init_rows_unf-df_unf.shape[0])/init_rows_unf*100:.2f}%')
    
    
    return df_f, df_unf


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
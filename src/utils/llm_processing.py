import itertools
import pandas as pd

def fix_player_path(path):
    """
    Fix the path by removing the last element if it is '<'

    Args:
        path (str) : path to fix

    Returns:
        path (str) : fixed path
    """
    result_stack = []

    for element in path:
        if element == '<':
            if result_stack:  # Remove the last element only if the stack is not empty
                result_stack.pop()
        else:
            result_stack.append(element)

    return result_stack

def process_player_path_data(df,min_games=10,max_length=50):
    """
    Process the player path data to get the path length and the number of games played by each player

    Args:
        df (pandas dataframe) : dataframe containing the player path data
        min_games (int) : minimum number of games played by a player, default 10 used to filter out players with less than 10 games
        max_length (int) : maximum length of the path, default 50 used to filter out path with length greater than 50

    Returns:
        df (pandas dataframe) : dataframe containing the player path data with the path length and the number of games played by each player
    """
    df["path"] = df["path"].str.split(";")
    df["length"] = df["path"].map(len)
    df["start"] = df["path"].str[0]
    df["end"] = df["path"].str[-1]

    df = df[["start", "end", "path", "length"]]
    df = df.groupby(["start", "end"]).apply(lambda x: x)
    df.drop(columns=["start", "end"], inplace=True)

    df["counts"] = df.groupby(by=["start", "end"],).size().sort_values(ascending=False)
    # Keep only path with more than min_games games
    df = df.loc[df["counts"]>=min_games]
    df = df[df['length']<max_length]

    return df

def remove_consecutive_duplicates(x):
    """
    Remove identical consecutive elements in a list 

    Args:
        x (list): list to clean

    Returns:
        y (list): x without 2 identical consecutive elements
    """
    return [i[0] for i in itertools.groupby(x)]

def remove_periodic_loop(x):
    """
    Remove periodic loops

    Args:
        x (list): list to clean

    Returns:
        y (list): x without periodic loops
    """
    to_remove = []
    for j in range(1, len(x)-1):
        for i in range(len(x)-j):
            if x[i:i+j]==x[i+j:i+2*j]:
                for k in range(i, i+j):
                    to_remove.append(k)

    return [i for j, i in enumerate(x) if j not in to_remove]

def stop_when_found(x):
    """
    Cut the list to the first occurence of the target word.

    Args:
        x (list): list to clean

    Returns:
        y (list): x stopping at the first appearence of word
    """
    if list(x).index(x[-1])!=len(list(x))-1:
        return x[:list(x).index(x[-1])+1]
    else:
        return x


def solution_on_already_visited_page(x):
    """
    Cut the list after first occurence of the second to last page leading to the target.
    i.e. we assume a human player always click the target word if the word is present on a page visited.
    Pb.: maybe other visited pages lead to the target word and we don't know it.

    Args:
        x (list): list to clean

    Returns:
        y (list): x stopping at the first appearence of the last page leading to the target + target.
    """

    if list(x).index(x[-2])!=len(list(x))-2:
        return x[:list(x).index(x[-2])+1]+[x[-1]]
    else:
        return x
    
def post_processing(x):
    """
    Apply post processing on a path

    Args:
        x (list): list to clean

    Returns:
        y (list): x cleaned
    """
    x = stop_when_found(x)
    x = solution_on_already_visited_page(x)
    x = remove_consecutive_duplicates(x)
    x = remove_periodic_loop(x)
    return x

def combined_finished_unfinished(path_finished, path_unfinished):
    """
    Combine the finished and unfinished paths

    Args:
        path_finished (pandas dataframe): dataframe containing the finished paths
        path_unfinished (pandas dataframe): dataframe containing the unfinished paths

    Returns:
        by_path (pandas dataframe): dataframe containing
    """
    path_finished.path = path_finished.path.str.split(';')
    path_finished["length"] = path_finished.path.map(len) 
    path_finished['start'] = path_finished.path.str[0]
    path_finished['end'] = path_finished.path.str[-1]
    path_unfinished.path = path_unfinished.path.str.split(';')
    path_unfinished["start"] = path_unfinished.path.str[0]
    path_unfinished["end"] = path_unfinished.target
    path_finished["finished"] = 1
    path_unfinished["finished"] = 0
    path_finished["count"] = 1
    path_unfinished["count"] = 1

    # combine finished and unfinished paths
    paths = pd.concat([path_finished[['start','end','count']], path_unfinished[['start','end','count']]])
    by_path = paths[["start","end","count"]].groupby(by=['start', 'end']).count().sort_values(by='count', ascending=False).reset_index()
    return by_path
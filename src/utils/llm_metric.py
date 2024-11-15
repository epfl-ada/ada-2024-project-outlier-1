def check_if_in_CI(x,model):
    """
    Check if the model performance is in the confidence interval of the player performance

    Args:
        x (pandas dataframe): row of the dataframe
        model (str): model to compare

    Returns:
        bool: True if the model performance is in the confidence interval of the player performance
    """
    return ((x['mean']-x['std'])<=x[model]) and ((x['mean']+x['std'])>=x[model])


def jaccard_similarity(x):
    """
    Compute the Jaccard similarity between two paths

    Args:
        x (pandas dataframe): row of the dataframe

    Returns:
        float: Jaccard similarity between the two paths
    """
    return len(set(x['path_x']).intersection(set(x['path_y'])))/len(set(x['path_x']).union(set(x['path_y'])))
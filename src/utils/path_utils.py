import os
from tqdm import tqdm
import pandas as pd


def fix_path(path):
    """
    Fix the path by removing the last element if it is '<'
    """
    path_elements = path.split(';')
    result_stack = []

    for element in path_elements:
        if element == '<':
            if result_stack:  # Remove the last element only if the stack is not empty
                result_stack.pop()
        else:
            result_stack.append(element)
    return result_stack


def check_if_link_exists(row, links):
    """
    Check if in the path of the row, there is a faster way to reach the target
    Also, check if the path is valid
    """
    path_list = fix_path(row['path'])
    # Convert list to a set for quick lookup and calculate target links
    path_set = set(path_list)
    target = row['target']
    
    # Filter links2024 to get only relevant links with target linkSource in path
    last_links = set(links.loc[links['linkTarget'] == target, 'linkSource'])
    intersecting_links = path_set.intersection(last_links)
    
    if intersecting_links:
        # Verify that each path[i+1] exists in the linkTarget of path[i]
        for i in range(len(path_list) - 1):
            current_node = path_list[i]
            next_node = path_list[i + 1]
            # Check if next_node is in linkTarget of current_node in links2024
            if not ((links['linkSource'] == current_node) & (links['linkTarget'] == next_node)).any():
                return None, -1  # Return None, -1 if the link is invalid
        
        # Find the index of the first occurrence in path_list
        index = next((i for i, link in enumerate(path_list) if link in intersecting_links), None)
        return path_list[index], len(path_list) - index - 1
    else:
        return path_list[-1], 0
    

def create_comparison_unfinished_path(DATA_PATH, path_unfinished, df_link2007, df_link2024, load=True):
    '''
    Create a dataframe that contains the path, target, final_link2007, final_link2024, distance2007, and distance2024

    Parameters:
    - DATA_PATH: the path to save the csv file
    - path_unfinished: the dataframe that contains the path and target
    - df_link2007: the dataframe that contains the links in 2007
    - df_link2024: the dataframe that contains the links in 2024
    - load: whether to load the csv file or process the data and save it to a csv file

    Returns:
    - df_unfinished: the dataframe that contains the path, target, final_link2007, final_link2024, distance2007, and distance2024
    '''

    if load:
        return pd.read_csv(os.path.join(DATA_PATH, 'comparison_unfinished_path.csv'))
    
    df_unfinished = []

    for _, data in tqdm(path_unfinished.iterrows(), total=len(path_unfinished), desc='Processing data'):
        output_link, distance = check_if_link_exists(data, df_link2007)
        output_link2, distance2 = check_if_link_exists(data, df_link2024)

        # add the data to the output dataframe
        df_unfinished.append({'path': data['path'], 'target': data['target'], 'final_link2007': output_link, 'final_link2024': output_link2, 'distance2007': distance, 'distance2024': distance2})
        
    df_unfinished = pd.DataFrame(df_unfinished)

    # save df_unfinished to a csv file
    df_unfinished.to_csv(os.path.join(DATA_PATH, 'comparison_unfinished_path.csv'), index=False)
    return df_unfinished


def print_result_comparison_paths(df_comparison):
    """
    Print the result of the comparison between the paths
    """
    print("==== Path possible ? ====")

    print(f"Path won't change between 2007 and 2024 : {len(df_comparison[(df_comparison.distance2007 == 0) & (df_comparison.distance2024 == 0)])}")
    print(f"Path impossible to do in 2024 but possible in 2007 : {len(df_comparison[(df_comparison.distance2007 >= 0) & (df_comparison.distance2024 == -1)])}")
    print(f"Path impossible to do in 2007 and 2024 : {len(df_comparison[(df_comparison.distance2007 == -1) & (df_comparison.distance2024 == -1)])}")
    print(f"Path still possible in 2024 : {len(df_comparison[(df_comparison.distance2007 >= 0) & (df_comparison.distance2024 >= 0)])}")

    print()
    print("==== Path faster or slower in 2024 ? ====")
    print(f"Path faster in 2024 than 2007: {len(df_comparison[df_comparison.distance2007 < df_comparison.distance2024])}")
    print(f"Path slower or equal in 2024 than 2007: {len(df_comparison[(df_comparison.distance2007 >= df_comparison.distance2024) & (df_comparison.distance2024 >= 0) & (df_comparison.distance2007 > 0)])}")


def check_if_link_exists_finished(row, links):
    path_list = fix_path(row['path'])
    if len(path_list) > 1:
        target = row['target']
        path_list.remove(target)
        
        # Convert list to a set for quick lookup and calculate target links
        path_set = set(path_list)
        
        # Filter links2024 to get only relevant links with target linkSource in path
        last_links = set(links.loc[links['linkTarget'] == target, 'linkSource'])
        intersecting_links = path_set.intersection(last_links)
        
        if intersecting_links:
            # Verify that each path[i+1] exists in the linkTarget of path[i]
            for i in range(len(path_list) - 1):
                current_node = path_list[i]
                next_node = path_list[i + 1]
                # Check if next_node is in linkTarget of current_node in links2024
                if not ((links['linkSource'] == current_node) & (links['linkTarget'] == next_node)).any():
                    return None, -1  # Return None, -1 if the link is invalid
            
            # Find the index of the first occurrence in path_list
            index = next((i for i, link in enumerate(path_list) if link in intersecting_links), None)
            return path_list[index], len(path_list) - index - 1
        else:
            return path_list[-1], 0
    else:
        return path_list[0], 0

def create_comparison_finished_path(DATA_PATH, path_finished, df_link2007, df_link2024, load=True):
    '''
    Create a dataframe that contains the path, target, final_link2007, final_link2024, distance2007, and distance2024

    Parameters:
    - DATA_PATH: the path to save the csv file
    - path_finished: the dataframe that contains the path and target
    - df_link2007: the dataframe that contains the links in 2007
    - df_link2024: the dataframe that contains the links in 2024
    - load: whether to load the csv file or process the data and save it to a csv file

    Returns:
    - df_finished: the dataframe that contains the path, target, final_link2007, final_link2024, distance2007, and distance2024
    '''

    if load:
        return pd.read_csv(os.path.join(DATA_PATH, 'comparison_finished_path.csv'))
    
    df_finished = []

    for _, data in tqdm(path_finished.iterrows(), total=len(path_finished), desc='Processing data'):
        output_link, distance = check_if_link_exists_finished(data, df_link2007)
        output_link2, distance2 = check_if_link_exists_finished(data, df_link2024)

        # add the data to the output dataframe
        df_finished.append({'path': data['path'], 'target': data['target'], 'final_link2007': output_link, 'final_link2024': output_link2, 'distance2007': distance, 'distance2024': distance2})
        
    df_finished = pd.DataFrame(df_finished)

    # save df_finished to a csv file
    df_finished.to_csv(os.path.join(DATA_PATH, 'comparison_finished_path.csv'), index=False)
    return df_finished
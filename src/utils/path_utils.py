import os
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt


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

    Parameters:
    - row: the row of the dataframe
    - links: the dataframe that contains the links

    Returns:
    - output_link: the final link
    - distance: the distance
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
    """
    Create a dataframe that contains the path, target, final_link2007, final_link2024, distance2007, and distance2024

    Parameters:
    - DATA_PATH: the path to save the csv file
    - path_unfinished: the dataframe that contains the path and target
    - df_link2007: the dataframe that contains the links in 2007
    - df_link2024: the dataframe that contains the links in 2024
    - load: whether to load the csv file or process the data and save it to a csv file

    Returns:
    - df_unfinished: the dataframe that contains the path, target, final_link2007, final_link2024, distance2007, and distance2024
    """
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

    Parameters:
    - df_comparison: the dataframe that contains the path, target, final_link2007, final_link2024, distance2007, and distance2024

    Returns:
    - None
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
    """
    Check if in the path of the row, there is a faster way to reach the target
    Also, check if the path is valid

    This function is used for the finished path

    Parameters:
    - row: the row of the dataframe
    - links: the dataframe that contains the links

    Returns:
    - output_link: the final link
    """
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
    """
    Create a dataframe that contains the path, target, final_link2007, final_link2024, distance2007, and distance2024

    Parameters:
    - DATA_PATH: the path to save the csv file
    - path_finished: the dataframe that contains the path and target
    - df_link2007: the dataframe that contains the links in 2007
    - df_link2024: the dataframe that contains the links in 2024
    - load: whether to load the csv file or process the data and save it to a csv file

    Returns:
    - df_finished: the dataframe that contains the path, target, final_link2007, final_link2024, distance2007, and distance2024
    """

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

def _categorize_efficiency(diff):
    """
    Categorize the efficiency difference into 2007, No Change, or 2024 (tools for plotting)

    Parameters:
    - diff: the difference in the efficiency

    Returns:
    - the category of the efficiency
    """
    if diff > 0:
        return '2024'
    elif diff < 0:
        return '2007'
    else:
        return 'No Change'


def plot_comarison_length_path(df_data_finished, df_data_unfinished, title="Paths shortened in 2007 vs 2024 (length)"):
    """
    Plot comparison between the shortened paths in 2007 and 2024 (in terms of length)

    Parameters:
    - df_data_finished: the dataframe that contains the finished path, target, final_link2007, final_link2024, distance2007, and distance2024
    - df_data_unfinished: the dataframe that contains the unfinished path, target, final_link2007, final_link2024, distance2007, and distance2024
    - title: the title of the plot

    Returns:
    - None
    """
        
    df_data_finished['efficiency_diff'] = df_data_finished['distance2024'] - df_data_finished['distance2007']
    df_data_finished['efficiency_category'] = df_data_finished['efficiency_diff'].apply(_categorize_efficiency)
    category_counts_finished = df_data_finished['efficiency_category'].value_counts()

    df_data_unfinished['efficiency_diff'] = df_data_unfinished['distance2024'] - df_data_unfinished['distance2007']
    df_data_unfinished['efficiency_category'] = df_data_unfinished['efficiency_diff'].apply(_categorize_efficiency)
    category_counts_unfinished = df_data_unfinished['efficiency_category'].value_counts()

    # Reordering categories for consistent plotting
    categories_ordered = ['2007', 'No Change', '2024']

    category_counts_finished = category_counts_finished.reindex(categories_ordered)
    category_counts_unfinished = category_counts_unfinished.reindex(categories_ordered)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    sns.barplot(
        x=category_counts_unfinished.index,
        y=category_counts_unfinished.values,
        palette=["#1f77b4", "#cccccc", "#ff7f0e"],  # Blue, Grey, Orange
        ax=axs[0],
        hue=category_counts_unfinished.index
    )

    axs[0].set_title("Unfinished Paths")
    axs[0].set_ylabel("Number of Paths")
    axs[0].set_xlabel("Efficiency Category")
    axs[0].set_ylim(0, 18000)

    for i, v in enumerate(category_counts_unfinished.values):
        axs[0].text(i, v + 50, str(v), ha='center', color='black')

    sns.barplot(
        x=category_counts_finished.index,
        y=category_counts_finished.values,
        palette=["#1f77b4", "#cccccc", "#ff7f0e"],  # Blue, Grey, Orange
        ax=axs[1],
        hue=category_counts_finished.index
    )

    axs[1].set_title("Finished Paths")
    axs[1].set_ylabel("Number of Paths")
    axs[1].set_xlabel("Efficiency Category")
    axs[1].set_ylim(0, 18000)

    for i, v in enumerate(category_counts_finished.values):
        axs[1].text(i, v + 50, str(v), ha='center', color='black')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_comparison_clicks_saved(df_data_finished, df_data_unfinished, title="Paths shortened in 2007 vs 2024 (clicks saved)"):
    """
    Plot comparison between the shortened paths in 2007 and 2024 (in terms of clicks saved)
    
    Parameters:
    - df_data_finished: the dataframe that contains the finished path, target, final_link2007, final_link2024, distance2007, and distance2024
    - df_data_unfinished: the dataframe that contains the unfinished path, target, final_link2007, final_link2024, distance2007, and distance2024
    - title: the title of the plot

    Returns:
    - None
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    sns.kdeplot(df_data_unfinished['distance2007'],
    label="2007",
    color="#1f77b4",
    fill=True,
    alpha=0.5,
    ax=axs[0]
    )

    sns.kdeplot(df_data_unfinished['distance2024'],
    label="2024",
    color="#ff7f0e",
    fill=True,
    alpha=0.5,
    ax=axs[0]
    )

    axs[0].set_title("Unfinished Paths")
    axs[0].set_xlabel("Number of Saved Clicks")
    axs[0].set_ylabel("Density")
    axs[0].legend(title="Wikipedia Version")

    sns.kdeplot(df_data_finished['distance2007'],
    label="2007",
    color="#1f77b4",
    fill=True,
    alpha=0.5,
    ax=axs[1]
    )

    sns.kdeplot(df_data_finished['distance2024'],
    label="2024",
    color="#ff7f0e",
    fill=True,
    alpha=0.5,
    ax=axs[1]
    )

    axs[1].set_title("Finished Paths")
    axs[1].set_xlabel("Number of Saved Clicks")
    axs[1].set_ylabel("Density")
    axs[1].legend(title="Wikipedia Version")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def compare_shortened_path_plot(df_data, title="Comparison of shortened unfinished paths in 2007 vs 2024"):
    """
    Plot comparison between the shortened paths in 2007 and 2024

    Parameters:
    - df_data: the dataframe that contains the path, target, final_link2007, final_link2024, distance2007, and distance2024
    - title: the title of the plot

    Returns:
    - None
    """

    # Categorize paths based on efficiency difference
    def categorize_efficiency(diff):
        if diff > 0:
            return '2024'
        elif diff < 0:
            return '2007'
        else:
            return 'No Change'
    
    df_data['efficiency_diff'] = df_data['distance2024'] - df_data['distance2007']
    df_data['efficiency_category'] = df_data['efficiency_diff'].apply(categorize_efficiency)

    # Aggregate data for the diverging bar chart
    category_counts = df_data['efficiency_category'].value_counts()

    # Reordering categories for consistent plotting
    categories_ordered = ['2007', 'No Change', '2024']
    category_counts = category_counts.reindex(categories_ordered)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})

    sns.barplot(
    x=category_counts.index,
    y=category_counts.values,
    palette=["#1f77b4", "#cccccc", "#ff7f0e"],  # Blue, Grey, Orange
    ax=ax[0],
    hue=category_counts.index
    )

    ax[0].set_title("Paths shortened in 2007 vs 2024")
    ax[0].set_ylabel("Number of Paths")
    ax[0].set_xlabel("Efficiency Category")
    ax[0].set_ylim(0, 18000)

    for i, v in enumerate(category_counts.values):
        ax[0].text(i, v + 50, str(v), ha='center', color='black')

    # remove distance2007 = 0 and distance2024 = 0
    df_data = df_data[(df_data['distance2007'] != 0) | (df_data['distance2024'] != 0)]

    sns.kdeplot(df_data['distance2007'],
    label="2007",
    color="#1f77b4",
    fill=True,
    alpha=0.5,
    ax=ax[1]
    )

    sns.kdeplot(df_data['distance2024'],
    label="2024",
    color="#ff7f0e",
    fill=True,
    alpha=0.5,
    ax=ax[1]
    )

    ax[1].set_title("Distribution of click saved in 2007 vs 2024")
    ax[1].set_xlabel("Number of Saved Clicks")
    ax[1].set_ylabel("Density")
    ax[1].legend(title="Wikipedia Version")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

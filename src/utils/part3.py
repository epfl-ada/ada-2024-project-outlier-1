import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import wikipediaapi # need python 3.9 or higher to work
from urllib.parse import unquote

import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer

from node2vec import Node2Vec
from gensim.models import Word2Vec

from sklearn.metrics.pairwise import cosine_similarity


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


def get_node2vec_model(Graph, DATA_PATH, model_name, load=True):
    '''
    Get the node2vec model

    Parameters:
    - Graph: the graph to train the node2vec model
    - load: whether to load the model or train the model

    Returns:
    - model: the node2vec model
    '''

    if load:
        return Word2Vec.load(os.path.join(DATA_PATH, model_name))
    
    model = Node2Vec(Graph, dimensions=64, walk_length=30, num_walks=200, workers=1)
    model = model.fit(window=10, min_count=1, batch_words=4)
    model.save(os.path.join(DATA_PATH, model_name))
    return model

def get_cosine_similarities(embeddings):
    '''
    Get the cosine similarities between the embeddings

    Parameters:
    - embeddings: the embeddings to calculate the cosine similarities

    Returns:
    - cosine_similarities: the cosine similarities
    '''

    return cosine_similarity(np.array(list(embeddings.values())))


def extract_first_paragraph(text):
    # Split the text by new lines
    lines = text.split('\n')

    # ignore the 5 first lines
    lines = lines[5:]

    # while a line don't begin by "   ", remove it
    while lines[0][:3] != "   ":
        lines = lines[1:]

    # List to accumulate lines of the first paragraph
    paragraph_lines = []
    paragraph_started = False

    for l in lines:
        if l[:3] == "   " or not l:
            paragraph_started = True
            paragraph_lines.append(l)
        elif paragraph_started:
            break

    # Join the lines to form the paragraph
    paragraph = '\n'.join(paragraph_lines)

    # return paragraph
    return paragraph

def get_summaries2007(article_path, path_2007, csv_name, load=True):

    if load:
        return pd.read_csv(os.path.join(path_2007, csv_name))
      
    summaries2007 = {'article' : [], 'content' : []}

    for filename in tqdm(os.listdir(article_path)):
        with open(os.path.join(article_path, filename), 'r', encoding='utf8') as f:
            text = f.read() 

        paragraph = extract_first_paragraph(text)
        summaries2007['article'].append(unquote(filename[:-4]))
        summaries2007['content'].append(paragraph.strip())

    summaries2007 = pd.DataFrame(summaries2007)
    summaries2007.to_csv(os.path.join(path_2007, csv_name), index=False, encoding='utf8')
    return summaries2007

def get_summaries2024(article_df, path_2024, csv_name, load=True):

    if load:
        return pd.read_csv(os.path.join(path_2024, csv_name))

    wiki_wiki = wikipediaapi.Wikipedia('ada (anasse.elboudiri@epfl.ch)', 'en')

    summaries2024 = {'article' : [], 'content' : []}

    for article in article_df['article']:
        page = wiki_wiki.page(article)
        if page.exists():
            summaries2024['article'].append(article)
            summaries2024['content'].append(page.summary)

    # article names that are different in 2007 and 2024
    old_names = ["Athletics_%28track_and_field%29",
                "Bionicle__Mask_of_Light",
                "Directdebit",
                "Newshounds",
                "Star_Wars_Episode_IV__A_New_Hope",
                "Wikipedia_Text_of_the_GNU_Free_Documentation_License",
                "X-Men__The_Last_Stand"]

    new_names = ["Track_and_field",
                "Bionicle:_Mask_of_Light",
                "Direct_debit",
                "News_Hounds",
                "Star_Wars_(film)",
                "Wikipedia:Text_of_the_GNU_Free_Documentation_License",
                "X-Men:_The_Last_Stand"]

    for old, new in zip(old_names, new_names):
        page = wiki_wiki.page(new)
        if page.exists():
            summaries2024['article'].append(unquote(old))
            summaries2024['content'].append(page.summary)
    
    summaries2024 = pd.DataFrame(summaries2024)
    summaries2024.to_csv(os.path.join(path_2024, csv_name), index=False, encoding='utf8')
    return summaries2024

def get_links_similarity(DATA, path2007, path2024, article_path_txt, articles, G_2007, G_2024, df_link2007, df_link2024, load=True):

    if load:
        return pd.read_csv(os.path.join(path2007, 'links2007_similarity.csv')), pd.read_csv(os.path.join(path2024, 'links2024_similarity.csv'))
    
    n2v_model_2007 = get_node2vec_model(G_2007, DATA, 'node2vec_2007.model')
    n2v_model_2024 = get_node2vec_model(G_2024, DATA, 'node2vec_2024.model')

    n2v_embeddings_2007 = {node: n2v_model_2007.wv[node] for node in G_2007.nodes()}
    n2v_embeddings_2024 = {node: n2v_model_2024.wv[node] for node in G_2024.nodes()}

    n2v_cosine_similarities_2007 = get_cosine_similarities(n2v_embeddings_2007)
    n2v_cosine_similarities_2024 = get_cosine_similarities(n2v_embeddings_2024)

    df_link2007['n2v_similarity'] = df_link2007.apply(lambda x: n2v_cosine_similarities_2007[list(G_2007.nodes()).index(x['linkSource']),
                                                                                     list(G_2007.nodes()).index(x['linkTarget'])], axis=1)

    df_link2024['n2v_similarity'] = df_link2024.apply(lambda x: n2v_cosine_similarities_2024[list(G_2024.nodes()).index(x['linkSource']), 
                                                                                        list(G_2024.nodes()).index(x['linkTarget'])], axis=1)
    
    summaries2007 = get_summaries2007(article_path_txt, path2007, 'summaries2007.csv')
    summaries2024 = get_summaries2024(articles, path2024, 'summaries2024.csv')

    model = SentenceTransformer('all-MiniLM-L6-v2')

    sbert_embeddings_2007 = {article: model.encode(text) for article, text in zip(summaries2007['article'], summaries2007['content'])}
    sbert_embeddings_2024 = {article: model.encode(text) for article, text in zip(summaries2024['article'], summaries2024['content'])}
    sbert_cosine_similarities_2007 = get_cosine_similarities(sbert_embeddings_2007)
    sbert_cosine_similarities_2024 = get_cosine_similarities(sbert_embeddings_2024)

    df_link2007['sbert_similarity'] = df_link2007.apply(lambda x: sbert_cosine_similarities_2007[list(sbert_embeddings_2007.keys()).index(x['linkSource']),
                                                                                        list(sbert_embeddings_2007.keys()).index(x['linkTarget'])], axis=1)

    df_link2024['sbert_similarity'] = df_link2024.apply(lambda x: sbert_cosine_similarities_2024[list(sbert_embeddings_2024.keys()).index(x['linkSource']), 
                                                                                        list(sbert_embeddings_2024.keys()).index(x['linkTarget'])], axis=1)
    
    df_link2007['similarity'] = df_link2007.apply(lambda x: np.mean([x['n2v_similarity'], x['sbert_similarity']]), axis=1)
    df_link2024['similarity'] = df_link2024.apply(lambda x: np.mean([x['n2v_similarity'], x['sbert_similarity']]), axis=1)

    df_link2007.to_csv(os.path.join(path2007, 'links2007_similarity.csv'), index=False)
    df_link2024.to_csv(os.path.join(path2024, 'links2024_similarity.csv'), index=False)

    return df_link2007, df_link2024


def plot_similarity_distributions(similarities):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    sns.histplot(similarities['n2v_similarity_2007'], ax=axs[0], color='blue', kde=True, label='2007')
    sns.histplot(similarities['n2v_similarity_2024'], ax=axs[0], color='red', kde=True, label='2024')
    axs[0].set_title('Node2Vec Similarity Distribution')
    axs[0].set_xlabel('Node2Vec Similarity')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    sns.histplot(similarities['sbert_similarity_2007'], ax=axs[1], color='blue', kde=True, label='2007')
    sns.histplot(similarities['sbert_similarity_2024'], ax=axs[1], color='red', kde=True, label='2024')
    axs[1].set_title('SBERT Similarity Distribution')
    axs[1].set_xlabel('SBERT Similarity')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    # set x and y limits
    axs[0].set_xlim(-0.15, 1)
    axs[1].set_xlim(-0.15, 1)
    plt.tight_layout()
    plt.show()


def plot_similarity_distribution(similarity):
    plt.figure(figsize=(12, 6))
    sns.histplot(similarity['similarity_2007'], kde=True, color='blue', label='2007')
    sns.histplot(similarity['similarity_2024'], kde=True, color='red', label='2024')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution')
    plt.legend()
    plt.show()
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
    '''
    Extract the first paragraph from the plain text article

    Parameters:
    - text: the plain text article (should be in the format of the articles in the wikispeedia 2007 dataset)

    Returns:
    - paragraph: the first paragraph of the article
    '''

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
    '''
    Get the summaries of the articles in the wikispeedia 2007 dataset

    Parameters:
    - article_path: the path to the plain text article text files folder
    - path_2007: the path to the folder to save the csv file
    - csv_name: the name of the csv file to save the summaries

    Returns:
    - summaries2007: the dataframe that contains the article names and the summaries
    '''
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


def get_summaries2024(article_df_names2024, path_2024, csv_name, load=True):
    """
    Get the summaries of the articles in the wikispeedia 2024 with wikipedia api

    Parameters:
    - article_df_names2024: the dataframe that contains the article old names (2007) and the new names (2024)
    - path_2024: the path to the folder to save the csv file
    - csv_name: the name of the csv file to save the summaries
    - load: whether to load the summaries or get them from wikipedia api

    Returns:
    - summaries2024: the dataframe that contains the article names and the summaries
    """

    if load:
        return pd.read_csv(os.path.join(path_2024, csv_name))

    wiki_wiki = wikipediaapi.Wikipedia('ada (anasse.elboudiri@epfl.ch)', 'en')

    summaries2024 = {'article' : [], 'content' : []}
    for old_name, new_name in article_df_names2024[["article", "new_name"]].values:
        page = wiki_wiki.page(new_name)
        if page.exists():
            summaries2024['article'].append(old_name)
            summaries2024['content'].append(page.summary)
    
    summaries2024 = pd.DataFrame(summaries2024)
    summaries2024.to_csv(os.path.join(path_2024, csv_name), index=False, encoding='utf8')
    return summaries2024


def get_links_similarity(DATA, path2007, path2024, article_path_txt, article_df_names2024, G_2007, G_2024, df_link2007, df_link2024, load_similarity=True, load=True):
    """
    Get the similarity between the links using node2vec and SBERT embeddings

    Parameters:
    - DATA: the path to the data folder
    - path2007: the path to the wikispeedia 2007 dataset folder
    - path2024: the path to the wikispeedia 2024 dataset folder
    - article_path_txt: the path to the plain text article text files folder
    - article_df_names2024: the dataframe that contains the article old names (2007) and the new names (2024)
    - G_2007: the graph of the wikispeedia 2007 dataset
    - G_2024: the graph of the wikispeedia 2024 dataset
    - df_link2007: the dataframe that contains the links of the wikispeedia 2007 dataset
    - df_link2024: the dataframe that contains the links of the wikispeedia 2024 dataset
    - load_similarity: whether to load the similarities or calculate them
    - load: whether to load the links or calculate them

    Returns:
    - df_link2007: the dataframe that contains the links of the wikispeedia 2007 dataset with the similarities
    - df_link2024: the dataframe that contains the links of the wikispeedia 2024 dataset with the similarities
    """

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
    
    summaries2007 = get_summaries2007(article_path_txt, path2007, 'summaries2007.csv', load=load_similarity)
    summaries2024 = get_summaries2024(article_df_names2024, path2024, 'summaries2024.csv', load=load_similarity)

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
    """
    Plot the similarity distributions of the node2vec and SBERT embeddings

    Parameters:
    - similarities: the dataframe that contains the similarities

    Returns:
    - None
    """
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


def plot_similarity_distribution(similarities):
    """
    Plot the similarity distribution (node2vec and SBERT embeddings)

    Parameters:
    - similarity: the dataframe that contains the similarities

    Returns:
    - None
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 2]})

    sns.histplot(similarities['similarity_2007'], kde=True, color='blue', label='2007', ax=axs[1])
    sns.histplot(similarities['similarity_2024'], kde=True, color='red', label='2024', ax=axs[1])
    axs[1].set_xlabel('Similarity')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    colors = {'similarity_2007': 'blue', 'similarity_2024': 'red'}
    sns.boxplot(data=similarities, orient='h', ax=axs[0], palette=colors)
    axs[0].set_xlabel('Similarity')

    plt.tight_layout()
    plt.suptitle('Similarity Distribution', y=1.02, fontsize=14)
    plt.show()

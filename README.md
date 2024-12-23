# Back to the Future: Time-Traveling through Wikispeedia 

### Link to our DataStory : 
https://titantek.github.io/ada-outlier-datastory/

### Abstract

After a few games of Wikispeedia, one can easily get discouraged: why is that game so difficult? Are you bad at the game or is the game too old for you? 
Our hypothesis is that those difficulties stem from the outdated structure of Wikipedia from 2007 used in the game. In this project, we thus explore how Wikipedia evolved between 2007 and 2024 and how it impacts Wikispeedia games. To answer this, we compare the structure and the hyperlinks network between the two time points and also analyze how the 2024 version of Wikipedia would impact the paths of finished and unfinished games included in the dataset. Additionally, we would like to analyze how a large language model (LLM) would perform at playing Wikispeedia and see if the changes in Wikipedia between 2007 and 2024 affect its performance. 
This study sheds light on how Wikipedia's growth influences user navigation and provides insights on how players would perform on a 2024 version of Wikispeedia.

### Research Questions:

1. What are the factors of success of a player on Wikispeedia in 2007?
2. How did the Wikipedia structure change between 2024 and 2007? Is it better linked in 2024 than in 2007? 
3. How would this different structure impact the players? Would the target article be easier to reach from the source in 2024 compared to 2007? Is this also true for the unfinished paths?
4. Does an LLM perform better on 2007 or 2024 Wikipedia? 
5. Can the behavior of a player be simulated by a LLM or some other NLP model? How to assess this resemblance? 
6. Is the current structure more intuitive than the old one? Is it more specialized and less broad? Would that help when playing the game? 

### Additionnal Datasets used: 
- Wikipedia from 2024 

This project mainly relies on comparing the performance of Wikispeedia players on two different Wikipedia versions: 2007 and 2024. Wikipedia is an open-source database which allows us to download a portion of its current version. Since Wikispeedia does not contain the entire 2007 Wikipedia but only 4604 articles, we only downloaded the corresponding 4604 articles from 2024. Our goal is to correctly process these retrieved articles to compare them in a consistent way to Wikispeedia’s database. From both articles' names and links, we could inspect the path followed by players in 2007 and compare their efficiency. We would also like to inspect the general structure of Wikipedia articles and see if the changes that occurred since 2007 could have been useful for Wikispeedia players.

### Methods:

In this project, we focus on the performances of Wikispeedia players, the changes that occurred in the structure of Wikipedia between 2007 and 2024 and the links between these two elements. These are two very different tasks that we link. We analyze the effect that the structure could have on the players' performance. We thus decided to split our approach as follow:

1. Exploration and analysis of players’ performances in 2007

    By first looking at the performance of players in Wikispeedia 2007, we determine potential factors and correlations between the games played and the data structure, to better understand patterns of performance. For that we:

    - Compare some properties of the paths: the categories of the source and target articles, the number of links leading to the target, and the shortest path. 

    - Compare the distribution of these properties depending on the success of the games using appropriate statistical tests such as chi2 test and t-test.

    - Predict if a game will be won depending on the categories of the source and target articles, the number of links leading to the target, and the shortest path. Logistic regression is used.

2. Comparison of Wikipedia’s structure between 2007 and 2024

    To compare both databases we want to have the best equivalence possible between the two years. To obtain 2024 Wikipedia, we have downloaded current articles based on their past URL. This method has its own limits, so before analyzing the results we:

    - Identify URLs that no longer exist nowadays

    - Find and correct ambiguous and homonymous pages


    Once the data is fully and correctly retrieved, with the list of articles and links of 2024 exactly matching the list from 2007: 

    - Analyze the difference in structure between the two years: comparison of the number of links per page and the average number of links.

    - Compute and compare shortest path matrix using Floyd Warshall Algorithm: comparison of the average shortest path between the two years and comparison of the average shortest path across strongly connected components.

    - Assess Network differences: comparison of the average clustering coefficients, computation of the pagerank centrality of both networks, comparision of pageranks value for the most central nodes, comparison of the top 20 central nodes and comparison of the average node degree and reachability of the networks. 

3. Consequences of the changes of Wikipedia in player’s performances

    We explore how the structural changes of Wikipedia between 2007 and 2024 impact players’ decisions and paths in Wikispeedia. Specifically, we examine whether these changes make it easier to win games and evaluate the efficiency of the updated structure.

    - Player Path Analysis

        - assess if unfinished paths from 2007 could now be completed using the 2024 structure

        - check if new links from 2024 structure allow us to finish 2007 paths faster

    - Structural Comparison: 2007 vs. 2024

        - use Node2Vec to measure graph-similarity
        - use Sentence-BERT to measure semantic similarity in the first paragraph of the articles
        - evaluate which structure offers better navigation efficiency using these metrics, comparing their distributions of the metrics on the 2007 and 2024 datasets 

4. Consequences on LLMs performance

    We use Mistral and Llama3 models to simulate players' performance on Wikispeedia. We then compare their performance to the one of the players in 2007. To do so, we:

    - determine which games (start and target articles) we choose to play with the LLMs plotting the CCDF of played games
    - determine the maximum number of steps the LLMs can take
    - make the LLMs play the games by first giving the context of the game with an example of path with a reasoning for the choice of the path 
    - let the LLMs play the game by giving the start article and the options of links to follow.
    - analyze how many games Llama3 and Mistral successfully completed
    - analyze the path length distribution of the games played by the LLMs and the path length distribution of the players
    - compare the similarity of the paths found by the LLMs and the players with the Jaccard similarity
  
    Then based on the results of the comparison, we choose the model that mimics the players the best and play the games in 2024. We then compare the performance of the LLMs in 2024 to the one in 2007 by:

    - comparing the number of paths found by the LLMs in 2024 and 2007
    - comparing the path length distribution of the games played by the LLMs in 2024 and 2007


### Main Results:

1. Factors of success:

    Some categories have a positive or negative influence on the success rate. A longer shortest path decreases slightly the odds of success whereas the number of links to target strongly increases it. The model is not fully satisfying and could be improved if we had more data about the players and/or the games played.

2. Wikipedia structure evolution: 

    From our analysis, we observe that the structure of Wikipedia has evolved a lot in 2024. Many links are created which results in a better connectivity of the graph overall and a shorter average shortest path. We see that the network is less dominated by one very central node but has a more equally distributed centrality of main nodes. However, the overall structure of the two networks is the same, with one big cluster being strongly connected and containing the majority of the articles. Even though both graphs are very different, it is hard to infer how this difference would impact the players in their game performance and we perform further analysis in part 3 to study this. 

3. Changes in players' paths and performances : 

    We assess how the structural changes in Wikipedia impact the paths played in the 2007 dataset. The target link appears sooner in 2024 than in 2007 on the paths played, both for finished and unfinished paths. We use the similarity measure to compare the structures and see that the similarity between articles is higher for 2024 than for 2007. This shows that the 2024 network might be more precise and intuitive to navigate. 

4. LLMs performance: 

    From the results of the LLMs on 2007 Wikipedia, we observed that llama3 mimic better the players than Mistral. We then played the games in 2024 with llama3 and observed that the performance of the LLMs is better in 2024 than in 2007. This is due to the fact that the structure of Wikipedia has evolved and is now more connected.

Our analysis leads us to think that the players would be more performant on a Wikispeedia 2024 version. However, we would need some data from real players on the 2024 version to confirm it firmly.


### Group Contributions: 

Gabrielle Blouvac (AKA: Our Official Graphic Designer): part 2, created images for the datastory

Anasse El Boudiri (AKA: The Pie-Chart and Iframe Hater): part 3, bonus: convinced the team to choose this dataset <3

Julia Guignon (AKA: The Plotly Toxic Lover): part 1

Jan Steiner (AKA: The LLM genius): part 4, bonus: saver of the website content bar

Eglantine Vialaneix (AKA: The Scrapping Expert): scrapping expert of 2024 Wikipedia, preprocessing of 2024 data, part 2.0

All: READ_ME, proof-reading, layout of the website (html+css), homeworks

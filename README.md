# Back to the Future : Time-Traveling through Wikispeedia 

### Abstract

After a few games of Wikispeedia, one can easily get discouraged: why is that game so difficult? Are you bad at the game or is the game too old for you? 
Our hypothesis is that those difficulties stem from the outdated structure of Wikipedia from 2007 used in the game. In this project, we thus explore how Wikipedia evolved between 2007 and 2024 and how it impacts Wikispeedia games. To answer this, we compare the structure and the hyperlinks network between the two time points and also analyze how the 2024 version of Wikipedia would impact the paths of finished and unfinished games included in the dataset. Additionally, we would like to analyze how a large language model (LLM) would perform at playing Wikispeedia and see if the changes in Wikipedia between 2007 and 2024 affect its performance. 
This study sheds light on how Wikipedia's growth influences user navigation and provides insights on how players would perform on a 2024 version of Wikispeedia.

### Research Questions :

1. What are the factors of success of a player on Wikispeedia in 2007?
2. How did the Wikipedia structure change between 2024 and 2007? Is it better linked in 2024 than in 2007? 
3. How would this different structure impact the players? Would the target article be easier to reach from the source in 2024 compared to 2007? Is this also true for the unfinished paths?
4. Does an LLM perform better on 2007 or 2024 Wikipedia? 
5. Can the behavior of a player be simulated by a LLM or some other NLP model? How to assess this resemblance? 
6. Is the current structure more intuitive than the old one? Is it more specialized and less broad? Would that help when playing the game? 

### Additionnal Datasets used : 
- Wikipedia from 2024 

This project mainly relies on comparing the performance of Wikispeedia players on two different Wikipedia versions: 2007 and 2024. Wikipedia is an open-source database which allows us to download a portion of its current version. Since Wikispeedia does not contain the entire 2007 Wikipedia but only 4604 articles, we only downloaded the corresponding 4604 articles from 2024. Our goal is to correctly process these retrieved articles to compare them in a consistent way to Wikispeedia’s database. From both articles' names and links, we could inspect the path followed by players in 2007 and compare their efficiency. We would also like to inspect the general structure of Wikipedia articles and see if the changes that occurred since 2007 could have been useful for Wikispeedia players.

### Methods :

In this project, we focus on the performances of Wikispeedia players, the changes that occurred in the structure of Wikipedia between 2007 and 2024 and the links between these two elements. These are two very different tasks that we however want to link and would like to analyze the effect the latter could have had on the former. We thus decided to split our approach as follow :

1. Exploration and analysis of players’ performances in 2007

    By first looking at the performance of players in Wikispeedia 2007, we would like to determine potential factors and correlations between the games played and the data structure, to better understand patterns of performance. For that we will:

    - Compare properties of the source and target articles and paths such as categories, number of links leading to the target, shortest path, etc. 

    - Compare the distribution of these properties depending on the success of the games. For this, we will use appropriate statistical tests such as the ANOVA test.

    - Try to predict the success rate of a game depending on the properties of the source and target articles, using regression.

2. Comparison of Wikipedia’s structure between 2007 and 2024

    To compare both databases we would like to have the best equivalence possible between the two years. To obtain 2024 Wikipedia, we are downloading nowadays articles based on their past URL. This method has its own limits, so before analyzing the results we need to :

    - Identify URLs that disappeared since 2007

    - Find and correct ambiguous and homonymous pages


    Once the data is fully and correctly retrieved, with the list of articles and links of 2024 exactly matching the list from 2007: 

    - Analyze difference in structure between the two years : comparison of the number of links per pages, average number of links.

    - Compute and compare shortest path matrix using Floyd Warshall Algorithm : comparison of the average shortest path between the two years and comparison of the average shortest path across strongly connected components.

    - Assess Network differences : comparison of the average clustering coefficients, computation of the pagerank centrality of both networks, comparision of pageranks value for the most central nodes, comparison of the top 20 central nodes and comparison of the average node degree and reachability of the networks. 

3. Consequences of Wikipedia’s changes in player’s performances

    We aim to explore how Wikipedia's structural changes from 2007 to 2024 would impact players’ decisions and paths in Wikispeedia. Specifically, we will examine whether these changes make it easier to complete paths and evaluate the efficiency of the updated structure.

    - Player Path Analysis

        - Assess if unfinished paths from 2007 could now be completed faster using the 2024 structure

        - Compare finished 2007 paths to potential routes in 2024 for efficiency improvements

    - Structural Comparison : 2007 vs. 2024

        - Use Node2Vec and Sentence-BERT to measure and compare structural similarities

        - Evaluate which structure offers better navigation efficiency using these metrics.

4. Consequences on LLMs performance

    By prompting and launching different LLMs models (Mistral and Llama) to play the game in both years, we would like to compare their performance. We need to:

    - Set a precise and adequate metric to quantify the performance of the models. 

    - Prompt engineer the queries to the models.

    - Analyze the results and compare the differences in performance between both years, if any.

    We would also like to try to mimic Wikispeedia gamers behavior to infer how humans would have performed on 2024 Wikispeedia. This task is not guaranteed to be feasible and will probably take a great amount of time. We reserve this idea in case of motivation and time resources.

### Main Results :

1.

2. Wikipedia structure evolution : 

    From our analysis, we observe that the structure of Wikipedia has evolved a lot in 2024. Many links are created which results in a better connectivity of the graph overall and a shorter average shortest path. We see that the network is less dominated by one very central node but has a more equally distributed centrality of main nodes. However the overall structure of the two networks is the same, with one big cluster being strongly connected and containing the majority of the articles. Even though both graphs are very different, it is hard to infer how this difference would impact the players in their game performance and we perform further analysis in part 3 to study this. 

3. 

4. 


### Group Contributions : 

Julia Guignon : part 1 
Anasse El Boudiri : part 3
Jan Steiner : part 4
Eglantine Vialaneix : scrapping of 2024 Wikipedia, preprocessing of 2024 data, part 2.0 
Gabrielle Blouvac : part 2, created images for the datastory 

# Back to the Future: Time Travelling through Wikispeedia
This is a template repo for your project to help you organise and document your code better. 
Please use this structure for your project and document the installation, usage and structure as below.

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>

# install requirements
pip install -r pip_requirements.txt
```

### How to use the library

Tell us how the code is arranged, any explanations goes here.

#### Ollama library setup

1) Install [Docker](https://docs.docker.com/get-started/get-docker/)
2) Follow instruction to use [Ollama container](https://hub.docker.com/r/ollama/ollama) on CPU/GPU
3) Run model locally : `docker exec -it <container name> ollama run <model>`
   e.g. (`docker exec -it ollama ollama run mistral`)


## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

NB : Please add the `plaintext_articles` folder from original Wikispeedia dataset to the `data/2007/plain_text_articles/` directory.

import os
import pandas as pd

DATA = "data/"
DATA_LLAMA = DATA + "llama3/"
DATA_MISTRAL = DATA + "mistral/"
YEAR = 2024

# Load the data
files = os.listdir(DATA_LLAMA+f"{YEAR}/")
df_llama = pd.DataFrame()
for file in files:
    df_llama = pd.concat([df_llama, pd.read_csv(DATA_LLAMA+f"{YEAR}/"+file)], axis=0)
df_llama = df_llama.reset_index(drop=True)
df_llama.to_csv(DATA_LLAMA+f"llm_paths{YEAR}.csv", index=False)

# files = os.listdir(DATA_MISTRAL+f"{YEAR}/")
# df_mistral = pd.DataFrame()
# for file in files:
#     df_mistral = pd.concat([df_mistral, pd.read_csv(DATA_MISTRAL+f"{YEAR}/"+file)], axis=0)
# df_mistral = df_mistral.reset_index(drop=True)
# df_mistral.to_csv(DATA_MISTRAL+f"llm_paths.csv", index=False)



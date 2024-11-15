import lmql 
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import os
import asyncio as asyncio

# lmql.serve("Llama-2-13b-hf", cuda=True, port=9999, trust_remote_code=True)

# Specify the path where the model and tokenizer are saved
load_directory = "./local_models/meta-llama/openai-community/gpt2"  # Your local path

# Load the model and tokenizer from the specified directory
# model = AutoModel.from_pretrained(load_directory, trust_remote_code=True)
# print(type(model))
        
    
                
lmql.serve(load_directory, port=8079, cuda=True, trust_remote_code=True)
@lmql.query(
                model=load_directory,
                decoder="sample",
                temperature=0.5,
                top_k=10,
                max_len=4096
)
async def prompt():
    '''lqml
    "Say 'this is a test':[ANSWER]"
    '''

async def main():
    global result
    result = await prompt()
    print("I'm in the main")

async def run_main():
    print("I'm in run_main")
    await main()

asyncio.run(run_main())
print(result)

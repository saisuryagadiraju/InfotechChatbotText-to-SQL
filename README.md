# InfotechChatbotText-to-SQL


## InfoBridge Chatbot Assistant

A conversational AI assistant that integrates **Retrieval-Augmented Generation (RAG)** and **Text-to-SQL** capabilities to enable natural language querying over the **FHWA InfoBridge datasets**, powered by **Meta-LLaMA 3.1 8B Instruct (GGUF)** with **GPU acceleration via `llama-cpp-python`**.

---

## Project Summary

 Project Summary
This chatbot system supports two modes of interaction:

Semantic Search (RAG): Answers open-ended user questions about bridge technologies using a pre-scraped knowledge base.

Text-to-SQL (Natural Language to SQL): Converts structured user queries into SQL and fetches answers directly from a SQLite database.

Powered by LLaMA 3.1 8B: Using llama-cpp-python for fully local, GPU-accelerated inference (no OpenAI API required).

Flask-based Interface: Lightweight web UI using HTML and JavaScript.

## Project Structure
```
├── test.py                       # Main backend Flask app (entry point)
├── templates/
│   └── index.html                # Frontend chatbot interface
├── InfoBridge_scraped_data.json # RAG knowledge base
├── TextToSQL/
│   ├── sql.py                   # Text-to-SQL logic (invoked on * queries)
│   ├── bridges.db               # SQLite database of bridge info
│   └── *.csv                    # Raw structured datasets (used to build DB)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

---

##  Setup Instructions



### 1. Clone the Repository
```bash
cd InfoBridge-Chatbot
```

## 2. Create Environment and Install Dependencies
````
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
````
3. Required Model: LLaMA 3.1 8B Instruct (GGUF)
The chatbot uses the quantized Q4_K_M version from HuggingFace:
```
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)
```
Cluster and GPU Setup (Hopper ORC Cluster @ GMU)
This chatbot is optimized to run on GPU-enabled VMs from GMU's Hopper cluster. Follow the instructions below to ensure proper CUDA and module setup.
Required Environment (Tested on Hopper VM)
![image](https://github.com/user-attachments/assets/a496c71d-3edf-4f5a-b293-e3dad119927e)

4. Module Load Instructions
SSH into your Hopper VM and run:
```
module load cuda/12.6.3
module list
```
5.  Key CUDA Settings (optional but recommended)
Add these to your .bashrc or set before launching the chatbot:


```
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export GGML_CUDA_FORCE_MMQ=1   # Optional: optimize memory for GGUF

```

6.  llama-cpp-python with GPU
The LLaMA 3.1 8B model is loaded using llama-cpp-python, which allows running the entire quantized model on the GPU (no OpenAI API required).
Example usage in your test.py:
````
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=-1,       # Offload all layers to GPU
    batch_size=1024,
    f16_kv=True,
    flash_attn=True
)
````
Ensure your version of llama-cpp-python is compiled with CUDA:
```
pip install llama-cpp-python --upgrade

CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --force-reinstall llama-cpp-python --no-cache-dir

```
Launch the the model Flask App
```
python test.py
````

It will start the chatbot on:


Interaction Flow
![image](https://github.com/user-attachments/assets/0ac23ead-6d81-4d37-be93-d9f162b61b28)


#### Requirements
```
flask
llama-cpp-python
sentence-transformers
spacy
huggingface-hub
pandas
numpy
sqlite3
transformers
```
#### Install with:
```
pip install -r requirements.txt
```

Overall Note:
```
Text-to-SQL functionality is triggered with an asterisk (*) prefix.

Database (bridges.db) is built using CSVs in the TextToSQL/ folder.

Backend (test.py) and frontend (index.html) are tightly integrated.
````

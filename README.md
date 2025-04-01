# InfotechChatbotText-to-SQL
### Overview
InfoBridge Chatbot Assistant is a GPU-accelerated conversational system that enables natural language interaction with bridge infrastructure datasets. The system integrates:
Retrieval-Augmented Generation (RAG) for open-ended semantic questions.
Text-to-SQL translation for structured querying via SQLite.
A fully local large language model: Meta-LLaMA 3.1 8B Instruct (GGUF format), powered by llama-cpp-python with full CUDA GPU acceleration.


### Key Features
Semantic Search (RAG) over scraped FHWA InfoBridge content.

Natural Language to SQL conversion for structured data queries.

Fast, local inference using llama-cpp-python on Hopper ORC GPUs.

Interactive Web UI via Flask frontend.

### Project Structure
```
├── test.py                       # Main backend Flask app (entry point)
├── templates/
│   └── index.html                # Frontend chatbot interface
├── InfoBridge_scraped_data.json # RAG knowledge base
├── TextToSQL/
│   ├── sql.py                    # Text-to-SQL logic (invoked on * queries)
│   ├── bridges.db                # SQLite database
│   └── *.csv                     # Raw structured datasets (loaded into DB)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### Setup Instructions
1. Clone the Repository
```
git clone https://github.com/<your-repo>/InfotechChatbotText-to-SQL.git
cd InfotechChatbotText-to-SQL
```
2. Create Python Environment & Install Dependencies
````
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
````


3. LLaMA 3.1 Model Setup (GGUF)

```
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)
```
4. GPU + CUDA Configuration
(Hopper ORC Cluster @ GMU)--optional -- Visula Studio Instance was used for this project to run the codes and comands

*Load Modules 
```
module load cuda/12.6.3
module list
```

Optional (recommended for GGUF performance):
```
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export GGML_CUDA_FORCE_MMQ=1
```
5. GPU-Accelerated LLaMA Inference via llama-cpp-python

   Ensure llama-cpp-python is compiled with CUDA:
```
pip install llama-cpp-python --upgrade
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 \
pip install --force-reinstall llama-cpp-python --no-cache-dir
```
6. Launch the Chatbot
````
python test.py
````

This will start the backend Flask app on: 
Note: Port no may vary based on the configuration

```
http://localhost:6150/
```

#### RAG Model Interaction Instructions
General Q&A (RAG Mode)
Ask open-ended questions like:

“What is Hammer Sounding?”
“How to do Drilling.”

The chatbot responds using semantic search from InfoBridge_scraped_data.json.

#### Text-to-SQL Mode

Prefix queries with an asterisk * to invoke structured SQL-based responses.

Examples:
* Show the top 3 bridges with highest average daily traffic.
* Find bridges built before 1970 in Fairfax County.

The chatbot will generate SQL, run it against bridges.db, and return the results.

### Demo Summary (for Reviewers)


![image](https://github.com/user-attachments/assets/740aad43-f5fe-4c71-8517-788bdb3e255c)

### Required Python Packages
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
Install via:
```
pip install -r requirements.txt
```
#### Note

The chatbot uses only local inference. No OpenAI or cloud API is required.

CSV files under TextToSQL/ are used to construct the SQLite database.

No need to run sql.py separately — it’s integrated into test.py.

Conversations and SQL responses are dynamically handled in real-time.

####  How to Reproduce the Demo
```
Follow setup steps above

Run: python test.py

Navigate to: http://localhost:6150 -- new webpage wil open

Start chatting — try open-ended and *sql questions

All data, models, and responses remain local (no internet required post-download)
````



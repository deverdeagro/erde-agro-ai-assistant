# ERDE Agro AI Assistant

A local AI-powered company chatbot built using:

- Ollama (Local LLM)
- LangChain
- ChromaDB
- RAG (Retrieval Augmented Generation)

The goal is to create an internal/company assistant that can answer questions about:

- Company information
- Products
- Services
- FAQs
- Customer queries

Future versions will add:

- LangGraph agent workflows
- WhatsApp integration
- Lead collection
- Human escalation
- CRM integration


## Architecture

```
User
 |
 v
Chat Interface
 |
 v
LangChain
 |
 +----------------+
 |                |
 v                v

ChromaDB        Ollama
Vector DB       LLM
 |
 |
Company Documents
(PDF/TXT)
```


## Tech Stack

### AI

- Ollama
- Llama 3.1 8B

### Frameworks

- LangChain
- LangGraph (planned)

### Vector Database

- ChromaDB

### Backend (planned)

- FastAPI

### Integration (planned)

- WhatsApp Cloud API


## Project Structure

```
erde-agro-ai-assistant/

│
├── app/
│
│   ├── llm.py
│   │   Connects LangChain with Ollama
│   │
│   ├── rag.py
│   │   Loads company documents
│   │
│   ├── vectorstore.py
│   │   Creates Chroma vector database
│   │
│   ├── create_db.py
│   │   Generates embeddings and stores them
│   │
│   ├── chat.py
│   │   Runs chatbot
│
│
├── data/
│
│   ├── company documents
│   ├── PDFs
│   └── text files
│
│
├── chroma_db/
│   Vector database storage
│
│
├── requirements.txt
│
└── README.md
```


## Setup

### 1. Clone project

```bash
git clone <repo-url>

cd erde-agro-ai-assistant
```


## 2. Create virtual environment

Mac/Linux:

```bash
python -m venv .venv

source .venv/bin/activate
```


Windows:

```bash
python -m venv .venv

.venv\Scripts\activate
```


## 3. Install dependencies

```bash
pip install -r requirements.txt
```


## 4. Install Ollama

Download Ollama:

https://ollama.com


Check:

```bash
ollama --version
```


## 5. Pull models

LLM:

```bash
ollama pull llama3.1:8b
```

Embedding model:

```bash
ollama pull nomic-embed-text
```


Check installed models:

```bash
ollama ls
```


Expected:

```
llama3.1:8b

nomic-embed-text
```


## Running the project


### Step 1

Add company documents:

```
data/

company_profile.txt

products.txt

faq.txt

company.pdf
```


### Step 2

Create vector database:

```bash
python app/create_db.py
```


This will:

- Load documents
- Split into chunks
- Create embeddings
- Store in ChromaDB


### Step 3

Start chatbot:

```bash
python app/chat.py
```


Example:


```
You:
What does ERDE Agro do?


AI:
ERDE Agro is an agriculture technology company...
```


## How RAG works here

```
Question

    |
    v

Search ChromaDB

    |
    v

Find relevant company information

    |
    v

Send context + question to LLM

    |
    v

Generate answer
```


## Current Features

✅ Local LLM  
✅ No API cost  
✅ Company document knowledge  
✅ PDF support  
✅ TXT support  
✅ Vector search  
✅ RAG pipeline  


## Planned Features


### LangGraph Agent System

```
User

 |

Router Agent

 |

+----------------+
|                |
FAQ Agent     Sales Agent

 |

Response
```


### WhatsApp Integration

```
Customer

 |

WhatsApp API

 |

FastAPI

 |

LangGraph

 |

Ollama

 |

Response
```


### Future Agents

- Product agent
- Sales agent
- Support agent
- Lead qualification agent
- Analytics agent


## Development Notes

Keep documents updated.

When documents change:

Delete old database:

```bash
rm -rf chroma_db
```

Recreate:

```bash
python app/create_db.py
```


## License

Internal project.
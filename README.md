# WoW Classes RAG

A full-stack application that uses RAG (Retrieval-Augmented Generation) to answer questions about World of Warcraft classes.

### Covered Classes
- Blood, Frost and Unholy Death Knight
- Havoc and Vengeance Demon Hunter
- Windwalker, Mistweaver and Brewmaster Monk
- Protection Warrior
- Protection Paladin
- Guardian and Restoration Druid
- Holy and Discipline Priest
- Restoration Shaman
- Preservation Evoker


## Project Structure

```
wow-classes-rag/
├── api/                          # Backend (Python/FastAPI)
│   ├── classes/                  # Scraped class data files
│   ├── controller.py             # FastAPI routes/endpoints
│   ├── service.py                # Business logic and RAG implementation
│   ├── utils.py                  # Utility functions (web scraping, embeddings)
│   ├── classes.json              # Class configuration
│   ├── extraction_query.txt      # LLM prompt for content extraction
│   ├── main_query.txt            # LLM prompt for answering queries
│   ├── .env                      # Environment variables (API keys)
│   ├── .gitignore                # Backend-specific gitignore
│   ├── requirements.txt          # Python dependencies
│   ├── run_server.sh             # Backend startup script
│   └── milvus.db                 # Vector database (generated)
├── frontend/                     # Frontend (React) - to be created
├── run_server.sh                 # Backend startup script (root level)
├── requirements.txt              # Dependency reference
├── .gitignore                    # Root gitignore
└── README.md                     # This file
```

## Backend Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Install backend dependencies:
```bash
pip install -r api/requirements.txt
```

2. Create a `.env` file in the `api/` directory with your API keys:
```bash
cd api
cp .env.example .env  # If you have an example file
# Edit .env and add your FUELIX_API_KEY
```

### Running the Backend

From the project root:
```bash
./run_server.sh
```

Or from the api directory:
```bash
cd api
./run_server.sh
```

The backend will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Health check
- `GET /retrieve-answers?query_text=<query>` - Ask questions about WoW classes
- `GET /generate-knowledge-base` - Scrape and generate knowledge base from Icy Veins
- `GET /generate-embedding-vector` - Generate vector embeddings for the knowledge base

### Backend Workflow

1. **Generate Knowledge Base**: Scrapes WoW class information from Icy Veins
   ```bash
   curl http://localhost:8000/generate-knowledge-base
   ```

2. **Generate Embeddings**: Creates vector embeddings from the scraped content
   ```bash
   curl http://localhost:8000/generate-embedding-vector
   ```

3. **Query**: Ask questions about WoW classes
   ```bash
   curl "http://localhost:8000/retrieve-answers?query_text=How%20do%20I%20play%20a%20fire%20mage?"
   ```

## Frontend Setup (Coming Soon)

The frontend is a React application that provides a user-friendly interface for querying the RAG system.

### Prerequisites
- Node.js 16+
- npm or yarn

### Installation (when frontend is created)
```bash
cd frontend
npm install
```

### Running the Frontend (when created)
```bash
cd frontend
npm start
```

The frontend will be available at `http://localhost:3000`

## Technology Stack

### Backend
- **FastAPI**: Web framework
- **Milvus**: Vector database for embeddings
- **BeautifulSoup4**: Web scraping
- **python-dotenv**: Environment variable management
- **Requests**: HTTP client
- **Uvicorn**: ASGI server

### Frontend (Planned)
- **React**: UI framework
- **Axios**: HTTP client for API calls
- **React Router**: Navigation
- **CSS**: Styling (TBD)

## Development

### Environment Variables

The backend requires the following environment variables in `api/.env`:
- `FUELIX_API_KEY`: API key for LLM and embeddings
- `FUELIX_COMPLETIONS_URL`: (Optional) LLM completions endpoint
- `FUELIX_EMBEDDINGS_URL`: (Optional) Embeddings endpoint

### Code Organization

- **controller.py**: Defines API routes and request/response handling
- **service.py**: Contains business logic for RAG operations
- **utils.py**: Helper functions for web scraping and embeddings
- Import handling is designed to work both when run as a module and standalone

## License

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.html&hl=pt&sl=en&tl=pt&client=srp)

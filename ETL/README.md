
## ETL System Documentation
### Core Components
### Data Sources
 Crawls financial statements from CafeF portal for Vietnamese companies

 Handles 3 types of companies: Banks, Non-Banks, and Securities firms
 Data includes balance sheets, income statements, and cash flow statements

### Main Classes
- ETL_Args: Configuration class containing stock code lists for different company types
- SimpleETL: Main ETL orchestrator using CafeF crawler
- CafeFCrawlerFS: Financial statement crawler implementation
- DBHUB: Database hub for querying and searching financial data
- Mapping Files: Located in crawler/mapping_data/:

### Key Features
Data Crawling:

- Supports quarterly and annual reports
- Handles Vietnamese and English captions
- Maps raw data to standardized category codes

### Data Processing:

- Standardizes financial numbers
- Handles different company types separately
- Maps Vietnamese captions to English equivalents

### Database Integration:

- Supports Chroma vector database
- Enables semantic search capabilities
- Handles SQL queries for financial data

### File Structure

```
ETL/
├── crawler/
│   ├── mapping_data/    # Category code mappings
│   ├── base.py          # Abstract crawler classes
│   └── cafef_crawler.py # CafeF implementation
│
├── dbmanager/
│   ├── abstract_semantic_layer.py
│   ├── abstracthub.py          
│   ├── hub_horizontal.py    
│   ├── hub_vertical.py     
│   ├── mongodb.py    
│   ├── rerank.py       # Reranker module
│   └── setup.py        # Setup DBHUB and Semantic Layers
│
├── const.py            # Params for financial ratio calculation
├── ratio_index.py      # Calculate financial ratio  
├── connector.py        # Database connectivity
├── etl.py              # Main ETL orchestration
└── craw_data.ipynb     # Crawling examples
```

The system is designed to extract, transform and load Vietnamese financial statement data into a standardized format suitable for analysis and querying.

## Database Manager Documentation
### Core Components
#### BaseDBHUB
The base abstract class that defines the core database functionality:

```
class BaseDBHUB(BaseModel):
    conn: SkipValidation            # Database connection
    vector_db_company: Chroma       # Vector DB for company data 
    vector_db_sql: Chroma          # Vector DB for SQL queries
    multi_threading: bool = False   # Multi-threading support flag
```

### Implementations
### Vertical Database Structure
`HubVerticalBase` and `HubVerticalUniversal`

`HubHorizontalBase` and `HubHorizontalUniversal` 

### Key Features
#### Vector Search
- Company name similarity search
- Financial statement category search
- SQL query similarity search

#### Multi-threading Support
- Parallel processing for vector searches
- Thread pool executor for batch operations
- Performance optimization for large datasets

#### Database Operations
- PostgreSQL connection management
- Query execution and result formatting
- Data type handling and validation

### Configuration
Database setup can be customized through DBConfig:

```
class DBConfig(BaseModel):
    embedding: Union[str, HuggingFaceEmbeddings]  # Embedding model
    database_choice: str                          # DB structure type
```
The default configurations support both OpenAI and local embedding models.

### Initialize database hub
```
db = setup_db(OPENAI_VERTICAL_BASE_CONFIG)
```

### Search for company information
```
results = db.search_return_df("total assets", top_k=5)
```
This database management system provides a flexible and performant way to interact with financial report data using vector similarity search and SQL queries.
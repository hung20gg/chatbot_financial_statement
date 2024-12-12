The main purpose of the `agent` folder is to handle the conversion of natural language financial queries into structured SQL queries and to manage the interaction with various financial databases. It includes scripts for defining constants, reading and processing prompt files, and providing utility functions for data manipulation.

## Main Purpose of the `agent` Folder

* **Convert Natural Language to SQL** : The folder contains scripts that convert user queries into SQL queries to retrieve relevant financial data.
* **Manage Prompts and Templates** : It includes prompt files and templates that guide the breakdown of complex financial questions into simpler, actionable steps.
* **Utility Functions** : Provides utility functions for data manipulation and conversion, aiding in the processing and formatting of financial data.
* **Database Interaction** : Manages the interaction with various financial databases to fetch and process the required data for analysis.



### Main class [Chatbot] that handles:

* User interaction and conversation management
* Query routing to determine if financial analysis is needed
* Integration with database and Text2SQL components
* Conversation history tracking
* Response generation and formatting


### Purpose of [text2sql.py]

Main class for converting natural language financial queries into executable SQL queries through a multi-step process.

1. **Query Processing** :

* Takes natural language financial queries
* Breaks them down into logical steps
* Converts steps into SQL queries

2. **Integration** :

* Connects with database manager (BaseDBHUB)
* Interfaces with LLM models
* Uses prompt templates for consistent query generation

3. **Error Handling** :

* Logging setup for debugging
* Validation using pydantic models

This file acts as the core converter between natural language financial queries and executable SQL, serving as a critical component in the financial chatbot system.

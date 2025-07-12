import boto3  # AWS SDK for Python
from langchain_community.document_loaders import JSONLoader  # Utility to load JSON files
#from langchain.llms import Bedrock  # Large Language Model (LLM) from Anthropic
from langchain_community.chat_models import BedrockChat  # Chat interface for Bedrock LLM
from langchain.embeddings import BedrockEmbeddings  # Embeddings for Titan model
from langchain.memory import ConversationBufferWindowMemory  # Memory to store chat conversations
from langchain.indexes import VectorstoreIndexCreator  # Create vector indexes
from langchain.vectorstores import FAISS  # Vector store using FAISS library
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split text into chunks
from langchain.chains import ConversationalRetrievalChain  # Conversational retrieval chain
from langchain.callbacks.manager import CallbackManager

# Create a Boto3 client for Bedrock Runtime
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime"#,
    #region_name="us-east-1"
)

# Function to get the LLM (Large Language Model)
def get_llm():
    model_kwargs = {  # Configuration for Anthropic model
        "max_tokens": 512,  # Maximum number of tokens to generate
        "temperature": 0.2,  # Sampling temperature for controlling randomness
        "top_k": 250,  # Consider the top k tokens for sampling
        "top_p": 1,  # Consider the top p probability tokens for sampling
        "stop_sequences": ["\n\nHuman:"]  # Stop sequence for generation
    }
    # Create a callback manager with a default callback handler
    callback_manager = CallbackManager([])
    
    llm = BedrockChat(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Set the foundation model
        #model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        #model_id="meta.llama3-70b-instruct-v1:0",
        #model_kwargs=model_kwargs,  # Pass the configuration to the model
        callback_manager=callback_manager
        
    )

    return llm

# Function to load the schema file based on the schema type
def load_schema_file(schema_type):
    if schema_type == 'Schema_Type_A':
        schema_file = "Table_Schema_A.json"  # Path to Schema Type A
    elif schema_type == 'Schema_Type_B':
        schema_file = "Table_Schema_B.json"  # Path to Schema Type B
    elif schema_type == 'Schema_Type_C':
        schema_file = "Table_Schema_C.json"  # Path to Schema Type C
    return schema_file

# Function to get the vector index for the given schema type
def get_index(schema_type):
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",
                                   client=bedrock_runtime)  # Initialize embeddings

    db_schema_loader = JSONLoader(
        file_path=load_schema_file(schema_type),  # Load the schema file
        jq_schema='.',  # Select the entire JSON content
        text_content=False)  # Treat the content as text

    db_schema_text_splitter = RecursiveCharacterTextSplitter(  # Create a text splitter
        separators=["separator"],  # Split chunks at the "separator" string
        chunk_size=10000,  # Divide into 10,000-character chunks
        chunk_overlap=100  # Allow 100 characters to overlap with previous chunk
    )

    db_schema_index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,  # Use FAISS vector store
        embedding=embeddings,  # Use the initialized embeddings
        text_splitter=db_schema_text_splitter  # Use the text splitter
    )

    db_index_from_loader = db_schema_index_creator.from_loaders([db_schema_loader])  # Create index from loader

    return db_index_from_loader

# Function to get the memory for storing chat conversations
def get_memory():
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)  # Create memory

    return memory

# Template for the question prompt
template = """ Read table information from the context. Each table contains the following information:
- Name: The name of the table
- Description: A brief description of the table
- Columns: The columns of the table, listed under the 'columns' key. Each column contains:
  - Name: The name of the column
  - Description: A brief description of the column
  - Type: The data type of the column
  - Synonyms: Optional synonyms for the column name
- Sample Queries: Optional sample queries for the table, listed under the 'sample_data' key

Given this structure, Write a SQL query using Cloudera syntax that would retrieve the data for following question. The produced query should be functional, efficient, and adhere to best practices in SQL query optimization.


Question: {}
"""


# Function to get the response from the conversational retrieval chain
def get_rag_chat_response(input_text, memory, index):
    llm = get_llm()  # Get the LLM

    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(
        llm, index.vectorstore.as_retriever(), memory=memory, verbose=True)  # Create conversational retrieval chain

    chat_response = conversation_with_retrieval.invoke({"question": template.format(input_text)})  # Invoke the chain

    # get just sql
    #sql_query = chat_response.split("```sql")
    #sql_query = sql_query[1].split("\n```\n\n")[0]
    #sql_query = "```sql" + sql_query + "```\n"

    #return sql_query
    return chat_response['answer']  # Return the answer
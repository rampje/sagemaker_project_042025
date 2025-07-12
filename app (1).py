import streamlit as st
import pandas as pd
import json
import model1 as mod
import prompts as pr
import sqlite3

# Set the model ID, schema file, and database file
model_id = "meta.llama3-70b-instruct-v1:0"
schema_file = "data/projects_schema2.json"
db_file = "data/demo1.db"


# Open and load the JSON file
with open(schema_file, 'r') as file:
    schema = json.load(file)
    

st.title("Reseach Projects Chatbot Demo")
 
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# Get user input
user_input = st.chat_input("Chat with your bot here", max_chars=100)
        
if user_input:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Create 1st prompt for model with user input
    prompt1 = pr.prompt1(schema, user_input)
    
    # Create SQL query from natural language question
    sql_query = mod.run_model(prompt1, model_id)
    print(sql_query)
    
    # connect to database and query the table 
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    query_success = False
    try:
        result_df = pd.read_sql_query(sql_query, conn, index_col=None)
        query_success = True
    except Exception as e:
        print(f"ERROR OCCURRED: {e}")
        chat_response = sql_query
        
    conn.close()

    if query_success is True:
        # Create 2nd prompt for model using SQL query result from 1st prompt
        prompt2 = pr.prompt2(user_input, result_df, sql_query)
    
        # Create chat response with natural language question and SQL result
        chat_response = mod.run_model(prompt2, model_id)
        print(chat_response)
    
    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(chat_response)

    # Add chatbot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": chat_response})
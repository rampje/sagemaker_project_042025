def prompt1(schema, user_input):
    prompt = f"""
    The following JSON has table information.
    Each table contains the following information:
    - Name: The name of the table
    - Description: A brief description of the table
    - Columns: The columns of the table, listed under the 'columns' key. Each column contains:
      - Name: The name of the column
      - Description: A brief description of the column
      - Type: The data type of the column
      - Synonyms: Optional synonyms for the column name
    - Sample Queries: Optional sample queries for the table, listed under the 'sample_data' key
    schema information:
    
    JSON_START
    
    {schema}
    
    JSON_END
    
    Given this information, Write a SQL query that retrieves the data asked in the question below using only the column names in the JSON. 
    
    QUESTION: {user_input}

    Please only provide the SQL query in your answer. If the question is ambiguous, say why and don't provide a SQL query. 
    If your query requires unaggregated "project_name" or "project_id", include "link" column in SELECT statement
    """
    return prompt


def prompt2(user_input, sql_query_result, sql_query):
    prompt = f"""
    Given the following SQL query

    SQL_QUERY_START
    {sql_query}
    SQL_QUERY_END
    
    And given the information in the SQL query result in the dataframe below

    DATAFRAME_START
    {sql_query_result.to_string()}
    DATAFRAME_END

    Provide a natural language answer to the question below. 
    If the dataframe contains "project_name" format the projects as hyperlinks using the "link" column.
    If the dataframe has more than 1 attribute, include a table in your answer.
    Don't include "link" column
    
    QUESTION: {user_input}
    """
    return prompt
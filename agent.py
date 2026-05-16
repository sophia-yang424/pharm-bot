from typing import TypedDict

from databricks import sql
import os
import pandas as pd
from dotenv import load_dotenv
import openai

load_dotenv()

hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
token = os.getenv("DATABRICKS_TOKEN")
#setting up connection, remember, tab is to accept autocorrect
with sql.connect(
    server_hostname=hostname,
    http_path=http_path,
    access_token=token,
) as connection:
    #test query to get top 10 drugs by total claims
    with connection.cursor() as cursor:
        #cursor is object that allows us to execute queries and fetch results from databricks server
        cursor.execute("""
            SELECT Brnd_Name, SUM(Tot_Clms) AS trx
            FROM partd_prescribers
            GROUP BY Brnd_Name
            ORDER BY trx DESC
            LIMIT 10
        """) #a test query to get top 10 drugs by total claims, select brand name and its total claims overall
        rows = cursor.fetchall() #getting all rows from the query result, rows is a list of tuples, each tuple is a row in the result
        cols = [desc[0] for desc in cursor.description] #getting column names from the cursor description, cursor.description is a list of tuples, each tuple has info about a column, the first element of the tuple is the column name

df = pd.DataFrame(rows, columns=cols)
print(df)

#now lets make agent
#state is the data thta moves thru graph, every step can modify it
#define the llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class State(TypedDict):
    #a typeddict is like a dict but with specified keys and value types, it helps us keep track of what data we have in our state and what type it is
    question: str
    sql: str
    result: str
    answer: str

#nodes: each is a function that takes in state and outputs state, they can modify any part of the state
#define graph w nodes
graph_builder = StateGraph(State)
graph_builder.add_node("generate_sql", generate_sql) # each node has a name and a function, the function takes in state and outputs state
graph_builder.add_node("validate_sql", validate_sql)
graph_builder.add_node("execute_sql", execute_sql)
graph_builder.add_node("generate_answer", generate_answer)
#edges=where execution goes next after a node finishes. need a start and end edge, no start end node tho
graph_builder.add_edge(START, "generate_sql")
graph_builder.add_edge("generate_sql", "validate_sql")
graph_builder.add_edge("validate_sql", "execute_sql")
graph_builder.add_edge("execute_sql", "generate_answer")    
graph_builder.add_edge("generate_answer", END)  

graph = graph_builder.compile() #compiling turns it into an executable graph, we can now run it, initilly its justr an outline
#give the llm context!! in this function, its the only one we use the llm for
#use an f string for the context so its easier to give prompt as one long string since f string lets u insert dynamic varss as {x} inside the string
def generate_sql(state: State) -> State:
    query = state["question"] #our query is the question field in the state typeddict
    prompt = f"""

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
#from db_tool import run_query
#df = run_query("""
 #           SELECT * FROM partd_prescribers LIMIT 1
  #         """)

#pd.options.display.max_columns = None
#pd.options.display.max_rows = None
#print(df.head())


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
    val_error: bool
    val_error_msg: str

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
graph_builder.add_conditional_edge("validate_sql", route_edge)
graph_builder.add_edge("execute_sql", "generate_answer")    
graph_builder.add_edge("generate_answer", END)  
#nodes have function and name, edges have beginning node and destination node, loops and conditional edges are allowed
#graphs are collectin of edges and nodes, they define the flow of execution, they are like a blueprint for how agent proceses input and outputs final result

graph = graph_builder.compile() #compiling turns it into an executable graph, we can now run it with diff inputs, initilly its justr an outline
#give the llm context!! in this function, its the only one we use the llm for
#use an f string for the context so its easier to give prompt as one long string since f string lets u insert dynamic varss as {x} inside the string

#to get final answer can print the final state with the answer field, state[answer]
def generate_sql(state: State) -> State:
    query = state["question"] #our query is the question field in the state typeddict
    prompt = f"""
    You are a helpful pharmaceutical business analyst SQL assistant, turning natural language into SQL
Business definitions:
- TRx means total prescriptions. In this dataset, use SUM(tot_clms) as TRx.
- TRx share means drug TRx divided by total TRx in the selected comparison set.
- NBRx cannot be calculated from this dataset because patient-level new-to-brand history is unavailable.

Rules:
- Only use table partd_prescribers.
- Only generate SELECT queries.
- Limit results to 20 rows unless asked otherwise
-Never use any of the following: "insert", "update", "delete", "drop", "alter", "create", "merge", "truncate"
-Only return SQL
-Only use values from the following schema:
Table name: partd_prescribers
Columns:
- Prscrbr_NPI:
National Provider Identifier (unique provider ID).

- Prscrbr_Last_Org_Name:
Prescriber last name or organization name.

- Prscrbr_First_Name:
Prescriber first name.

- Prscrbr_City:
Prescriber city.

- Prscrbr_State_Abrvtn:
Prescriber state abbreviation.

- Prscrbr_State_FIPS:
Numeric FIPS code representing the provider's state.

- Prscrbr_Type:
Provider specialty/type (ex: Internal Medicine, Endocrinology).

- Prscrbr_Type_Src:
Source used to determine provider specialty/type.

- Brnd_Name:
Brand drug name.

- Gnrc_Name:
Generic/molecule drug name.

- Tot_Clms:
Total prescription claims. Use SUM(Tot_Clms) as a proxy for TRx.

- Tot_30day_Fills:
Standardized number of 30-day prescription fills.

- Tot_Day_Suply:
Total days of medication supplied across prescriptions.

- Tot_Drug_Cst:
Total drug cost associated with the claims.

- Tot_Benes:
Total number of unique beneficiaries/patients receiving the drug.

- GE65_Sprsn_Flag:
Suppression flag for age 65+ metrics. Indicates whether some values are hidden/suppressed for privacy.

- GE65_Tot_Clms:
Total prescription claims for beneficiaries age 65 and older.

- GE65_Tot_30day_Fills:
Standardized 30-day fills for beneficiaries age 65 and older.

- GE65_Tot_Drug_Cst:
Total drug cost for beneficiaries age 65 and older.

- GE65_Tot_Day_Suply:
Total medication days supplied for beneficiaries age 65 and older.

- GE65_Bene_Sprsn_Flag:
Suppression flag for beneficiary counts age 65+.

- GE65_Tot_Benes:
Total number of beneficiaries/patients age 65 and older. 

Question: {query}
Errors: {state["val_error_msg"] if state["val_error"] else "None"}
"""

#to get final answer can print the final state with the answer field, state["answer"]

def validate_sql(state: State) -> State:
    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "merge", "truncate"]
    state["val_error"] = False
    lower = state["sql"].lower()
    for word in forbidden:
         if word in lower:
             state["val_error"] = True
             state["val_error_msg"] = f"SQL validation error: found forbidden word '{word}' in query."
             return state
    if "select" not in lower:
        state["val_error"] = True
        state["val_error_msg"] = "SQL validation error: query must contain 'SELECT'."
        return state
    return state

def route_edge(state: State) -> str:
    if state["val_error"] == True:
        return "generate_sql" #name of node to go to, we want to go back to gen ssql if there an error
    else:
        return "execute_sql" #if no error, then proceed

#def execute_sql(state: State) -> State:
    
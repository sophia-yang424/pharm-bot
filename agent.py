from typing import TypedDict
from langchain_tavily import TavilySearch
from databricks import sql
import os
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
#langchain is the part that integrates the llm into our graph, the langgraph stuff is then done on top this
load_dotenv()

hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
token = os.getenv("DATABRICKS_TOKEN")
#setting up connection, remember, tab is to accept autocorrect
#with sql.connect(
    #server_hostname=hostname,
    #http_path=http_path,
    #access_token=token,
#) as connection:
    #test query to get top 10 drugs by total claims
 #   with connection.cursor() as cursor:
        #cursor is object that allows us to execute queries and fetch results from databricks server
       # cursor.execute("""
        #    SELECT Brnd_Name, SUM(Tot_Clms) AS trx
        #    FROM partd_prescribers
        #    GROUP BY Brnd_Name
          #  ORDER BY trx DESC
          #  LIMIT 10
      #  """) #a test query to get top 10 drugs by total claims, select brand name and its total claims overall
       # rows = cursor.fetchall() #getting all rows from the query result, rows is a list of tuples, each tuple is a row in the result
        #cols = [desc[0] for desc in cursor.description] #getting column names from the cursor description, cursor.description is a list of tuples, each tuple has info about a column, the first element of the tuple is the column name

#df = pd.DataFrame(rows, columns=cols)
#print(df)
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
tavily_search = TavilySearch(
    max_results=2,
    topic="general"
)
tools = [tavily_search]
llm_with_tools = llm.bind_tools(tools)
counter_limit = 2
class State(TypedDict):
    #a typeddict is like a dict but with specified keys and value types, it helps us keep track of what data we have in our state and what type it is
    question: str
    sql: str
    result: str
    search_answer: str
    answer: str
    combined_answer: str
    val_error: bool 
    val_error_msg: str 
    tool_decision: str # "sql", "tools", or "both"
    eval_decision: str # "final", "sql", or "tools"
    counter: int #to keep track of how many times we have gone thru the loop, to prevent infinite loops


#give the llm context!! in this function, its the only one we use the llm for
#use an f string for the context so its easier to give prompt as one long string since f string lets u insert dynamic varss as {x} inside the string

#to get final answer can print the final state with the answer field, state[answer]

#prompt llm to decide if it needs tools

schema = """Table name: partd_prescribers
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
"""
def classify_question(state: State) -> State:
    return {}
def should_continue(state: State) -> State:
    prompt = f"""The user has asked the question {state["question"]}. Based on the question, and the given schema: {schema}, if you think an answer that addresses all parts of the user's question can be 
        answered from querying the table, return "sql". If you think to completely address the question, you need external info from a web search api call, return "tools". If you think the question needs both, return "both". Only return one of these three options as a string, and nothing else.
        Classify the question:
- Return "sql" if the ENTIRE question can be answered from the database
- Return "tools" if the ENTIRE question needs external web search (no dataset info needed at all)
- Return "both" if ANY part of the question requires the database AND any other part requires external info

Return only one of: "sql", "tools", "both"
        """
    response = llm.invoke(prompt)
    answer = response.content.strip().lower()
    if answer not in ["sql", "tools", "both"]:
        answer = "both"
    return answer

    #then add conditional edge and an extra node before gen_sql to classify problem, and a tools node for if tool is needed
def generate_sql(state: State) -> State:
   # state["val_error"] = False 
    #state["val_error_msg"] = ""
    #you can access parts of state by indexing it, but you cant update the fields just by indexing
    query = state["question"] #our query is the question field in the state typeddict
    prompt = f"""
    You are a helpful pharmaceutical business analyst SQL assistant, turning natural language into SQL.
    Only answer parts that you believe are answerable based on the dataset of the following schema: {schema}. If a part of the question is not answerable based on that dataset, ignore that part and do not include it in the SQL query. Only generate SQL for the parts of the question that are answerable with the given dataset.
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
-Only use values from the following schema: {schema}
Question: {query}
Errors: { state["val_error_msg"] if state["val_error"] else "None"}
"""
    response = llm.invoke(prompt)
    sql = response.content.strip()
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1]
        sql = sql.rsplit("```", 1)[0].strip()
    return {"sql": sql}
#langgraph is designed to update your state via this dict entry, given the key
#cannot do state state["sql"] = response, the changes wont persist,need to return those vals
#cannot init values like this either,to let changes persist, 
# => you have to pass them as args in graph.invoke when starting the run, or return them in a dict and langgraph will update the state w that using the key u gave

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

def execute_sql(state: State) -> State:
    from db_tool import run_query
    import re
    raw = state["sql"]
    blocks = re.findall(r"```(?:sql)?\s*(SELECT.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if not blocks:
        blocks = [q.strip() for q in re.split(r";\s*\n", raw) if q.strip().upper().startswith("SELECT")]
    if not blocks:
        blocks = [raw]
    combined = ""
    for i, query in enumerate(blocks):
        query = query.strip().rstrip(";")
        df = run_query(query)
        combined += f"\nQuery {i+1} results:\n{df.to_string(index=False)}\n"
    return {"result": combined}
   #just the key value of the part we want to return, langgrpah is designed to update ur state if you just give it this
def generate_answer(state: State) -> State:
    result = state["result"]
    prompt = f"""You are a helpful pharmaceutical business analyst SQL assistant, summarize the answer
    to the question given below (denoted by 'Question: '), using the results from the following table denoted by 'Table: '.
    into natural language.
    The SQL query used to get the resulting table is: {state["sql"]}
    Mention that tot_clms is being used to infer TRx, if needed.
- Focus only on the parts of the question answerable from this dataset
- Do not attempt to answer parts requiring external data
- If the table is empty or has no results, say exactly: "EMPTY_RESULT: The dataset does not contain relevant data to answer this question."
    Question: {state["question"]}
    Table: {state["result"]}
"""
    response = llm.invoke(prompt)
    print("sql \n")
    #response is an ai message object, for our purposes rn for backend we just need the content of the response message, but it has other fields like unique message id etc
    return {"answer": response.content.strip()}
#nodes: each is a function that takes in state and outputs state, they can modify any part of the state
#define graph w nodes
def search_call(state: State) -> State:
    search_query = llm.invoke(f"Convert the web-searchable part of this question into a SHORT search query of under 50 words. Return ONLY the search query, no explanation, no answer: {state['question']}").content.strip()
    response = tavily_search.invoke(search_query)
    answer = llm.invoke(f"""The user has asked the question {state["question"]}.
                Based on the question, and the given search results:
               {response}, turn this into natural language answer to the user's question. Only use the search results given, and do not make up any info that is not in the search results. If the search results do not have relevant info to answer the question, say "Based on the search results, I could not find relevant info to answer the question." """)
    print("Search query:", search_query)
    print("Tavily response:", str(response)[:300])
    print("Tavily response:", str(response)[:300])
    print("search \n")
    #debug
    return {"search_answer": answer.content.strip()}

def evaluate(state: State) -> State:
    curr_answer = ""
    if state["counter"] > counter_limit:
        return {"eval_decision": "final"}
    if state.get("answer") and "EMPTY_RESULT" in state["answer"]:
        return {"eval_decision": "tools", "counter": state["counter"] + 1}
    sql_answer = state.get("answer", "")
    has_sql = bool(sql_answer) and "EMPTY_RESULT" not in sql_answer
    has_search = bool(state.get("search_answer"))
    if has_search and has_sql:
        curr_answer = state["search_answer"] + " " + sql_answer
        eval_prompt = f"""The user has asked the question {state["question"]}.
Here are two potential answers to the user's question, one based on querying the database of schema: {schema} and one based on using web search tools:
Answer based on database query results: {state["answer"] if state["answer"] else ""}
Answer based on web search results: {state["search_answer"] if state["search_answer"] else ""}   
If you believe the combined answer {curr_answer} is accurate and complete, return "final". 
If you think the answer has not completely addressed all parts of the user's question or is missing critical info that could be found via either further querying the aforementioned database of schema: {schema}, return "sql". If you think the answer is missing critical info that could be found via using web search tools, return "tools". Only return one of these 3 options as a string, and nothing else.
Criteria: Does this answer FULLY address every part of the user's question?
- If any part of the question requires information not in the database (global data, clinical info, external context), you MUST return "tools"
- If more specific database (of aforementioned schema) querying would help, return "sql"  
- ONLY return "final" if every part of the question is completely answered
 a partial answer is not sufficient, if any part of the question is not answered, then answer is not complete and you should not return final.
"""
    elif has_search:
         eval_prompt = f"""The user has asked the question {state["question"]}.
Answer based on web search results: {state["search_answer"] if state["search_answer"] else ""}   
If you believe the combined answer is accurate and complete, return "final". 
If you think the answer has not completely addressed all parts of the user's question or is missing critical info that could be found via either querying the database of schema: {schema}, return "sql". If you think the answer is missing critical info that could be found via further using web search tools, return "tools". Only return one of these 3 options as a string, and nothing else.
Criteria: Does this answer FULLY address every part of the user's question?
- If any part of the question requires information not in the database (global data, clinical info, external context), you MUST return "tools"
- If more specific database (of aforementioned schema) querying would help, return "sql"  
- ONLY return "final" if every part of the question is completely answered.
 a partial answer is not sufficient, if any part of the question is not answered, then answer is not complete and you should not return final.
"""   
    elif has_sql:
         eval_prompt = f"""The user has asked the question {state["question"]}.
Answer based on database query results: {state["answer"] if state["answer"] else ""} 
If you believe the combined answer is accurate and complete, return "final". 
If you think the answer has not completely addressed all parts of the user's question or is missing critical info that could be found via either further querying the aforementioned database of schema: {schema}, return "sql". If you think the answer is missing critical info that could be found via using web search tools, return "tools". Only return one of these 3 options as a string, and nothing else.
Criteria: Does this answer FULLY address every part of the user's question?
- If any part of the question requires information not in the database (global data, clinical info, external context), you MUST return "tools"
- If more specific database (of aforementioned schema)querying would help, return "sql"  
- ONLY return "final" if every part of the question is completely answered
 a partial answer is not sufficient, if any part of the question is not answered, then answer is not complete and you should not return final.
"""
    else:
        curr_answer = "No answer generated."
        return {"eval_decision": "final", "counter": state["counter"] + 1, "combined_answer": ""} #tech a bug to ever reach this state but for code to not crash just treat as final iteration with no answer

    decision = llm.invoke(eval_prompt).content.strip().lower()
    return {"eval_decision": decision, "counter": state["counter"] + 1, "combined_answer": curr_answer}
def e(state: State) -> State:
    return state["eval_decision"]
def combine_answers(state: State) -> State:
    if state.get("answer") and state.get("search_answer") and state.get("combined_answer"):
        prompt = f"""The user has asked the question {state["question"]}. The answer is currently {state.get("combined_answer")}. If you had to combine the SQL-based answer and the tool-based answer into one final answer to the user, how would you combine them?
Combine the following two pieces of information into one final answer to the user's question. If you think one answer is more relevant to the user's question, prioritize that answer in the combined answer. If both answers have relevant info, combine the two answers in a way that is coherent and useful to the user. If one answer is not relevant and the other is, just give the relevant answer as the final answer. If neither answer is relevant, say "Based on the search results and the database query results. You can also trim parts of the answer if they are redundant or not useful. Do not just give a list of the two answers, but actually combine them into one final answer in natural language.
"""
        response = llm.invoke(prompt)
        return {"combined_answer": response.content.strip()}
    else:
        return {}

graph_builder = StateGraph(State)
graph_builder.add_node("classify", classify_question) 
graph_builder.add_node("tools", search_call) 
graph_builder.add_node("generate_sql", generate_sql) # each node has a name and a function, the function takes in state and outputs state
graph_builder.add_node("validate_sql", validate_sql)
graph_builder.add_node("execute_sql", execute_sql)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("eval", evaluate)
graph_builder.add_node("combine", combine_answers)
#edges=where execution goes next after a node finishes. need a start and end edge, no start end node tho
graph_builder.add_edge(START, "classify")
graph_builder.add_conditional_edges("classify", should_continue, {"sql": "generate_sql", "tools": "tools", "both": "generate_sql"})
graph_builder.add_edge("generate_sql", "validate_sql")
graph_builder.add_conditional_edges("validate_sql", route_edge)
graph_builder.add_edge("execute_sql", "generate_answer")    
graph_builder.add_edge("generate_answer", "eval") 
graph_builder.add_edge("tools", "eval") 
graph_builder.add_conditional_edges("eval", e, {"sql": "generate_sql", "tools": "tools", "final": "combine"})
graph_builder.add_edge("combine", END) 
#nodes have function and name, edges have beginning node and destination node, loops and conditional edges are allowed
#graphs are collectin of edges and nodes, they define the flow of execution, they are like a blueprint for how agent proceses input and outputs final result

graph = graph_builder.compile() #compiling turns it into an executable graph, we can now run it with diff inputs, initilly its justr an outline


#graph.invoke is qhat actually runs the compiled graph in  the order we define nodes via edges
if __name__ == "__main__":
    question = input("User query: ")
    while question != "quit":
        result = {}
        for step in graph.stream({
            "question": question,
            "val_error": False,
            "val_error_msg": "",
            "counter": 0
        }):
            print(list(step.keys()))
            node_output = list(step.values())[0]
            if node_output:
                result.update(node_output)
        print("SQL query generated:", result.get("sql", "N/A"))
        sql_ans = result.get("answer", "")
        final = (result.get("combined_answer") or
                 (sql_ans if "EMPTY_RESULT" not in sql_ans else "") or
                 result.get("search_answer", "N/A"))
        print("Final answer:", final)
        question = input("User query: ")


    #result is the state at end of execution, aka the fields passed thru whole pipeline
    



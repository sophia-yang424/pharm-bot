import os
import databricks.sql as sql
import pandas as pd
load_dotenv()
hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
token = os.getenv("DATABRICKS_TOKEN")
def run_query(query):
    with sql.connect(
        server_hostname=hostname,
        http_path=http_path,
        access_token=token,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]
    
    return pd.DataFrame(rows, columns=cols)
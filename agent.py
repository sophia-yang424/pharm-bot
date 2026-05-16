from databricks import sql
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
token = os.getenv("DATABRICKS_TOKEN")

with sql.connect(
    server_hostname=hostname,
    http_path=http_path,
    access_token=token,
) as connection:
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT Brnd_Name, SUM(Tot_Clms) AS trx
            FROM partd_prescribers
            GROUP BY Brnd_Name
            ORDER BY trx DESC
            LIMIT 10
        """)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]

df = pd.DataFrame(rows, columns=cols)
print(df)

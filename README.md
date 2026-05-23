Link to the dataset: https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug  
To run:  
Download the .csv file, and upload to Databricks as a table  
Create a .env file in the same directory as this project, with your Databricks server hostname, HTTP path, user token, and OpenAI key. If you are using the search_branch or ui_branch, then you also need to add a TAVILY API key
Run the program with this .env file  
(Program currently exists as an interactive keyboard input chatbot in the terminal, UI will be added eventually)
This branch (main) includes only dataset querying

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.schema import SystemMessage

from tools.sql import list_tables, run_query_tool, describe_tables_tool
from tools.report import write_report_tool

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(openai_api_key=openai_api_key)

tables = list_tables()
print(f"Tables:\n{tables}")

prompt = ChatPromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    messages=[
        SystemMessage(
            content=(
                "You are an AI that has access to a SQLite database with tables.\n"
                f"The database has tables of: \n{tables}\n"
                "Do not make any assumptions about what tables exist "
                "or what columns exist. Instead, use the 'describe_tables' function"
            )
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)

tools = [run_query_tool, describe_tables_tool, write_report_tool]

agent = OpenAIFunctionsAgent(llm=chat, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)

# human_query = "how many users are in the database?"
# human_query = "How many users have provided an address?"
human_query = (
    "Summarize the top 5 most popular products? Write the results to a report file."
)
agent_executor(human_query)

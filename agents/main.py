import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory

from tools.sql import list_tables, run_query_tool, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

handler = ChatModelStartHandler()
chat = ChatOpenAI(openai_api_key=openai_api_key, callbacks=[handler])

tables = list_tables()
print(f"Tables:\n{tables}")

prompt = ChatPromptTemplate(
    input_variables=["input", "chat_history", "agent_scratchpad"],
    messages=[
        SystemMessage(
            content=(
                "You are an AI that has access to a SQLite database with tables.\n"
                f"The database has tables of: \n{tables}\n"
                "Do not make any assumptions about what tables exist "
                "or what columns exist. Instead, use the 'describe_tables' function"
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)

tools = [run_query_tool, describe_tables_tool, write_report_tool]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_openai_functions_agent(llm=chat, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

# human_query = "how many users are in the database?"
# human_query = "How many users have provided an address?"
human_query = "How many orders are there? Write the results to an html report"
agent_executor.invoke(
    {
        "input": human_query,
    }
)
# agent_executor({"input": "Repeat the exact process for users."})

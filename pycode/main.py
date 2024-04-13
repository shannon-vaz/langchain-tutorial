from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import os
import argparse
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

llm = OpenAI(openai_api_key=openai_api_key)
code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very short {language} function that will {task}",
)
test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following code in {language}:\n{code}",
)
code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")
test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key="test")
chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"],
)

result = chain.invoke(
    {
        "language": args.language,
        "task": args.task,
    }
)

print("-" * 80)
print(">" * 10, "Code:")
print(result["code"])
print(">" * 10, "Test:")
print(result["test"])

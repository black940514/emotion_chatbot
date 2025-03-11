from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
# from api_key import openai_api_key
from langgraph.prebuilt import create_react_agent
from assist_tools import *
from dotenv import load_dotenv
import os
# .env 파일 로드
load_dotenv()

# 환경 변수 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model='gpt-4o', temperature=0)
memory = MemorySaver()
tools =[send_email, make_calendar]
agent = create_react_agent(llm, tools=tools, checkpointer=memory)
config = {"configurable":{"thread_id": "1"}}

while True:
    user_input = input("User : ")
    if user_input.lower() in ["exit", "q"]:
        print("Goodbye!")
        break
    for e in agent.stream({"messages": [("user", user_input)]}, config=config):
        for v in e.values():
            print("Chatbot: ", v["messages"][-1].content)
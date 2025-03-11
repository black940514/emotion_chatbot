from typing import Annotated, Literal, Sequence, TypedDict
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
# from api_key import openai_api_key
from build_retriever import get_retriever
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
import gradio as gr
from utils import *

from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()
# 환경 변수 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

graph = StateGraph(State)
retreiver = get_retriever()
retreiver_tool = create_retriever_tool(
    retreiver,
    "retrieve_preference",
    "Search and return information about the preference of the user relevant to queries."
)
tools = [retreiver_tool]

def grade(state) -> Literal["generate_with_retrieval", "generate_without_retrieval"]:
    print("Check relevance")
    class grading(BaseModel):
        binary_score: str = Field(discription="Relevance score 'yes' or 'no'")
    model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, streaming=True, model='gpt-4o')
    llm_with_tool = model.with_structured_output(grading)
    prompt = PromptTemplate(
        template="""
                You are a grader assessing relevance of a retrieved document to a user question. \n
                Here is the retrieved document: {context} \n
                Here is the user question: {question} \n
                If the document contains keywords or semantic meaning related to user question, grade it as relevant.\n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
                    """,
        input_variables=["context","question"]
    )
    chain = prompt | llm_with_tool
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content[0]["text"]
    docs = last_message.content
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score
    if score=='yes':
        print("---Docs relevant.")
        return "generate_with_retrieval"
    else:
        print("---Docs not relevant.")
        return "generate_without_retrieval"

def agent(state):
    print("Agent")
    messages = state["messages"]
    model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, streaming=True, model='gpt-4o')
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

def generate_without_retrieval(state):
    print("Generate without retrieval")
    messages = state["messages"]
    model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, streaming=True, model='gpt-4o')
    query = messages[0].content
    llm_chain = model | StrOutputParser()
    msg = [
        HumanMessage(
            content=query
        )
    ]
    response = llm_chain.invoke(msg)
    return {"messages": [response]}

def generate_with_retrieval(state):
    print("Generate with retrieval")
    prompt = hub.pull("rlm/rag-prompt")
    messages = state["messages"]
    model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, streaming=True, model='gpt-4o')
    query = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    msg = [
        HumanMessage(
            content=query
        )
    ]

    rag_chain = prompt | model | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": msg})
    return {"messages": [response]}

## graph structure
graph.add_node("agent", agent)
retrieve = ToolNode(tools)
graph.add_node("retrieve", retrieve)
graph.add_node("generate_without_retrieval", generate_without_retrieval)
graph.add_node("generate_with_retrieval", generate_with_retrieval)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    }
)
graph.add_conditional_edges(
    "retrieve",
    grade
)
graph.add_edge("generate_with_retrieval", END)
graph.add_edge("generate_without_retrieval", END)

#RUN
agent = graph.compile()

def send_query(input_img, input_txt):
    msg = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "You are a compassionate emotional counseling assistant. Given the following query and an image, carefully analyze the user's current emotional state and provide empathetic, thoughtful advice and guidance. Ensure your response is supportive and helps the user understand and address their feelings.\nquery: " + input_txt
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(input_img)}"}
                }
            ]
        )
    ]
    inputs = {"messages": msg}

    for output in agent.stream(inputs):
        for key, value in output.items():
            print("Output from node : "+key)
            out = value["messages"][-1]
            if isinstance(out,str):
                yield out
            else:
                yield out.pretty_repr()

# demo = gr.Interface(send_query, inputs=[gr.Image(label='input', type='pil'),
#                                         gr.Textbox(label='query')],
#                                         outputs=gr.Textbox(label='output'))
# demo.launch(share=True)

# Blocks를 사용하여 더 자유로운 레이아웃과 스타일링을 적용합니다.
with gr.Blocks(css="""
    .title {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .input-area, .output-area {
        margin: 10px;
    }
""") as demo:
    gr.Markdown("<div class='title'>김태연의 감정 상담 챗봇</div>")
    gr.Markdown("<div class='subtitle'>사진을 업로드하고 상담내용을 입력해 주세요.</div>")
    
    with gr.Row():
        with gr.Column(elem_classes="input-area"):
            input_img = gr.Image(label="사용자 이미지", type="pil")
            input_txt = gr.Textbox(label="질문", placeholder="본인의 현재 상태에 대한 질문을 입력하세요..", lines=2)
            submit_btn = gr.Button("전송")
        with gr.Column(elem_classes="output-area"):
            output_txt = gr.Textbox(label="챗봇 응답", placeholder="상담 답변이 여기에 표시됩니다..", lines=12)
    
    submit_btn.click(fn=send_query, inputs=[input_img, input_txt], outputs=output_txt)

demo.launch(share=True)
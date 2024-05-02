from langchain.agents import Tool, ConversationalChatAgent, AgentExecutor
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from datetime import date
import os

# from config import CHAT_MODEL
from tools.focused_labs_q_and_a_tool import create_vector_db_tool
from utils import is_answer_formatted_in_json, output_response, transform_source_docs
from langchain_community.llms.bedrock import Bedrock
from langchain.agents import Tool

class Agent:

    def __init__(self, personality):
        self.personality = personality
        self.llm = Bedrock(
            # model_id=model_llama2,
            model_id="mistral.mixtral-8x7b-instruct-v0:1",
            region_name=os.getenv('AWS_DEFAULT_REGION'),
            credentials_profile_name=os.getenv('AWS_PROFILE'),
            # model_kwargs=model_kwargs_llama2,
            model_kwargs={"max_tokens":2048,"top_k":50,"top_p":1,"temperature":0},
            streaming=False
        )
        self.agent_executor = self.create_agent_executor()

    def create_agent_executor(self):
        q_and_a_tool = create_vector_db_tool(llm=self.llm)
        tools = [
            Tool(
                name="Hello QA",
                return_direct=True,
                func=lambda query: _parse_source_docs(q_and_a_tool, query),
                # description="Search and retrieve documents related to Hello (other naming for Hello: ['TIH', 'Hello,', 'This Is Hello'])."
                description="useful for when you need to answer questions about Hello documents Knowledge Base"
            ),
        ]
        memory = ConversationBufferWindowMemory(llm=self.llm, k=10, memory_key="chat_history", return_messages=True,
                                                human_prefix="user", ai_prefix="assistant", input_key="input")
        custom_agent = ConversationalChatAgent.from_llm_and_tools(llm=self.llm,
                                                                  tools=tools,
                                                                  verbose=True,
                                                                  max_iterations=3,
                                                                  early_stopping_method='generate',
                                                                  handle_parsing_errors=True,
                                                                  memory=memory,
                                                                  input_variables=["input", "chat_history",
                                                                                   "agent_scratchpad"],
                                                                  system_message=
                                                                  f"""
                                                                  <s> [INST] Have a conversation with a human, answering the 
                                                                  following as best you can and try to use a tool to help. 
                                                                  You have access to the following tools: 
                                                                  Hello QA-useful for when you need to answer
                                                                  questions about Hello.If you don't know the 
                                                                  answer don't make one up, just say "Hmm, I'm not sure 
                                                                  please contact hello@thisishello.it for further assistance."
                                                                  Answer questions from the perspective of a {self.personality} [INST] </s>"""
                                                                  )
        return AgentExecutor.from_agent_and_tools(agent=custom_agent,    # Sets the system to use openai functions
                                                  tools=tools,                   # Sets the tools visible to the LLM
                                                  memory=memory,
                                                  return_intermediate_steps=False,     # Get a list of traces of the trajectory of the chain
                                                #   max_execution_time=1000 # The maximum amount of wall clock time to spend in the execution loop
                                                  max_iterations=3,          # Sets the number of intermediate steps
                                                  early_stopping_method="force",   # Applies final pass to generate an output if max iterations is reached
                                                  verbose=True)

    def query_agent(self, user_input):
        try:
            response = self.agent_executor.invoke(input=user_input)
            if is_answer_formatted_in_json(response):
                return response
            return f"""
            {{
                "result": "{response}",
                "sources": []
            }}"""

        except ValueError as e:
            response = str(e)
            response_prefix = "Could not parse LLM output: `\nAI: "
            if not response.startswith(response_prefix):
                raise e
            response_suffix = "`"
            if response.startswith(response_prefix):
                response = response[len(response_prefix):]
            if response.endswith(response_suffix):
                response = response[:-len(response_suffix)]
            output_response(response)
            return response


def _parse_source_docs(q_and_a_tool: RetrievalQA, query: str):
    result = q_and_a_tool({"question": query})
    return transform_source_docs(result)

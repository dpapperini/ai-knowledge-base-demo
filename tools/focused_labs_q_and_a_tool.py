import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.llms.bedrock import Bedrock
from langchain_community.retrievers.bedrock import AmazonKnowledgeBasesRetriever

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def create_vector_db_tool(llm: Bedrock):
    # Get retriever from vectorstore
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=os.getenv('AWS_HELLO_KNOWLEDGE_BASE_ID'),
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 3}},
        # endpoint_url=endpoint_url
        region_name=os.getenv('AWS_DEFAULT_REGION'),
        credentials_profile_name=os.getenv('AWS_PROFILE')
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        return_source_documents=True,
        input_key="question",
        retriever=retriever
    )

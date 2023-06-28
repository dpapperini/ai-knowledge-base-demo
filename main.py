import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import get_redis_connection
from importer import NOTION_INDEX_NAME, WEB_SCRAPE_INDEX_NAME, \
    import_notion_data, number_of_stored_notion_docs, \
    import_web_scrape_data, number_of_stored_web_scrape_docs, \
    compose_graph

allowed_origins = [
    "http://localhost:3000",
]

query_engines = {}


def init_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def load_notion_documents():
    print("Loading notion docs")
    redis_client = get_redis_connection()
    redis_client.ft(NOTION_INDEX_NAME).dropindex()
    import_notion_data()
    print("Number of stored notion docs: ", number_of_stored_notion_docs())


def load_web_scrape_documents():
    print("Loading web scraped docs")
    redis_client = get_redis_connection()
    redis_client.ft(WEB_SCRAPE_INDEX_NAME).dropindex()
    import_web_scrape_data()
    print("Number of store web scraped docs: ", number_of_stored_web_scrape_docs())


class Question(BaseModel):
    text: str
    role: str


prompt_intros = {
    "none": "You know a lot about Focused Labs",
    "developer": "Your role at Focused Labs is software developer",
    "designer": "Your role at Focused Labs is UX / UI designer",
    "executive": "Your role at Focused Labs is stakeholder",
    "client": "You are a prospective client evaluating Focused Labs as a potential partner"
}


def expand_prompt(question: Question):
    if question.role is None:
        question.role = "none"
    prompt = prompt_intros[question.role]
    return prompt + ". " + question.text


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_logging()

    # Load documents. This can be slow so block the API until ready
    load_notion_documents()
    load_web_scrape_documents()

    # Build a graph from the indexes just created and store its query engine
    query_engines['doc_query_engine'] = compose_graph()
    yield
    query_engines['doc_query_engine'] = None


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/query/")
async def query(question: Question):
    prompt = expand_prompt(question)
    print(prompt)
    response = query_engines['doc_query_engine'].query(prompt)
    return {"response": response}

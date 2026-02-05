import os
import sys
from dotenv import load_dotenv

# 1. Force Load Environment Variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print(" API Key Missing")
    sys.exit(1)
os.environ["OPENAI_API_KEY"] = api_key

from typing import TypedDict
from langgraph.graph import StateGraph, END
from llama_index.core import (
    PropertyGraphIndex, 
    SimpleDirectoryReader, 
    Settings, 
    StorageContext
)
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import ChatOpenAI

# 2. Configuration
Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
extraction_llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
PERSIST_DIR = "./storage"

# 3. Robust Loading/Indexing Logic
try:
    if os.path.exists(PERSIST_DIR):
        print("--- Loading Existing Knowledge Graph ---")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        # Fix for the ValueError: explicitly pass the graph_store
        index = PropertyGraphIndex(
            property_graph_store=storage_context.property_graph_store,
            storage_context=storage_context,
            llm=Settings.llm,
            embed_model=Settings.embed_model
        )
    else:
        raise FileNotFoundError
except Exception as e:
    print(f"--- Index not found or corrupted ({e}). Creating new... ---")
    documents = SimpleDirectoryReader("./data").load_data()
    documents = documents[:2]
    # For testing, you might want documents = documents[:20]
    
    kg_extractor = SchemaLLMPathExtractor(
        llm=extraction_llm,
        possible_entities=["INGREDIENT", "SPICE", "TECHNIQUE", "DISH"],
        possible_relations=["REQUIRED_FOR", "PAIRS_WITH", "USED_IN"],
        strict=False
    )

    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[kg_extractor],
        show_progress=True
    )
    index.storage_context.persist(persist_dir=PERSIST_DIR)

# 4. Agent Nodes with Iteration Tracking
class ChefState(TypedDict):
    query: str
    recipe_draft: str
    is_authentic: bool
    feedback: str
    iterations: int # Track loops to prevent infinite cycles

kg_engine = index.as_query_engine(include_text=True, similarity_top_k=5)

def research_recipe(state: ChefState):
    iters = state.get("iterations", 0)
    print(f"--- Researcher working (Attempt {iters + 1}) ---")
    
    query = state["query"]
    if state.get("feedback"):
        query += f" The previous attempt was rejected. Fix this: {state['feedback']}"
    
    res = kg_engine.query(query)
    return {"recipe_draft": str(res), "iterations": iters + 1}

def critique_recipe(state: ChefState):
    print("--- Critic Reviewing ---")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # If the researcher already said it can't find info, don't loop forever
    if "sorry" in state["recipe_draft"].lower() or "not possible" in state["recipe_draft"].lower():
        return {"is_authentic": False, "feedback": "Data not found in source PDF."}

    prompt = (
        "Review this Bengali recipe for authenticity. "
        "If it is a complete and correct recipe, reply ONLY with 'AUTHENTIC'. "
        "Otherwise, explain what is missing."
        f"\n\nRecipe: {state['recipe_draft']}"
    )
    res = llm.invoke(prompt)
    feedback = res.content.strip()
    return {"feedback": feedback, "is_authentic": "AUTHENTIC" in feedback.upper()}

# 5. Graph with "Stop Condition"
def should_continue(state: ChefState):
    # Stop if authentic OR if we've tried 3 times
    if state["is_authentic"] or state["iterations"] >= 3:
        return END
    return "researcher"

workflow = StateGraph(ChefState)
workflow.add_node("researcher", research_recipe)
workflow.add_node("reviewer", critique_recipe)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "reviewer")
workflow.add_conditional_edges("reviewer", should_continue)

app = workflow.compile()

if __name__ == "__main__":
    # Test query
    inputs = {"query": "খাঁটি গরুর তেহারি রান্নার রেসিপি", "iterations": 0}
    for output in app.stream(inputs):
        print(output)
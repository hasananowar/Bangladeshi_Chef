import os
import sys
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, Settings
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("CRITICAL ERROR: API_KEY not found.")
    print("Please create a file named .env in this folder: ", os.getcwd())
    sys.exit(1)
else:
    print(f"API Key loaded: {api_key[:5]}...*******")

os.environ["OPENAI_API_KEY"] = api_key

# --- 0. Configuration & Knowledge Graph Setup ---
# Use GPT-4o for robust Bengali handling
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# Define the Knowledge Graph Schema for Cooking
entities = ["INGREDIENT", "SPICE", "TECHNIQUE", "DISH", "UTENSIL"]
relations = ["REQUIRED_FOR", "PAIRS_WITH", "USED_IN", "PREPARED_VIA", "ALTERNATE_OF"]

validation_schema = {
    "SPICE": ["REQUIRED_FOR", "PAIRS_WITH"],
    "INGREDIENT": ["USED_IN", "PREPARED_VIA"],
    "DISH": ["PREPARED_VIA"]
}

kg_extractor = SchemaLLMPathExtractor(
    llm=Settings.llm,                    # Pass the LLM explicitly
    possible_entities=entities,          # WAS: entity_types
    possible_relations=relations,        # WAS: relation_types
    kg_validation_schema=validation_schema, # WAS: validation_schema
    strict=False
)

print("--- Indexing Knowledge Graph (This may take time) ---")
# Ensure your PDF is in the './data' folder
documents = SimpleDirectoryReader("./data").load_data()
index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[kg_extractor],
    show_progress=True
)
print("--- Indexing Complete ---")

# --- 1. Define Agent State ---
class ChefState(TypedDict):
    query: str
    recipe_draft: str
    is_authentic: bool
    feedback: str
    iteration_count: int

# --- 2. Define Nodes ---

# Node A: The Researcher (Uses the Knowledge Graph)
kg_engine = index.as_query_engine(include_text=True)

def research_recipe(state: ChefState):
    print(f"--- Researcher Working (Iteration {state.get('iteration_count', 0)}) ---")
    
    # If there is feedback, append it to the query to refine the search
    current_query = state["query"]
    if state.get("feedback"):
        current_query = f"{state['query']}. IMPORTANT adjustment based on critic feedback: {state['feedback']}"
    
    # Query the Property Graph
    res = kg_engine.query(current_query)
    
    return {
        "recipe_draft": str(res), 
        "iteration_count": state.get("iteration_count", 0) + 1
    }

# Node B: The Critic (Checks Authenticity)
critic_llm = ChatOpenAI(model="gpt-4o")

def critique_recipe(state: ChefState):
    print("--- Critic Reviewing ---")
    
    # Bengali System Prompt for the Critic
    system_instruction = """
    আপনি একজন কঠোর বাংলাদেশী খাদ্য সমালোচক। প্রদত্ত রেসিপিটি খাঁটি কিনা তা যাচাই করুন।
    - সরিষার তেল, পাঁচ ফোড়ন বা বাগার দেওয়া ঠিকমতো হয়েছে কিনা দেখুন।
    - যদি সব ঠিক থাকে, তবে শুধু 'AUTHENTIC' শব্দটি লিখুন।
    - যদি ভুল থাকে, তবে ইংরেজিতে বা বাংলায় স্পষ্ট করে লিখুন কী কী বাদ পড়েছে।
    """
    
    prompt = f"{system_instruction}\n\nRecipe to check: {state['recipe_draft']}"
    res = critic_llm.invoke(prompt)
    
    response_text = res.content
    is_authentic = "AUTHENTIC" in response_text.upper()
    
    return {"feedback": response_text, "is_authentic": is_authentic}

# --- 3. Build the Graph ---
workflow = StateGraph(ChefState)

workflow.add_node("researcher", research_recipe)
workflow.add_node("reviewer", critique_recipe)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "reviewer")

def check_authenticity(state: ChefState):
    if state["is_authentic"]:
        return "end"
    # Safety break to prevent infinite loops
    if state["iteration_count"] > 3:
        return "end"
    return "researcher"

workflow.add_conditional_edges(
    "reviewer",
    check_authenticity,
    {"end": END, "researcher": "researcher"}
)

app = workflow.compile()

# --- 4. Run the Agent ---
if __name__ == "__main__":
    inputs = {"query": "How to make authentic Shorshe Ilish?", "iteration_count": 0}
    print(f"Querying: {inputs['query']}")
    
    # Stream the output
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished Node: {key}")
            if key == "reviewer":
                print(f"Critic Status: {'Authentic' if value['is_authentic'] else ' Needs Improvement'}")
    
    print("\nFinal Recipe:")
    print(output['reviewer'].get('feedback') if not output['reviewer']['is_authentic'] else "Recipe is Perfect.")
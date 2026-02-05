# Bangladeshi Chef

An advanced AI application demonstrating **Agentic Workflows** and **Multilingual Property Graph RAG** to preserve the authenticity of Bangladeshi cuisine.

## Technical Highlights
- **Frameworks:** LangGraph (State Machines), LlamaIndex (Data Orchestration).
- **Core Logic:** Property Graph Indexing for entity-relationship mapping (Spices → Techniques → Dishes).
- **Multilingual Support:** Native Bengali processing using GPT-4o and Multilingual Embeddings.
- **Agentic Loop:** A Researcher-Critic loop that iterates on recipe drafts until cultural authenticity is verified.



## Importance of this App
Standard RAG (Vector-only) often fails in specialized domains like regional cooking because:
1. **Context Loss:** It treats ingredients as flat lists. Our **Property Graph** understands that *Mustard Oil* is a prerequisite for *Shorshe Ilish*, not just an optional fat.
2. **Hallucination:** Generic LLMs often "Westernize" recipes. This app uses a **Critic Agent** to enforce traditional Bangladeshi culinary standards.
3. **In-Context Learning:** The system learns specific regional variations directly from the provided PDF cookbook without requiring model fine-tuning.

##  Setup Instructions

### Environment Creation
Ensure you have Conda installed:
```bash
conda create -n chef_env python=3.10 -y
conda activate chef_env
pip install -r requirements.txt
```

### To run the Agent logic (CLI)
```bash
python Bangladeshi_Chef.py
```
### To launch the Web Interface
```bash
streamlit run app.py
```
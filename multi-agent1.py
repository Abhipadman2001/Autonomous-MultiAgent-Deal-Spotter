import os
import warnings
import urllib.request
import sys
import time
from datetime import datetime
import json

# Increase recursion depth to prevent errors with deep object chains in AI libraries
sys.setrecursionlimit(3000)

# ==============================================================================
# 0. SETUP & ENVIRONMENT
# ==============================================================================

# --- COMPATIBILITY PATCH FOR NOTEBOOKS/GRADIO ---
try:
    if hasattr(sys.stdout, 'write') and not hasattr(sys.stdout, 'buffer'):
        class DummyBuffer:
            def write(self, b): pass
            def flush(self): pass
        sys.stdout.buffer = DummyBuffer()
    
    if hasattr(sys.stderr, 'write') and not hasattr(sys.stderr, 'buffer'):
        class DummyBuffer:
            def write(self, b): pass
            def flush(self): pass
        sys.stderr.buffer = DummyBuffer()
except Exception:
    pass

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain") 

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["OPENAI_API_KEY"] = "NA"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# --- LIBRARY CHECKS ---
def check_library(lib_name, pip_name):
    try:
        __import__(lib_name)
    except ImportError:
        print(f"❌ Missing Library: '{pip_name}'.")
        print(f"   Please run: pip install {pip_name}")
        sys.exit(1)

check_library("dotenv", "python-dotenv") # Added python-dotenv
check_library("duckduckgo_search", "duckduckgo-search") # Keeping as backup
check_library("chromadb", "chromadb")
check_library("langchain_community", "langchain-community")
check_library("crewai_tools", "crewai-tools")
check_library("tavily", "tavily-python") 

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Verify TAVILY_API_KEY is loaded
if not os.environ.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY") == "your_actual_api_key_here":
    print("⚠️ WARNING: TAVILY_API_KEY is missing or invalid in your .env file!")
    print("Please add your key to the .env file before running.")

import chromadb
from chromadb.utils import embedding_functions
import gradio as gr
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from tavily import TavilyClient

# ==============================================================================
# 1. DATABASE SETUP (CHROMA + OLLAMA)
# ==============================================================================

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_NAME = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "shopping_knowledge_base"

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url=OLLAMA_URL,
    model_name=EMBED_MODEL
)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef
)

# --- LOAD DATA INTO DB ---
base_dir = os.path.dirname(os.path.abspath(__file__))
txt_file_path = os.path.join(base_dir, 'shopping_targets.txt')

def load_data_to_chroma():
    if not os.path.exists(txt_file_path):
        with open(txt_file_path, 'w') as f:
            f.write("Product: Test Product\nTarget Price: $100\nNotes: Test notes.")
    
    with open(txt_file_path, 'r') as f:
        text = f.read()
    
    chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
    if len(chunks) < 2 and len(text) > 50:
        chunks = [c.strip() for c in text.split('\n') if c.strip()]
    
    if not chunks: return
    
    ids = [f"id_{i}" for i in range(len(chunks))]
    
    if collection.count() > 0:
        existing_ids = collection.get()['ids']
        if existing_ids:
            collection.delete(existing_ids)

    BATCH_SIZE = 5
    print(f"⏳ Loading {len(chunks)} entries in batches of {BATCH_SIZE}...")
    
    try:
        ollama_ef(["warmup"]) 
    except:
        pass 

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        try:
            collection.add(documents=batch_chunks, ids=batch_ids)
            print(f"   - Processed batch {i//BATCH_SIZE + 1}")
        except Exception as batch_error:
            print(f"   ❌ Error in batch {i}: {batch_error}")
            
    print(f"✅ Database Ready: All data loaded.")

try:
    load_data_to_chroma()
except Exception as e:
    print(f"⚠️ Warning: Could not load ChromaDB. Error: {e}")

# ==============================================================================
# 2. CUSTOM TOOLS
# ==============================================================================

class WebSearchInput(BaseModel):
    query: str = Field(..., description="The product name to search for.")

class WebSearchTool(BaseTool):
    name: str = "Tavily Global Retailer Search"
    description: str = "Searches the web for live prices from ANY Indian online retailer (not just Amazon/Flipkart)."
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str) -> str:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key or "your_actual_api_key_here" in api_key:
            return "❌ Error: TAVILY_API_KEY not found or is invalid. Please set it in the .env file."
            
        try:
            # We append 'buy online india' to ensure we get Indian results
            # but we DO NOT restrict domains, allowing any shop to appear.
            search_query = f"{query} price buy online india"
            
            client = TavilyClient(api_key=api_key)
            response = client.search(query=search_query, max_results=7)
            
            results = response.get('results', [])
            
            if not results:
                return "No results found."

            formatted_output = []
            for res in results:
                url = res.get('url', 'No URL')
                content = res.get('content', 'No content')
                formatted_output.append(f"SOURCE: {url}\nDETAILS: {content}\n")
            
            return "\n-------------------\n".join(formatted_output)

        except Exception as e:
            return f"Tavily Search Error: {e}"

web_search_tool = WebSearchTool()

class ChromaSearchInput(BaseModel):
    query: str = Field(..., description="The product name to look up in the database.")

class ChromaSearchTool(BaseTool):
    name: str = "Search Knowledge Base"
    description: str = "Searches the shopping list for target prices using Vector Search."
    args_schema: Type[BaseModel] = ChromaSearchInput

    def _run(self, query: str) -> str:
        try:
            results = collection.query(query_texts=[query], n_results=2)
            documents = results['documents'][0]
            if not documents: return "No relevant target price found in the database."
            return "Found in Knowledge Base:\n" + "\n---\n".join(documents)
        except Exception as e:
            return f"Database Error: {str(e)}"

chroma_tool = ChromaSearchTool()

# ==============================================================================
# 3. AGENT CONFIGURATION
# ==============================================================================

my_llm = LLM(model=f"ollama/{MODEL_NAME}", base_url=OLLAMA_URL)

def create_crew(product_name):
    now = datetime.now()
    current_date_str = now.strftime("%B %Y")
    
    # Agent 1: Scout
    scout = Agent(
        role='Deal Scout',
        goal=f'Find the ABSOLUTE LOWEST price for {product_name} from ANY Indian retailer and identify the specific Store Name.',
        backstory=(
            "You are an unbiased price extractor. "
            "You search the entire web for the best deal. "
            "You DO NOT favor Amazon or Flipkart. You treat all reputable Indian stores equally "
            "(e.g., Tatacliq, Reliance, Croma, Vijay Sales, or smaller verified shops). "
            "You simply extract the lowest number that looks like a valid price and the store it belongs to. "
            "CRITICAL: When calling tools, provide the query as a simple string. Do NOT pass nested dictionaries or JSON schemas."
        ),
        tools=[web_search_tool],
        llm=my_llm,
        verbose=True
    )

    # Agent 2: Analyst
    analyst = Agent(
        role='Deal Analyst',
        goal='Compare found prices against targets.',
        backstory=(
            "Financial analyst. You strictly compare the specific product found by the scout. "
            "CRITICAL: When calling tools, provide the query as a simple string. Do NOT pass nested dictionaries or JSON schemas."
        ),
        tools=[chroma_tool],
        llm=my_llm,
        verbose=True
    )

    # Agent 3: Notifier
    notifier = Agent(
        role='Notification Manager',
        goal='Draft the alert.',
        backstory="Concise notification writer.",
        llm=my_llm,
        verbose=True
    )

    # Tasks
    task1 = Task(
        description=(
            f"Find the current price for '{product_name}' in India using the WebSearchTool. "
            "The tool will return text from various websites. "
            "1. Identify the Lowest Price (look for ₹ symbol). "
            "2. Identify the specific Store Name (e.g. Croma, Reliance Digital, Tatacliq, etc.). "
            "3. Return ONLY the Price and Store Name. "
        ),
        expected_output="The lowest verified price and the specific Store Name.",
        agent=scout
    )
    
    task2 = Task(
        description=f"Search Knowledge Base for {product_name} target price. Compare with found price.",
        expected_output="Comparison and Verdict.",
        agent=analyst
    )
    
    task3 = Task(
        description=(
            "Write the notification based on the findings. "
            "Ensure the Store Name is clearly visible. "
            "Verdict: [BUY/WAIT] "
            "Price: [Amount] "
            "Store: [Store Name] "
            "Analysis: [Brief comparison] "
        ),
        expected_output="Formatted text with price and store name.",
        agent=notifier
    )

    return Crew(
        agents=[scout, analyst, notifier],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True,
    )

# ==============================================================================
# 4. GRADIO UI
# ==============================================================================

def is_ollama_running():
    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=1)
        return True
    except:
        return False

def run_deal_spotter(product_name, history):
    if not product_name: 
        return "⚠️ Please enter a product name.", history, history
    if not is_ollama_running(): 
        return "❌ Ollama is not running! Please start your Ollama server.", history, history
    
    status = f"### 🕵️‍♂️ Searching for: {product_name}...\n"
    
    try:
        crew = create_crew(product_name)
        result = crew.kickoff()
        
        timestamp = datetime.now().strftime("%I:%M %p")
        history.insert(0, [timestamp, product_name])
        
        return status + str(result), history, history
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {e}\n{traceback.format_exc()}"
        return error_msg, history, history

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
body { font-family: 'Roboto', sans-serif !important; background-color: #f3f4f6; }
.gradio-container { max_width: 1100px !important; margin: 40px auto !important; background: white; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.08); padding: 30px !important; border: 1px solid #e5e7eb; }
#header-title { text-align: center; color: #111827; margin-bottom: 8px; font-size: 2.2em; font-weight: 700; }
.description { text-align: center; color: #6b7280; margin-bottom: 30px; font-size: 1.1em; }
.input-box textarea, .input-box input { border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; font-size: 16px; }
.input-box textarea:focus, .input-box input:focus { border-color: #4f46e5; box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2); }
button.primary-btn { background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%); border: none; color: white !important; font-weight: 600; font-size: 16px; border-radius: 8px; padding: 12px 24px; transition: transform 0.1s ease, box-shadow 0.2s ease; cursor: pointer; }
button.primary-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(79, 70, 229, 0.3); background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%); }
.output-markdown { border: 1px solid #e5e7eb; padding: 25px; border-radius: 12px; background: #f9fafb; margin-top: 25px; min-height: 150px; }
.footer-text { text-align: center; margin-top: 20px; font-size: 0.85em; color: #9ca3af; }
.history-sidebar { background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; height: 100%; }
.history-title { font-size: 1.1em; font-weight: 600; color: #334155; margin-bottom: 15px; border-bottom: 2px solid #cbd5e1; padding-bottom: 10px; }
</style>
"""

with gr.Blocks(title="AI Deal Spotter Pro") as demo:
    gr.HTML(custom_css)
    
    gr.Markdown("# 🛍️ AI Deal Spotter Pro", elem_id="header-title")
    gr.Markdown("Live Verified Prices from ANY Indian Retailer", elem_classes=["description"])
    
    history_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=7):
            with gr.Row(equal_height=True):
                inp = gr.Textbox(label="Product Name", placeholder="e.g. iPhone 15, Sony XM5...", scale=4, elem_classes=["input-box"], show_label=False, container=False)
                btn = gr.Button("Find Best Deal 🚀", variant="primary", scale=1, elem_classes=["primary-btn"])
            
            out = gr.Markdown(label="Deal Analysis Result", elem_classes=["output-markdown"], value="Ready to search! Enter a product above.")
            
        with gr.Column(scale=3, elem_classes=["history-sidebar"]):
            gr.Markdown("🕒 **Recent Searches**", elem_classes=["history-title"])
            
            history_table = gr.Dataframe(
                headers=["Time", "Product"], 
                datatype=["str", "str"],
                interactive=False,
                row_count=5,
                wrap=True
            )

    gr.Markdown("Powered by CrewAI, Ollama & Tavily", elem_classes=["footer-text"])

    btn.click(fn=run_deal_spotter, inputs=[inp, history_state], outputs=[out, history_state, history_table])
    inp.submit(fn=run_deal_spotter, inputs=[inp, history_state], outputs=[out, history_state, history_table])

if __name__ == "__main__":
    demo.launch(),
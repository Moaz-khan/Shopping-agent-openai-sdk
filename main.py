from dotenv import load_dotenv
import os
import requests
from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool
import rich
#Load Environment Variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") 

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")
# Setup OpenAI or Gemini Client
external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",  
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Function Tool: Get Products
@function_tool
def get_products():
    """
    Fetches a list of products from template6-six.vercel.app API.
    """
    url = "https://template6-six.vercel.app/api/products"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": p.get("title"),
                "price": p.get("price"),
                "discount": p.get("dicountPercentage"),
                "category": ", ".join(p.get("tags", [])),
                "isNew": p.get("isNew"),
                "description": p.get("description")
            }
            for p in data
        ]
    except requests.RequestException as e:
        return {"error": str(e)}

# Define Shopping Agent
agent = Agent(
    name="Shopping Agent",
    instructions="""
    You are a helpful shopping assistant. Use the product list from the API
    to recommend products based on the user's query. Be friendly and concise.
    """,
    tools=[get_products],
    model=model
)

# Smart Shopping Queries
shopping_queries = [
    "Show me all available products from the store.",
    "What are the newest products available?",
    "Which items are currently offering the biggest discount?",
    "Suggest something elegant for home decor.",
    "Do you have any cozy or comfy furniture recommendations?",
    "Can you show me rustic or vintage pieces for my living room?",
    "I'm looking for a stylish chair under 250. What do you recommend?",
    "Which items are great as birthday gifts?",
]

#Run Shopping Agent
for query in shopping_queries:
    rich.print(f"\n[b cyan]User Prompt:[/b cyan] {query}")
    result = Runner.run_sync(
        agent,
        input=query,
        run_config=config
    )
    rich.print(f"[yellow]Agent Response:[/yellow] {result.final_output}")
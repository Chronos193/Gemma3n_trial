import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langgraph.graph import StateGraph
from gemma3n_trial.agents.llm_agent import LLM_Agent
from gemma3n_trial.agents.search_agent import SearchAgent
from gemma3n_trial.agents.interface_agent import InterfaceAgent
from gemma3n_trial.agents.recipe_agent import RecipeAgent
from gemma3n_trial.agents import CookingGraphAgent, AgentState
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from gemma3n_trial.schema import DishName, RecipeSearchResult, DetailedRecipe

class PipelineState(BaseModel):
    user_query: str
    dish_name: Optional[str] = None
    recipes: Optional[List[RecipeSearchResult]] = None
    selected_recipe: Optional[RecipeSearchResult] = None
    detailed_recipe: Optional[Dict[str, Any]] = None

# Initialize memory in session_state (for conversation buffer)
if "cooking_agent_memory" not in st.session_state:
    st.session_state.cooking_agent_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output",
        k=2,  # Last 2 turns
    )

# Initialize LLM and API key
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")
spoonacular_api_key = os.getenv("SPOONACULAR_API_KEY")

llm_agent = LLM_Agent(llm)
search_agent = SearchAgent(spoonacular_api_key)
recipe_agent = RecipeAgent(spoonacular_api_key)
cooking_graph_agent = CookingGraphAgent(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)
# Inject persistent memory object
cooking_graph_agent.memory = st.session_state.cooking_agent_memory

def extract_dish_name_node(state: PipelineState) -> dict:
    dish_name_obj = llm_agent.invoke({"user_query": state.user_query})
    return {"dish_name": dish_name_obj.name}

def search_recipes_node(state: PipelineState) -> dict:
    results_obj = search_agent.invoke(state.dish_name)
    return {"recipes": results_obj.results}

def select_recipe_node(state: PipelineState) -> dict:
    user_choice = st.session_state.get("user_choice", 1)
    interface_agent = InterfaceAgent(user_choice)
    cooking_state = {
        "recipe_options": state.recipes,
        "selected_recipe": None
    }
    selected = interface_agent.invoke(cooking_state)
    return {"selected_recipe": selected["selected_recipe"]}

def fetch_detailed_recipe_node(state: PipelineState) -> dict:
    cooking_state = {
        "recipe_options": state.recipes,
        "selected_recipe": state.selected_recipe
    }
    new_state = recipe_agent.invoke(cooking_state)
    detailed_recipe = new_state.get("detailed_recipe")
    return {"detailed_recipe": detailed_recipe}

# Graph Setup
graph = StateGraph(state_schema=PipelineState)
graph.add_node("extract_dish_name", extract_dish_name_node)
graph.add_node("search_recipes", search_recipes_node)
graph.add_node("select_recipe", select_recipe_node)
graph.add_node("fetch_detailed_recipe", fetch_detailed_recipe_node)

graph.add_edge("extract_dish_name", "search_recipes")
graph.add_edge("search_recipes", "select_recipe")
graph.add_edge("select_recipe", "fetch_detailed_recipe")
graph.set_entry_point("extract_dish_name")

app = graph.compile()

# Streamlit UI
st.title("👩‍🍳 Cooking Assistant")

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "recipes" not in st.session_state:
    st.session_state.recipes = []
if "user_choice" not in st.session_state:
    st.session_state.user_choice = 1
if "detailed" not in st.session_state:
    st.session_state.detailed = None
if "selected_recipe" not in st.session_state:
    st.session_state.selected_recipe = None

user_query = st.text_input("📝 What would you like to cook today?", "Butter Chicken")

# First run: extract dish and get recipes
if st.button("Find Recipes"):
    with st.spinner("⏳ Thinking... Finding the best options for you!"):
        partial_graph = StateGraph(state_schema=PipelineState)
        partial_graph.add_node("extract_dish_name", extract_dish_name_node)
        partial_graph.add_node("search_recipes", search_recipes_node)
        partial_graph.add_edge("extract_dish_name", "search_recipes")
        partial_graph.set_entry_point("extract_dish_name")
        partial_app = partial_graph.compile()

        result = partial_app.invoke({"user_query": user_query})
        st.session_state.pipeline_result = result
        st.session_state.recipes = result.get("recipes", [])

# Let user choose recipe after partial run
if st.session_state.recipes:
    st.subheader("Select a Recipe")
    titles = [r.title for r in st.session_state.recipes]
    st.session_state.user_choice = st.selectbox(
        "Choose a recipe", 
        list(range(1, len(titles) + 1)), 
        format_func=lambda x: titles[x-1]
    )

    if st.button("Show Selected Recipe"):
        # Full pipeline continuation with selected recipe
        result = app.invoke({
            "user_query": user_query,
        })
        st.session_state.detailed = result.get("detailed_recipe")
        st.session_state.selected_recipe = result.get("selected_recipe")

# Show detailed recipe
if st.session_state.detailed:
    detailed = st.session_state.detailed

    #st.markdown("### 📖 Here's your detailed recipe")
    #st.markdown(f"**🍲 Title**: {detailed.get('title')}")
    #st.markdown(f"**📝 Summary**: {detailed.get('summary')}")
    #st.markdown(f"**🧑‍🍳 Instructions**:\n{detailed.get('instructions')}")
    #st.markdown(f"**🫒 Ingredients**: {', '.join(detailed.get('ingredients', []))}")
    #st.markdown(f"**⏱️ Ready in**: {detailed.get('readyInMinutes')} minutes")
    #st.markdown(f"**👥 Servings**: {detailed.get('servings')}")

    st.subheader("🤖 Ask Questions About This Recipe")
    if "followup_history" not in st.session_state:
        st.session_state.followup_history = []
    followup_input = st.text_input("💬 Ask a question about the recipe:")
    ask_button = st.button("Ask")

    if ask_button and followup_input:
        detailed_recipe_obj = DetailedRecipe(**detailed)
        agent_state = AgentState(
            detailed_recipe=detailed_recipe_obj,
            user_input=followup_input
        )
        agent_state = cooking_graph_agent.invoke(agent_state)
        st.session_state.followup_history.append(("You", followup_input))
        st.session_state.followup_history.append(("Assistant", agent_state.response))
        st.markdown(f"**🤖 Assistant says**: {agent_state.response}")

    # Show last 2 turns of conversation
    if st.session_state.followup_history:
        st.markdown("#### 🗨️ Conversation History (last 2 turns)")
        for speaker, message in st.session_state.followup_history[-4:]:
            st.markdown(f"**{speaker}:** {message}")
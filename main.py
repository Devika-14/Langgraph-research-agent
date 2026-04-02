import operator
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

# Load environment variables
load_dotenv()

# --- State Definition ---
class AgentState(TypedDict):
    topic: str
    draft: str
    critique: str
    missing_info: str
    search_queries: List[str]
    source_materials: Annotated[List[str], operator.add]
    iteration: int
    quality_threshold_met: bool
    quality_score: int  # NEW: 0-10 score from critique


# --- Tool Setup ---
tavily_tool = TavilySearchResults(max_results=3)

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# --- Graph Nodes ---
def research_initial(state: AgentState):
    """Searches the web for the initial topic."""
    topic = state["topic"]
    print(f"--- RESEARCH INITIAL: {topic} ---")
    
    results = tavily_tool.invoke({"query": topic})
    formatted_results = "\n".join([f"Source: {r['url']}\nContent: {r['content']}" for r in results])
    
    return {"source_materials": [formatted_results]}


def draft(state: AgentState):
    """Drafts the initial report."""
    print("--- DRAFTING ---")
    topic = state["topic"]
    materials = "\n\n".join(state["source_materials"])
    
    prompt = f"""Write a comprehensive report on the topic '{topic}' based on the following source materials.
    Source Materials:
    {materials}
    
    Report:"""
    
    response = llm.invoke(prompt)
    return {"draft": response.content, "iteration": 1}


def critique(state: AgentState):
    """Critiques the draft, assigns a quality score, and identifies missing information."""
    print("--- CRITIQUING ---")
    draft = state["draft"]
    topic = state["topic"]
    
    prompt = f"""You are an expert evaluator. Review the following report on the topic '{topic}'.
    Identify any major gaps, inaccuracies, or missing information.
    
    Report:
    {draft}
    
    Provide your output in the following JSON format:
    {{
      "critique": "Your detailed critique of the report",
      "missing_info": "Specific details that are missing and need to be researched",
      "search_queries": ["query 1", "query 2"],
      "quality_score": <integer from 0 to 10 reflecting overall report quality>,
      "quality_threshold_met": true/false (Set to true only if quality_score >= 8 and no major missing info)
    }}
    """
    
    from langchain_core.output_parsers import JsonOutputParser
    parser = JsonOutputParser()
    prompt += "\n" + parser.get_format_instructions()
    
    chain = llm | parser
    try:
        result = chain.invoke(prompt)
        score = result.get("quality_score", 0)
        quality_met = result.get("quality_threshold_met", False)

        # FIX 1: Always print the score before any routing decision
        print(f"  Quality Score: {score}/10 | Threshold Met: {quality_met}")

        return {
            "critique": result.get("critique", ""),
            "missing_info": result.get("missing_info", ""),
            "search_queries": result.get("search_queries", []),
            "quality_score": score,
            "quality_threshold_met": quality_met
        }
    except Exception as e:
        print(f"Error parsing critique output: {e}")
        return {
            "critique": "Failed to parse critique",
            "missing_info": "Unknown",
            "search_queries": [topic + " more details"],
            "quality_score": 0,
            "quality_threshold_met": False
        }


def research_missing(state: AgentState):
    """Searches for the missing information identified in the critique."""
    print("--- RESEARCHING MISSING INFO ---")
    queries = state["search_queries"]
    
    new_materials = []
    for query in queries:
        print(f"  Searching: {query}")
        try:
            results = tavily_tool.invoke({"query": query})
            formatted = "\n".join([f"Source: {r['url']}\nContent: {r['content']}" for r in results])
            new_materials.append(f"Query: {query}\n" + formatted)
        except Exception as e:
            print(f"Search failed for query '{query}': {e}")
            
    return {"source_materials": new_materials}


def improve(state: AgentState):
    """Improves the draft based on the critique and new source materials."""
    print("--- IMPROVING ---")
    draft = state["draft"]
    topic = state["topic"]
    critique = state["critique"]
    materials = "\n\n".join(state["source_materials"])
    iteration = state.get("iteration", 0)
    
    prompt = f"""You are an expert writer and researcher. Improve the following draft on the topic '{topic}'.
    
    Previous Draft:
    {draft}
    
    Critique of Previous Draft:
    {critique}
    
    All Source Materials (Old and New):
    {materials}
    
    Rewrite the report to address the critique completely and incorporate any new relevant information from the source materials. Make sure the flow is coherent.
    
    Improved Report:"""
    
    response = llm.invoke(prompt)
    # FIX 2: increment iteration here only, not in draft node too
    return {"draft": response.content, "iteration": iteration + 1}


# --- Routing/Edges ---
def should_continue(state: AgentState):
    """Determines the next node based on the critique score and iteration count."""
    quality_met = state.get("quality_threshold_met", False)
    iteration = state.get("iteration", 1)
    score = state.get("quality_score", 0)

    print(f"  Routing Decision — Score: {score}/10 | Quality Met: {quality_met} | Iteration: {iteration}")

    # FIX 3: cap at 3 improvement iterations (not counting the initial draft)
    if quality_met or iteration >= 3:
        return "human_review"
    else:
        return "research_missing"


# --- Build the Graph ---
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

builder = StateGraph(AgentState)

builder.add_node("research_initial", research_initial)
builder.add_node("draft", draft)
builder.add_node("critique", critique)
builder.add_node("research_missing", research_missing)
builder.add_node("improve", improve)

def human_review(state: AgentState):
    print("--- HUMAN REVIEW NODE EXECUTED ---")
    return state

builder.add_node("human_review", human_review)

builder.add_edge(START, "research_initial")
builder.add_edge("research_initial", "draft")
builder.add_edge("draft", "critique")

builder.add_conditional_edges(
    "critique",
    should_continue,
    {
        "human_review": "human_review",
        "research_missing": "research_missing"
    }
)

builder.add_edge("research_missing", "improve")
builder.add_edge("improve", "critique")
builder.add_edge("human_review", END)

graph = builder.compile(checkpointer=memory, interrupt_before=["human_review"])


# --- CLI Execution Script ---
if __name__ == "__main__":
    import uuid
    import sys
    
    print("Welcome to the LangGraph Reflection Researcher!")

    google_key = os.environ.get("GOOGLE_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    
    if not google_key or google_key == "your_google_api_key_here":
        print("ERROR: GOOGLE_API_KEY is not set correctly in your environment or .env file.")
        sys.exit(1)
        
    if not tavily_key or tavily_key == "your_tavily_api_key_here":
        print("ERROR: TAVILY_API_KEY is not set correctly in your environment or .env file.")
        sys.exit(1)
        
    if len(sys.argv) > 1:
        user_topic = sys.argv[1]
    else:
        user_topic = input("\nEnter the topic you want to research: ")
    
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    initial_state = {
        "topic": user_topic,
        "source_materials": [],
        "iteration": 0,
        "quality_score": 0
    }
    
    print("\nStarting the Agent...\n")
    
    for event in graph.stream(initial_state, thread, stream_mode="values"):
        if "draft" in event:
            print("\n>> Draft updated.\n")
            
    print("\n--- GRAPH PAUSED FOR HUMAN REVIEW ---")
    
    current_state = graph.get_state(thread).values
    print("\nCURRENT DRAFT:")
    print("="*40)
    print(current_state.get("draft", "No draft created."))
    print("="*40)
    
    print(f"\nIterations taken: {current_state.get('iteration')}")
    print(f"Final Quality Score: {current_state.get('quality_score', 'N/A')}/10")
    print(f"Critique from last loop: {current_state.get('critique')}")
    
    print("\nWhat would you like to do?")
    print("1. Accept draft and Finish (Approve)")
    print("2. Provide feedback to the agent to improve it")
    print("3. Let it run again (if max iterations not reached)")
    
    # FIX 4: actually wait for user input instead of hardcoding '1'
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == '1':
        print("Approving draft. Resuming graph...")
        for event in graph.stream(None, thread, stream_mode="values"):
            pass 
        print("Workflow Completed Successfully.")
        
    elif choice == '2':
        feedback = input("Enter your specific feedback for the agent: ")
        
        print("Injecting feedback and routing back to 'improve'...")
        
        new_critique = f"HUMAN FEEDBACK: {feedback}\n\nPrevious Bot Critique: {current_state.get('critique')}"
        
        graph.update_state(
            thread,
            {
                "critique": new_critique, 
                "quality_threshold_met": False,
                "quality_score": 0,
                "iteration": max(0, current_state.get("iteration", 1) - 1)
            },
            as_node="critique"
        )
        
        print("\nResuming agent with human feedback...\n")
        for event in graph.stream(None, thread, stream_mode="values"):
            if "draft" in event:
                print("\n>> Draft updated.\n")
                
        final_state = graph.get_state(thread).values
        print("\nFINAL DRAFT:")
        print("="*40)
        print(final_state.get("draft"))
        print("="*40)
        
    else:
        print("Resuming graph normally...")
        for event in graph.stream(None, thread, stream_mode="values"):
            pass
        print("Workflow Completed.")
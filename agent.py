"""
Smart Playlist Curator Agent with LangChain and Groq
Creates personalized music playlists based on mood, activity, and preferences
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime
import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# STEP 1: Define Custom Tools for Music Playlist
# =============================================================================

@tool
def calculate_playlist_duration(num_songs: int, avg_song_duration: float = 3.5) -> str:
    """
    Calculate total playlist duration based on number of songs.
    
    Args:
        num_songs: Number of songs in the playlist
        avg_song_duration: Average song duration in minutes (default 3.5)
    """
    try:
        total_minutes = num_songs * avg_song_duration
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        
        if hours > 0:
            return f"{hours} hour(s) and {minutes} minute(s)"
        else:
            return f"{minutes} minute(s)"
    except Exception as e:
        return f"Error calculating duration: {str(e)}"

@tool
def get_mood_music_keywords(mood: str, activity: str) -> str:
    """
    Generate search keywords for music based on mood and activity.
    
    Args:
        mood: User's current mood (happy, sad, energetic, calm, etc.)
        activity: Current activity (workout, study, party, sleep, etc.)
    """
    try:
        mood = mood.lower().strip()
        activity = activity.lower().strip()
        
        # Define mood-based music characteristics
        mood_mapping = {
            "happy": "upbeat cheerful positive feel-good",
            "sad": "melancholic emotional ballad slow",
            "energetic": "high-energy fast-paced pumped powerful",
            "calm": "relaxing peaceful ambient chill mellow",
            "romantic": "love romantic intimate soft",
            "angry": "aggressive intense heavy powerful",
            "focused": "instrumental concentration deep-focus",
            "nostalgic": "throwback classic retro memories"
        }
        
        # Define activity-based music characteristics
        activity_mapping = {
            "workout": "gym training fitness motivation high-tempo",
            "study": "concentration focus instrumental background",
            "party": "dance celebration upbeat crowd-pleaser",
            "sleep": "lullaby sleep ambient soft gentle",
            "driving": "road-trip driving cruising",
            "cooking": "upbeat background feel-good casual",
            "meditation": "zen mindfulness ambient peaceful",
            "work": "productivity focus instrumental background"
        }
        
        mood_keywords = mood_mapping.get(mood, mood)
        activity_keywords = activity_mapping.get(activity, activity)
        
        return f"{mood_keywords} {activity_keywords} music songs playlist"
        
    except Exception as e:
        return f"Error generating keywords: {str(e)}"

@tool
def suggest_song_count(duration_minutes: int) -> str:
    """
    Suggest number of songs needed for requested duration.
    
    Args:
        duration_minutes: Desired playlist duration in minutes
    """
    try:
        avg_song_duration = 3.5  # minutes
        suggested_songs = int(duration_minutes / avg_song_duration)
        
        return f"For {duration_minutes} minutes, suggest approximately {suggested_songs} songs"
    except Exception as e:
        return f"Error calculating song count: {str(e)}"

@tool
def format_playlist_output(playlist_data: str) -> str:
    """
    Format playlist information into a structured output.
    
    Args:
        playlist_data: Raw playlist information as string
    """
    try:
        # This tool helps structure the final output
        return f"Formatted Playlist:\n{playlist_data}"
    except Exception as e:
        return f"Error formatting playlist: {str(e)}"

# =============================================================================
# STEP 2: Initialize Chat History
# =============================================================================

chat_history = []

# =============================================================================
# STEP 3: Create the Agent
# =============================================================================

def create_agent():
    """Initialize and return the playlist curator agent executor."""
    
    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=2048,
        timeout=30,
        max_retries=2
    )
    
    # Initialize Tavily search tool
    tavily_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False
    )
    
    # Define all tools
    tools = [
        calculate_playlist_duration,
        get_mood_music_keywords,
        suggest_song_count,
        format_playlist_output,
        tavily_tool
    ]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Smart Playlist Curator AI assistant specializing in creating personalized music playlists.

Your capabilities:
- Create playlists based on mood, activity, duration, and genre preferences
- Use web search to find popular songs, trending tracks, and artist recommendations
- Provide Spotify and YouTube search links for recommended songs
- Explain the flow and logic behind playlist curation
- Calculate accurate playlist durations

When creating a playlist:
1. Use get_mood_music_keywords tool to understand the musical direction
2. Use suggest_song_count tool to determine how many songs are needed
3. Use tavily_search_results_json to find popular songs matching the criteria
4. Search for specific songs with "song name artist Spotify" or "song name artist YouTube"
5. Use calculate_playlist_duration to verify the total duration
6. Present songs in a logical flow (e.g., start calm, build energy, then cool down)

Output format:
- Song list with artist names
- Brief explanation of playlist flow
- Total estimated duration
- Links to Spotify/YouTube when available

Be conversational and remember context from previous messages."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
        max_execution_time=90
    )
    
    return agent_executor

# =============================================================================
# STEP 4: Chat Function with Memory
# =============================================================================

def chat(user_input: str, agent_executor):
    """
    Process user input and maintain chat history.
    
    Args:
        user_input: The user's message
        agent_executor: The agent executor instance
    
    Returns:
        The agent's response
    """
    global chat_history
    
    try:
        if chat_history is None:
            chat_history = []
            
        # Format chat history
        formatted_history = []
        for msg in chat_history:
            if isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role == "human":
                    formatted_history.append(HumanMessage(content=content))
                elif role == "assistant" and content:
                    if isinstance(content, str):
                        formatted_history.append(AIMessage(content=content))
        
        if agent_executor is None:
            agent_executor = create_agent()
        
        # Prepare input
        input_data = {
            "input": user_input,
            "chat_history": formatted_history or []
        }
        
        # Run agent
        try:
            response = agent_executor.invoke(input_data)
            
            if response is None:
                output = "No response was generated. Please try again."
            elif isinstance(response, dict):
                output = response.get('output', '')
                if not output:
                    output = "I didn't get a proper response. Could you rephrase your question?"
            elif hasattr(response, 'output') and response.output is not None:
                output = str(response.output)
            else:
                output = str(response) if response is not None else "No response was generated."
                
            if not output or not isinstance(output, str):
                output = "I'm having trouble understanding. Could you rephrase your question?"
                
        except Exception as e:
            output = f"I encountered an error: {str(e)}. Could you please rephrase your question?"
        
        # Update chat history
        if output and output != 'No response generated':
            chat_history.append(("human", user_input))
            chat_history.append(("assistant", output))
        
        # Keep last 20 messages
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        return output if output else "I'm not sure how to respond to that. Could you rephrase?"
        
    except Exception as e:
        error_msg = f"Error in chat function: {str(e)}"
        print(error_msg)
        return "I'm sorry, I encountered an error processing your request. Please try again."

# =============================================================================
# STEP 5: Main Execution (CLI Interface)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎵 Smart Playlist Curator - AI Music Assistant")
    print("=" * 70)
    print("\n📋 Loading API keys from .env file...")
    
    # Check for API keys
    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠️  GROQ_API_KEY not found in .env file!")
        print("\nPlease create a .env file with:")
        print("GROQ_API_KEY=gsk-your-groq-key-here")
        print("TAVILY_API_KEY=tvly-your-tavily-key-here")
        sys.exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("\n⚠️  TAVILY_API_KEY not found in .env file!")
        print("\nPlease add to your .env file:")
        print("TAVILY_API_KEY=tvly-your-tavily-key-here")
        sys.exit(1)
    
    print("✅ API keys loaded successfully!")
    
    # Initialize agent
    print("\n🎵 Initializing playlist curator agent...")
    try:
        agent_executor = create_agent()
        print("✅ Agent ready!\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize agent: {str(e)}")
        sys.exit(1)
    
    # Interactive chat loop
    print("Chat with the Playlist Curator (type 'quit' to exit):\n")
    print("Example: 'Create a 1 hour workout playlist with energetic pop music'\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 🎵")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! Enjoy your music! 🎵")
            break
        
        if user_input.lower() == 'history':
            print("\n--- Chat History ---")
            if not chat_history:
                print("No chat history yet.")
            else:
                for role, message in chat_history:
                    preview = message[:100] + "..." if len(message) > 100 else message
                    print(f"{role}: {preview}")
            print("--- End History ---\n")
            continue
        
        if user_input.lower() == 'clear':
            chat_history.clear()
            print("\n✅ Chat history cleared!\n")
            continue
        
        try:
            response = chat(user_input, agent_executor)
            print(f"\n🎵 Playlist Curator: {response}\n")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.\n")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
import os
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🎵 Harmony AI - Spotify Curator",
    page_icon="🎵",
    layout="centered"
)

# --- STYLING ---
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1DB954;
        padding: 1rem 0;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
@st.cache_resource
def initialize_spotify():
    """Initialize Spotify client with caching"""
    load_dotenv()
    SCOPE = "playlist-modify-public playlist-modify-private user-library-read"
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPE))
        return sp, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def initialize_agent():
    """Initialize LangChain agent with caching"""
    
    def search_music_by_query(query: str) -> str:
        try:
            results = sp.search(q=query, limit=10, type='track')
            track_uris = [track['uri'] for track in results['tracks']['items']]
            if not track_uris:
                return "No tracks found for the query."
            return ", ".join(track_uris)
        except Exception as e:
            return f"Error during Spotify search: {e}"
    
    def create_and_populate_playlist(name: str, description: str, track_uris_str: str) -> str:
        try:
            user_id = sp.current_user()['id']
            playlist = sp.user_playlist_create(
                user=user_id, 
                name=name, 
                public=True, 
                description=description
            )
            track_uris = [uri.strip() for uri in track_uris_str.split(',') if uri.strip()]
            if track_uris:
                sp.playlist_add_items(playlist_id=playlist['id'], items=track_uris)
            return f"Playlist '{name}' successfully created with {len(track_uris)} songs. Playlist URL: {playlist['external_urls']['spotify']}"
        except Exception as e:
            return f"Error creating or populating playlist: {e}"

    @tool
    def spotify_search(query: str) -> str:
        """Search for songs on Spotify using natural language queries."""
        return search_music_by_query(query)

    @tool
    def spotify_create_playlist(name: str, description: str, track_uris_str: str) -> str:
        """Create a new playlist with specified tracks."""
        return create_and_populate_playlist(name, description, track_uris_str)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = [spotify_search, spotify_create_playlist]
    
    SYSTEM_PROMPT = """
You are 'Harmony AI', a world-class, helpful music curator. Your task is to interpret the user's
mood, activity, or request and use the provided Spotify tools to generate a perfectly curated playlist.

**Your Workflow:**
1.  **Analyze** the user's request (e.g., 'upbeat workout songs', 'sad reflective music').
2.  **Call the `spotify_search` tool first** with a highly specific and effective search query to find relevant tracks.
3.  **Review** the track URIs returned by the search tool.
4.  **Call the `spotify_create_playlist` tool last** with a suggested name and description that matches the user's intent.
    * The `track_uris_str` argument MUST be the comma-separated string of URIs returned by the search tool.

**Crucial Rules:**
* You **must** use the `spotify_search` tool before attempting to create a playlist.
* The final output to the user should be the result from the `spotify_create_playlist` tool, including the playlist URL.
* Do NOT invent song names or URIs. Only use the URIs provided by the search tool.
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    return agent_executor

# --- MAIN APP ---
st.markdown("<h1 class='main-header'>🎵 Harmony AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Your Personal AI Music Curator</p>", unsafe_allow_html=True)

# Initialize Spotify
sp, error = initialize_spotify()

if error:
    st.error(f"❌ Failed to connect to Spotify: {error}")
    st.info("Please ensure your .env file is configured correctly and your Redirect URI is set in the Spotify Dashboard.")
    st.stop()
else:
    st.success("✅ Connected to Spotify")

# Initialize Agent
agent_executor = initialize_agent()

st.markdown("---")

# User Input
st.subheader("What kind of music are you in the mood for?")

# Preset examples
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏃 Workout Energy"):
        st.session_state.user_input = "High energy workout music to keep me pumped"
with col2:
    if st.button("😌 Chill Vibes"):
        st.session_state.user_input = "Relaxing chill music for studying or unwinding"
with col3:
    if st.button("🎹 Focus Flow"):
        st.session_state.user_input = "Instrumental focus music for deep work"

# Text input
user_request = st.text_area(
    "Describe your perfect playlist:",
    value=st.session_state.get('user_input', ''),
    height=100,
    placeholder="e.g., 'upbeat 80s pop songs' or 'sad reflective indie music' or 'party dance hits'"
)

# Generate button
if st.button("🎧 Create My Playlist", type="primary"):
    if not user_request.strip():
        st.warning("Please describe the kind of music you want!")
    else:
        with st.spinner("🎵 Harmony AI is curating your perfect playlist..."):
            try:
                result = agent_executor.invoke({"input": user_request})
                st.success("✨ Playlist Created!")
                st.markdown(f"### {result['output']}")
                
                # Extract URL if present
                if "spotify.com" in result['output']:
                    url_start = result['output'].find("https://")
                    if url_start != -1:
                        url = result['output'][url_start:].split()[0]
                        st.markdown(f"[🎵 Open in Spotify]({url})")
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 0.9em;'>Powered by OpenAI & Spotify API</p>", unsafe_allow_html=True)
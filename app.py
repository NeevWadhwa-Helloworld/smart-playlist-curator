"""
Smart Playlist Curator - Streamlit UI
Beautiful chat interface for creating personalized music playlists
"""

import streamlit as st
import uuid
import time
import traceback
from typing import List, Dict
import streamlit.components.v1 as components

# Import agent utilities
try:
    from agent import create_agent, chat as agent_chat
except Exception as imp_err:
    create_agent = None
    agent_chat = None
    IMPORT_ERROR = imp_err
else:
    IMPORT_ERROR = None

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="🎵 Smart Playlist Curator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# Custom CSS Styling
# =============================================================================

st.markdown(
    """
    <style>
    /* Main container */
    .app-container {
        max-width: 1100px;
        margin: 10px auto;
    }
    
    /* Header styling */
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    
    .title {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        color: #7b8794;
        font-size: 13px;
        margin-top: 4px;
    }

    /* Chatbox styling */
    #chatbox {
        border-radius: 16px;
        padding: 20px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.04));
        box-shadow: 0 8px 32px rgba(2,6,23,0.08);
        max-height: 65vh;
        overflow-y: auto;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Message bubbles */
    .msg-user {
        margin-left: auto;
        margin-bottom: 14px;
        padding: 14px 16px;
        border-radius: 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        max-width: 75%;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        word-wrap: break-word;
        font-size: 15px;
    }
    
    .msg-assistant {
        margin-right: auto;
        margin-bottom: 14px;
        padding: 14px 16px;
        border-radius: 16px;
        background: linear-gradient(180deg, #1a1f2e, #252b3b);
        color: #e6eef8;
        max-width: 75%;
        box-shadow: 0 6px 20px rgba(2,6,23,0.15);
        word-wrap: break-word;
        font-size: 15px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .meta {
        font-size: 11px;
        color: rgba(255,255,255,0.6);
        margin-top: 6px;
        font-weight: 500;
    }

    /* Quick action buttons */
    .quick-actions {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 16px;
    }
    
    .quick-btn {
        padding: 8px 14px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        color: #667eea;
        border: 1px solid rgba(102, 126, 234, 0.3);
        font-size: 13px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .quick-btn:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        transform: translateY(-2px);
    }

    /* Input area */
    .input-info {
        color: #98a0a6;
        font-size: 12px;
        margin-top: 8px;
        text-align: center;
    }

    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-info {
        background: rgba(59, 130, 246, 0.1);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Responsive design */
    @media (max-width: 800px) {
        #chatbox {
            max-height: 55vh;
            padding: 14px;
        }
        .msg-user, .msg-assistant {
            max-width: 90%;
            font-size: 14px;
        }
        .title {
            font-size: 22px;
        }
    }

    /* Scrollbar styling */
    #chatbox::-webkit-scrollbar {
        width: 8px;
    }
    
    #chatbox::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
    }
    
    #chatbox::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.3);
        border-radius: 10px;
    }
    
    #chatbox::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Agent Initialization
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_agent_executor_cached():
    """Cache the agent executor to avoid reinitializing on every interaction."""
    if create_agent is None:
        raise RuntimeError(f"agent.create_agent import failed: {IMPORT_ERROR}")
    return create_agent()

def ensure_agent_ready():
    """Ensure agent is initialized and ready."""
    if st.session_state.get("agent_ready") and st.session_state.get("agent_executor"):
        return st.session_state["agent_executor"]
    agent_exec = get_agent_executor_cached()
    st.session_state["agent_executor"] = agent_exec
    st.session_state["agent_ready"] = True
    return agent_exec

# =============================================================================
# Session State Initialization
# =============================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False

if "last_error" not in st.session_state:
    st.session_state.last_error = None

# =============================================================================
# Header Section
# =============================================================================

st.markdown('<div class="app-container">', unsafe_allow_html=True)

col_left, col_right = st.columns([5, 1])
with col_left:
    st.markdown(
        '''
        <div class="header">
            <div>
                <div class="title">🎵 Smart Playlist Curator</div>
                <div class="subtitle">AI-powered personalized music playlists • Powered by LangChain & Groq</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

with col_right:
    if st.session_state.agent_ready:
        st.markdown('<span class="status-badge status-success">● Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-info">○ Starting</span>', unsafe_allow_html=True)

st.divider()

# =============================================================================
# Quick Action Buttons
# =============================================================================

st.markdown('<div class="quick-actions">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

quick_prompts = {
    "🏋️ Workout": "Create a 1 hour energetic workout playlist with pop and EDM",
    "📚 Study": "Create a 2 hour calm study playlist with instrumental focus music",
    "🎉 Party": "Create a 1.5 hour party playlist with upbeat dance hits",
    "😴 Sleep": "Create a 30 minute relaxing sleep playlist with ambient sounds"
}

for idx, (label, prompt) in enumerate(quick_prompts.items()):
    col = [col1, col2, col3, col4][idx]
    with col:
        if st.button(label, key=f"quick_{idx}", use_container_width=True):
            st.session_state.quick_prompt = prompt

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# Initialize Agent
# =============================================================================

if not st.session_state.agent_ready:
    try:
        with st.spinner("🎵 Initializing playlist curator..."):
            ensure_agent_ready()
            st.session_state.agent_ready = True
            st.rerun()
    except Exception as e:
        st.session_state.last_error = traceback.format_exc()
        st.error("❌ Agent initialization failed. Check the error details below.")
        with st.expander("Error Details"):
            st.code(st.session_state.last_error)
        st.stop()

# =============================================================================
# Chat Rendering Function
# =============================================================================

def render_chat_html(messages):
    """Render chat messages as HTML."""
    html_parts = ["<div id='chatbox'>"]
    
    if not messages:
        html_parts.append(
            "<div style='text-align:center; color:#7b8794; padding:40px;'>"
            "👋 Welcome! Tell me your mood, activity, and preferred duration.<br>"
            "I'll create the perfect playlist for you!<br><br>"
            "Example: <i>'Create a 1 hour energetic workout playlist'</i>"
            "</div>"
        )
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        ts = msg.get("ts", time.time())
        ts_str = time.strftime("%H:%M", time.localtime(ts))
        safe_content = content.replace("\n", "<br>")
        
        if role == "user":
            html_parts.append(
                f"<div style='display:flex; justify-content:flex-end;'>"
                f"<div class='msg-user'>{safe_content}<div class='meta'>You • {ts_str}</div></div>"
                f"</div>"
            )
        else:
            html_parts.append(
                f"<div style='display:flex; justify-content:flex-start;'>"
                f"<div class='msg-assistant'>{safe_content}<div class='meta'>Curator • {ts_str}</div></div>"
                f"</div>"
            )
    
    html_parts.append("</div>")
    return "\n".join(html_parts)

# =============================================================================
# Chat Display
# =============================================================================

chatbox_placeholder = st.empty()
chatbox_placeholder.markdown(render_chat_html(st.session_state.messages), unsafe_allow_html=True)

# =============================================================================
# Input Area
# =============================================================================

# Check if quick prompt was clicked
quick_prompt_text = st.session_state.get("quick_prompt", "")
if quick_prompt_text:
    default_text = quick_prompt_text
    st.session_state.quick_prompt = ""  # Clear after using
else:
    default_text = ""

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Your request:",
        key="user_input",
        height=100,
        placeholder="Example: Create a 1 hour energetic workout playlist with pop music...",
        value=default_text
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="input-info">💡 Specify mood, activity, duration, and genre preferences</div>', unsafe_allow_html=True)
    with col2:
        submit = st.form_submit_button("🎵 Create Playlist", use_container_width=True)

# =============================================================================
# Handle User Input
# =============================================================================

if submit and user_input and user_input.strip():
    text = user_input.strip()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": text, "ts": time.time()})
    chatbox_placeholder.markdown(render_chat_html(st.session_state.messages), unsafe_allow_html=True)
    
    # Add thinking placeholder
    st.session_state.messages.append({"role": "assistant", "content": "🎵 Creating your perfect playlist...", "ts": time.time()})
    chatbox_placeholder.markdown(render_chat_html(st.session_state.messages), unsafe_allow_html=True)
    
    # Get agent response
    try:
        agent_exec = st.session_state.get("agent_executor") or ensure_agent_ready()
        start = time.time()
        
        try:
            response = agent_chat(text, agent_exec)
            if response is None:
                response = "I couldn't create a playlist. Please try again with more details."
            elif not isinstance(response, str):
                response = str(response)
        except Exception as e:
            response = f"I encountered an error: {str(e)}. Please try again."
        
        # Update assistant message
        for i in range(len(st.session_state.messages) - 1, -1, -1):
            if st.session_state.messages[i]["role"] == "assistant":
                st.session_state.messages[i]["content"] = response
                st.session_state.messages[i]["ts"] = time.time()
                break
        
        chatbox_placeholder.markdown(render_chat_html(st.session_state.messages), unsafe_allow_html=True)
        
        elapsed = time.time() - start
        st.markdown(
            f"<div style='text-align:right; color:#98a0a6; font-size:12px; margin-top:8px;'>⚡ Response time: {elapsed:.2f}s</div>",
            unsafe_allow_html=True
        )
        
    except Exception as e:
        tb = traceback.format_exc()
        st.session_state.last_error = tb
        
        for i in range(len(st.session_state.messages) - 1, -1, -1):
            if st.session_state.messages[i]["role"] == "assistant":
                st.session_state.messages[i]["content"] = "⚠️ Failed to create playlist. Please try again."
                st.session_state.messages[i]["ts"] = time.time()
                break
        
        chatbox_placeholder.markdown(render_chat_html(st.session_state.messages), unsafe_allow_html=True)
        
        with st.expander("Error Details"):
            st.code(tb)
    
    # Auto-scroll to bottom
    components.html(
        """
        <script>
        const cb = document.getElementById('chatbox');
        if (cb) { cb.scrollTop = cb.scrollHeight; }
        </script>
        """,
        height=0
    )

# =============================================================================
# Auto-scroll on page load
# =============================================================================

components.html(
    """
    <script>
    const cb = document.getElementById('chatbox');
    if (cb) { cb.scrollTop = cb.scrollHeight; }
    </script>
    """,
    height=0
)

# =============================================================================
# Sidebar (optional history/controls)
# =============================================================================

with st.sidebar:
    st.markdown("### 🎵 Playlist Curator")
    st.markdown(f"**Session:** `{st.session_state.session_id[:8]}`")
    st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    
    st.divider()
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Specify mood (happy, energetic, calm, sad)
    - Mention activity (workout, study, party, sleep)
    - Set duration (30 min, 1 hour, 2 hours)
    - Add genre preferences (pop, rock, EDM, jazz)
    """)
    
    if st.session_state.last_error:
        st.divider()
        with st.expander("⚠️ Last Error"):
            st.code(st.session_state.last_error)

st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# Footer
# =============================================================================

st.markdown(
    "<div style='color:#98a0a6; margin-top:20px; font-size:12px; text-align:center;'>"
    "🔑 API keys stored in .env file • Built with LangChain, Groq & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
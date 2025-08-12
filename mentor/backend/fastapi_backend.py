import os
import sys
import json
import datetime
import uuid
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple

# Adjusting system path to locate necessary modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'storage')))

# Importing custom modules for database operations and the core engine
from shared.storage.handle_user import validate_login
from shared.storage.handle_mentor_chat_history import (
    save_chat,
    get_chats,
    get_chat_messages_with_state,
    save_user_preferences,
    get_user_preferences,
    init_db
)
from mentor.core.engine.mentor_engine import MentorEngine

# Initialize the FastAPI application
app = FastAPI()

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initializes the database when the application starts."""
    init_db()
    print("Database initialized.")

# --- CORS Middleware ---
# Allows cross-origin requests, essential for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Instantiate the core mentor engine
engine = MentorEngine()

# --- Pydantic Models for Request Bodies ---
class LoginRequest(BaseModel):
    user_id: str
    password: str

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[float] = None
    audio_url: Optional[str] = None

class ChatRequest(BaseModel):
    user_id: str
    chat_title: str
    chat_history: List[ChatMessage]

class StartSessionRequest(BaseModel):
    user_id: str
    learning_goal: Optional[str] = None
    skills: List[str]
    difficulty: str
    role: str

class TopicPromptRequest(BaseModel):
    topic: str
    user_id: Optional[str] = None

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Mentora Me API is running!"}

@app.post("/login")
async def login(req: LoginRequest):
    """Handles user login and validation."""
    print(f"-> /login called with user_id={req.user_id}")
    try:
        valid = validate_login(req.user_id, req.password)
        if not valid:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"success": True, "user_id": req.user_id}
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/start_session")
async def start_session(req: StartSessionRequest):
    """Starts a new learning session for a user."""
    print(f"-> Starting session for user: {req.user_id}")
    try:
        # Save user preferences for the session
        save_user_preferences(
            user_id=req.user_id,
            learning_goal=req.learning_goal,
            skills=req.skills,
            difficulty=req.difficulty,
            role=req.role
        )
        print(f"Saved general preferences for user {req.user_id}")

        # Build context for the mentor engine
        context = (
            f"Skills/Interests: {', '.join(req.skills)}\n"
            f"Difficulty: {req.difficulty}\n"
            f"User Role: {req.role}"
        )
        if req.learning_goal:
            context = f"Learning Goal: {req.learning_goal}\n" + context

        extra_instructions = (
            "You are a mentor who is very interactive and strict to particular domain. if someone asked something which is not related to that domain. give fallback answer. ask questions, quiz the user, "
            "summarize lessons, and check understanding."
        )

        # Generate introductory message and topics from the engine
        intro, topics, suggestions = await engine.generate_intro_and_topics(
            context_description=context,
            extra_instructions=extra_instructions
        )

        initial_current_topic = topics[0] if topics else None
        
        # Create a unique session title
        base_title_part = req.learning_goal or (req.skills[0] if req.skills else "New Session")
        sanitized_base_title = "".join(c for c in base_title_part if c.isalnum() or c == ' ').strip().replace(' ', '_')
        if not sanitized_base_title:
            sanitized_base_title = "Session"
        session_title = f"{sanitized_base_title}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:4]}"

        mentor_message_content = intro + "\n\nFeel free to ask questions anytime. Are you ready to begin?"
        mentor_message_content = mentor_message_content.replace("🔊", "").strip()

        # Create the initial message for the chat history
        initial_chat_history_entry = ChatMessage(
            role="assistant",
            content=mentor_message_content,
            timestamp=datetime.datetime.now().timestamp(),
            audio_url=None
        )

        # Save the new chat session to the database
        save_chat(
            user_id=req.user_id,
            title=session_title,
            messages_json=json.dumps([initial_chat_history_entry.dict()]),
            mentor_topics=topics,
            current_topic=initial_current_topic,
            completed_topics=[]
        )

        print(f"Session started successfully with title: {session_title}")
        # Return all necessary information to the frontend
        return {
            "intro_and_topics": mentor_message_content,
            "title": session_title,
            "topics": topics,
            "current_topic": initial_current_topic,
            "suggestions": suggestions
        }
    except Exception as e:
        print(f"X Error starting session: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not start session: {str(e)}")

@app.post("/chat")
async def chat(req: ChatRequest):
    """Handles an ongoing chat conversation."""
    print(f"-> /chat called by user: {req.user_id} for chat: '{req.chat_title}'")
    try:
        # Retrieve current chat state from the database
        result = get_chat_messages_with_state(req.user_id, req.chat_title)
        if result is None or not isinstance(result, tuple) or len(result) != 2:
            print("get_chat_messages_with_state returned None or unexpected format!")
            chat_messages, state = [], {}
        else:
            chat_messages, state = result

        # Extract session state and user preferences
        mentor_topics = state.get("mentor_topics", [])
        current_topic = state.get("current_topic")
        completed_topics = state.get("completed_topics", [])
        prefs = get_user_preferences(req.user_id)
        learning_goal = prefs.get("learning_goal")
        skills = prefs.get("skills", [])
        difficulty = prefs.get("difficulty", "medium")
        role = prefs.get("role", "student")

        # Pass all necessary context, including chat_title, to the engine
        reply, suggestions = await engine.chat(
            chat_history=[msg.dict() for msg in req.chat_history],
            user_id=req.user_id,
            chat_title=req.chat_title,
            learning_goal=learning_goal,
            skills=skills,
            difficulty=difficulty,
            role=role,
            mentor_topics=mentor_topics,
            current_topic=current_topic,
            completed_topics=completed_topics
        )

        # Create the new assistant message object
        mentor_message = ChatMessage(
            role="assistant",
            content=reply,
            timestamp=datetime.datetime.now().timestamp(),
            audio_url=None
        )

        # Update and save the chat history
        updated_history = req.chat_history + [mentor_message]
        save_chat(
            user_id=req.user_id,
            title=req.chat_title,
            messages_json=json.dumps([msg.dict() for msg in updated_history]),
            mentor_topics=mentor_topics,
            current_topic=current_topic,
            completed_topics=completed_topics
        )

        # Return both reply and suggestions to the frontend
        return {"reply": reply, "suggestions": suggestions}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail="Chat failed")

@app.get("/get_chats")
async def list_chats(user_id: str = Query(..., description="User ID")):
    """Retrieves a list of all past chat sessions for a user."""
    print(f"-> /get_chats called for user_id='{user_id}'")
    try:
        chats = get_chats(user_id)
        return {"chats": chats}
    except Exception as e:
        print(f"Error getting chats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chats")

@app.get("/get_chat_messages")
async def get_chat_messages_route(user_id: str = Query(..., description="User ID"), title: str = Query(..., description="Chat Title")):
    """Retrieves all messages and state for a specific chat session."""
    print(f"-> /get_chat_messages called with user_id='{user_id}', title='{title}'")
    try:
        result = get_chat_messages_with_state(user_id, title)
        if result is None or not isinstance(result, tuple) or len(result) != 2:
            print("get_chat_messages_with_state returned None or unexpected format!")
            messages, state = [], {}
        else:
            messages, state = result
        return {"messages": messages, "state": state}
    except Exception as e:
        print(f"Error getting chat messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chat messages")

@app.post("/get_topic_prompts")
async def get_topic_prompts(req: TopicPromptRequest):
    """Generates suggested user prompts for a given topic."""
    try:
        prefs = get_user_preferences(req.user_id) if req.user_id else {}
        context = ""
        if prefs:
            context = f"Learning Goal: {prefs.get('learning_goal','')}\nSkills: {', '.join(prefs.get('skills',[]))}\nDifficulty: {prefs.get('difficulty','')}\nRole: {prefs.get('role','')}"
        prompts = await engine.generate_topic_prompts(req.topic, context_description=context)
        return {"prompts": prompts}
    except Exception as e:
        # Fallback prompts in case of an error
        return {"prompts": [
            f"What are the basics of {req.topic}?",
            f"Can you give me a real-world example of {req.topic}?",
            f"How do I apply {req.topic} in practice?",
            f"What are common mistakes in {req.topic}?"
        ]}

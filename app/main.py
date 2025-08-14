from __future__ import annotations
import datetime as dt, json, uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.engine.mentor_engine import MentorEngine
from utils.handle_user import validate_login
from utils.handle_mentor_chat_history import init_db, save_chat, get_chats, get_chat_messages_with_state, save_user_preferences, get_user_preferences

app = FastAPI(title="Mentora API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

engine = MentorEngine()

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

@app.on_event("startup")
async def on_startup():
    init_db()

@app.get("/")
async def root():
    return {"message": "Mentora API is running"}

@app.post("/login")
async def login(req: LoginRequest):
    try:
        if not validate_login(req.user_id, req.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"success": True, "user_id": req.user_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {e}")

@app.post("/start_session")
async def start_session(req: StartSessionRequest):
    try:
        save_user_preferences(user_id=req.user_id, learning_goal=req.learning_goal, skills=req.skills, difficulty=req.difficulty, role=req.role)
        context = []
        if req.learning_goal: context.append(f"Learning Goal: {req.learning_goal}")
        context.append(f"Skills/Interests: {', '.join(req.skills)}")
        context.append(f"Difficulty: {req.difficulty}")
        context.append(f"User Role: {req.role}")
        context_str = "\n".join(context)
        extra = "You are a mentor who is very interactive and strict to particular domain. If someone asks something not related to that domain, give a polite fallback. Ask questions, quiz the user, summarize lessons, and check understanding."
        intro, topics, suggestions = await engine.generate_intro_and_topics(context_description=context_str, extra_instructions=extra)
        current_topic = topics[0] if topics else None
        base_title_part = req.learning_goal or (req.skills[0] if req.skills else "Session")
        safe = ''.join(c for c in base_title_part if c.isalnum() or c == ' ').strip().replace(' ', '_') or "Session"
        session_title = f"{safe}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:4]}"
        mentor_message = ChatMessage(role="assistant", content=(intro + "\n\nFeel free to ask questions anytime. Are you ready to begin?").replace("🔊","").strip(), timestamp=dt.datetime.now().timestamp(), audio_url=None)
        save_chat(user_id=req.user_id, title=session_title, messages_json=json.dumps([mentor_message.model_dump()]), mentor_topics=topics, current_topic=current_topic, completed_topics=[])
        return {"intro_and_topics": mentor_message.content, "title": session_title, "topics": topics, "current_topic": current_topic, "suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {e}")

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        result = get_chat_messages_with_state(req.user_id, req.chat_title)
        if result is None: chat_messages, state = [], {}
        else: chat_messages, state = result
        mentor_topics = state.get("mentor_topics", [])
        current_topic = state.get("current_topic")
        completed_topics = state.get("completed_topics", [])
        prefs = get_user_preferences(req.user_id) or {}
        learning_goal = prefs.get("learning_goal")
        skills = prefs.get("skills", [])
        difficulty = prefs.get("difficulty", "medium")
        role = prefs.get("role", "default")
        reply, suggestions = await engine.chat(chat_history=[m.model_dump() for m in req.chat_history], user_id=req.user_id, chat_title=req.chat_title, learning_goal=learning_goal, skills=skills, difficulty=difficulty, role=role, mentor_topics=mentor_topics, current_topic=current_topic, completed_topics=completed_topics)
        mentor_message = ChatMessage(role="assistant", content=reply, timestamp=dt.datetime.now().timestamp(), audio_url=None)
        updated = req.chat_history + [mentor_message]
        save_chat(user_id=req.user_id, title=req.chat_title, messages_json=json.dumps([m.model_dump() for m in updated]), mentor_topics=mentor_topics, current_topic=current_topic, completed_topics=completed_topics)
        return {"reply": reply, "suggestions": suggestions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

@app.get("/get_chats")
async def list_chats(user_id: str = Query(..., description="User ID")):
    try:
        return {"chats": get_chats(user_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chats: {e}")

@app.get("/get_chat_messages")
async def get_chat_messages_route(user_id: str = Query(..., description="User ID"), title: str = Query(..., description="Chat Title")):
    try:
        result = get_chat_messages_with_state(user_id, title)
        if result is None: messages, state = [], {}
        else: messages, state = result
        return {"messages": messages, "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat messages: {e}")

@app.post("/get_topic_prompts")
async def get_topic_prompts(req: TopicPromptRequest):
    try:
        prefs = get_user_preferences(req.user_id) if req.user_id else {}
        context = ""
        if prefs:
            context = f"Learning Goal: {prefs.get('learning_goal','')}\nSkills: {', '.join(prefs.get('skills',[]))}\nDifficulty: {prefs.get('difficulty','')}\nRole: {prefs.get('role','')}"
        prompts = await engine.generate_topic_prompts(req.topic, context_description=context)
        return {"prompts": prompts}
    except Exception:
        return {"prompts": [f"What are the basics of {req.topic}?", f"Can you give me a real-world example of {req.topic}?", f"How do I apply {req.topic} in practice?", f"What are common mistakes in {req.topic}?"]}
from __future__ import annotations
import json, logging, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from core.connection import Connection

log = logging.getLogger("mentora.engine")
logging.basicConfig(
    level=os.getenv("MENTORA_LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

class MentorEngine:
    def __init__(self) -> None:
        log.debug("Initializing MentorEngine...")
        self.conn = Connection()
        self.llm = self.conn.client
        self.model_name = self.conn.deployment_name
        self.prompts = self._load_yaml(Path(__file__).with_name("prompts.yaml"))
        self.conversation_summaries: Dict[str, str] = {}
        log.info("MentorEngine ready (model=%s)", self.model_name)

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        log.debug("Loading prompts from %s", path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            log.debug("Prompts loaded successfully")
            return data
        except Exception as e:
            log.warning("Failed to load prompts.yaml: %s", e)
            return {
                "default_instructions": "You are a helpful AI mentor.",
                "roles": {"default": "You are a general mentor."},
                "tasks": {
                    "generate_intro_and_topics": "Return JSON with greeting, topics[], concluding_question, suggestions[].",
                    "chat": {
                        "system_prompt": "{context_summary}\n{role_instruction}\n{default_instruction}\n{json_output_instruction}",
                        "user_prompt_wrapper": "Summary: {summary}\nContinue the conversation based on recent messages.",
                    },
                    "summarize_conversation": "Summarize key points.",
                    "generate_topic_prompts": "Return a JSON array with 4 short questions for {topic}."
                },
                "shared_components": {
                    "json_output_format": "{\"response\":\"<markdown>\",\"suggestions\":[\"q1\",\"q2\",\"q3\"]}"
                },
            }

    @staticmethod
    def _sanitize_text(s: Optional[str]) -> str:
        return (s or "").strip()

    @staticmethod
    def _safe_json_parse(response_text: str) -> Dict[str, Any]:
        """Safely parse JSON response with multiple fallback attempts"""
        if not response_text.strip():
            return {}
        
        try:
            # First attempt: direct parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        try:
            # Second attempt: look for nested JSON in "response" field
            data = json.loads(response_text)
            if isinstance(data, dict) and "response" in data:
                try:
                    return json.loads(data["response"])
                except (json.JSONDecodeError, TypeError):
                    return data
            return data
        except json.JSONDecodeError:
            pass
        
        # Third attempt: extract JSON-like content using regex
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Final fallback
        log.warning("Could not parse JSON from response: %s", response_text[:200])
        return {"error": "Failed to parse response", "raw_response": response_text}

    async def _get_conversation_summary(self, chat_title: str, chat_history: List[Dict[str, Any]]) -> str:
        log.debug("Getting conversation summary for title=%s, messages=%d", chat_title, len(chat_history))
        if len(chat_history) < 10:
            return self.conversation_summaries.get(chat_title, "")
        try:
            payload = [{
                "role": "user",
                "content": self.prompts["tasks"]["summarize_conversation"] + "\n\n" + json.dumps(chat_history)
            }]
            resp = await self.conn.generate_chat_completion(messages=payload, temperature=0.3)
            summary = self._sanitize_text(resp)
            if summary:
                self.conversation_summaries[chat_title] = summary
            log.debug("Generated summary: %s", summary)
            return summary
        except Exception as e:
            log.warning("Failed to generate summary: %s", e)
            return self.conversation_summaries.get(chat_title, "")

    async def generate_intro_and_topics(
        self,
        *,
        context_description: str,
        extra_instructions: Optional[str] = None,
        role: Optional[str] = None
    ) -> Tuple[str, List[str], List[str]]:
        log.info("Generating intro and topics for role=%s", role or "default")
        role = role or "default"
        
        default_behavior = self.prompts.get("default_instructions", "")
        role_prompt = self.prompts["roles"].get(role, self.prompts["roles"]["default"])
        prompt_template = self.prompts["tasks"]["generate_intro_and_topics"]

        prompt_content = prompt_template.format(
            context_description=self._sanitize_text(context_description),
            role_prompt=self._sanitize_text(role_prompt),
            default_behavior=self._sanitize_text(default_behavior),
            extra_instructions=self._sanitize_text(extra_instructions) or "",
        )

        log.debug("Prompt content for intro/topics: %s", prompt_content)
        messages = [{"role": "user", "content": prompt_content}]
        
        llm_response = await self.conn.generate_chat_completion(
            messages=messages, 
            temperature=0.5, 
            json_mode=True,
            max_tokens=2048
        )
        log.debug("Raw LLM response: %s", llm_response)

        parsed = self._safe_json_parse(llm_response)
        
        if "error" in parsed:
            log.warning("JSON parsing failed, using fallback")
            return self._fallback_intro()

        greeting = self._sanitize_text(parsed.get("greeting", ""))
        topics = [self._sanitize_text(str(t)) for t in parsed.get("topics", []) if str(t).strip()]
        question = self._sanitize_text(parsed.get("concluding_question", ""))
        suggestions = [self._sanitize_text(str(s)) for s in parsed.get("suggestions", []) if str(s).strip()]

        # Ensure we have at least some content
        if not greeting:
            greeting = "Hello! I'm your AI mentor, ready to guide you through your learning journey."
        if not topics:
            topics = ["Introduction", "Core Concepts", "Practical Applications", "Advanced Topics"]
        if not question:
            question = "What would you like to explore first?"
        if not suggestions:
            suggestions = ["What should I focus on?", "Can you explain the basics?", "Show me examples", "How does this work?"]

        log.info("Intro generated: greeting=%s, topics=%d, suggestions=%d", greeting[:50], len(topics), len(suggestions))
        
        intro = (
            f"{greeting}\n\nHere are the topics we'll explore:\n- " + "\n- ".join(topics) + f"\n\n{question}"
            if topics else f"{greeting}\n\n{question}"
        )
        return intro, topics, suggestions

    def _fallback_intro(self) -> Tuple[str, List[str], List[str]]:
        log.debug("Using fallback intro")
        return (
            "Hello! I'm your AI mentor, ready to guide you through your learning journey.\n\nLet's start exploring together!",
            ["Introduction", "Core Concepts", "Practical Applications", "Advanced Topics"],
            ["What should I focus on first?", "Can you explain the basics?", "Show me an example", "How does this apply to real world?"],
        )

    async def chat(
        self,
        *,
        chat_history: List[Dict[str, Any]],
        user_id: str,
        chat_title: str,
        learning_goal: Optional[str],
        skills: List[str],
        difficulty: str,
        role: str,
        mentor_topics: Optional[List[str]] = None,
        current_topic: Optional[str] = None,
        completed_topics: Optional[List[str]] = None
    ) -> Tuple[str, List[str]]:
        log.info("Chat request: user=%s, title=%s, messages=%d", user_id, chat_title, len(chat_history))
        if not chat_history:
            log.warning("Empty chat history")
            return "Please start the conversation with a message.", []

        summary = await self._get_conversation_summary(chat_title, chat_history)

        context_lines = [f"Role: {role}"]
        if learning_goal:
            context_lines.append(f"Learning Goal: {learning_goal}")
        if skills:
            context_lines.append(f"Skills: {', '.join(skills)}")
        context_lines.append(f"Difficulty: {difficulty}")
        if mentor_topics:
            context_lines.append(f"Topics: {', '.join(mentor_topics)}")
        if current_topic:
            context_lines.append(f"Current Topic: {current_topic}")
        if completed_topics:
            context_lines.append(f"Completed Topics: {', '.join(completed_topics)}")

        role_instruction = self.prompts["roles"].get(role, self.prompts["roles"]["default"])
        default_instruction = self.prompts.get("default_instructions", "")
        json_output_instruction = self.prompts["shared_components"].get("json_output_format", "")
        
        system_prompt = self.prompts["tasks"]["chat"]["system_prompt"].format(
            context_summary="\n".join(context_lines),
            role_instruction=role_instruction,
            default_instruction=default_instruction,
            json_output_instruction=json_output_instruction,
        )
        
        user_prompt = self.prompts["tasks"]["chat"]["user_prompt_wrapper"].format(
            summary=summary or "(no prior summary)"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *chat_history[-6:],  # Only use last 6 messages to avoid token limits
            {"role": "user", "content": user_prompt}
        ]

        log.debug("System prompt: %s", system_prompt)
        llm_response = await self.conn.generate_chat_completion(
            messages=messages, 
            temperature=0.5, 
            json_mode=True,
            max_tokens=1500
        )
        log.debug("Raw LLM chat response: %s", llm_response)

        parsed = self._safe_json_parse(llm_response)
        
        # Extract response and suggestions with multiple fallback strategies
        reply = ""
        suggestions = []
        
        if "error" not in parsed:
            # Try different response field names
            reply = (
                self._sanitize_text(parsed.get("response", "")) or
                self._sanitize_text(parsed.get("reply", "")) or
                self._sanitize_text(parsed.get("content", ""))
            )
            
            # Try to get suggestions
            suggestions_raw = (
                parsed.get("suggestions", []) or
                parsed.get("follow_up", []) or
                parsed.get("prompts", [])
            )
            suggestions = [self._sanitize_text(str(s)) for s in suggestions_raw if str(s).strip()][:4]
        
        # Fallbacks if parsing failed
        if not reply:
            if "error" in parsed and "raw_response" in parsed:
                reply = parsed["raw_response"]
            else:
                reply = "I'm having trouble formatting my response. Could you please rephrase your question?"
        
        if not suggestions:
            suggestions = [
                "Can you explain more?",
                "What should I know next?",
                "Give me an example",
                "What's the next step?"
            ]

        log.info("Reply generated: %s (suggestions=%d)", reply[:80], len(suggestions))
        return reply, suggestions[:4]  # Ensure max 4 suggestions

    async def generate_topic_prompts(
        self,
        topic: str,
        *,
        context_description: str = "",
        role: Optional[str] = None
    ) -> List[str]:
        log.info("Generating topic prompts for: %s", topic)
        role = role or "default"
        role_prompt = self.prompts["roles"].get(role, self.prompts["roles"]["default"])
        tmpl = self.prompts["tasks"]["generate_topic_prompts"]
        
        prompt = tmpl.format(
            topic=topic, 
            role_prompt=role_prompt, 
            context_description=context_description
        )

        log.debug("Topic prompt content: %s", prompt)
        messages = [{"role": "user", "content": prompt}]
        
        resp = await self.conn.generate_chat_completion(
            messages=messages, 
            temperature=0.5, 
            json_mode=True,
            max_tokens=1024
        )
        log.debug("Raw topic prompts response: %s", resp)

        parsed = self._safe_json_parse(resp)
        
        prompts = []
        if "error" not in parsed:
            # Try different array field names
            prompts_raw = (
                parsed if isinstance(parsed, list) else
                parsed.get("prompts", []) or
                parsed.get("questions", []) or
                parsed.get("suggestions", [])
            )
            prompts = [self._sanitize_text(str(s)) for s in prompts_raw if str(s).strip()]

        # Fallback prompts if parsing failed
        if not prompts:
            prompts = [
                f"What are the basics of {topic}?",
                f"Give me an example of {topic}",
                f"How to apply {topic} in practice?",
                f"What are common mistakes with {topic}?",
            ]
            log.info("Using fallback topic prompts for %s", topic)
        
        log.info("Generated %d topic prompts", len(prompts))
        return prompts[:4]  # Ensure max 4 prompts
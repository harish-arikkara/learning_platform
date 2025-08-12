# mentor/core/engine/mentor_engine.py

import os
import sys
import json
import yaml
from typing import Optional, Tuple, List, Dict, Any

# Adjust system path to find the 'connection' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from connection import Connection

class MentorEngine:
    def __init__(self):
        """Initializes the MentorEngine."""
        self.conn = Connection()
        self.llm_client = self.conn.get_llm()
        self.llm_deployment_name = self.conn.get_llm_deployment_name()
        
        self.prompts = self._load_yaml("prompts.yaml")
        self.conversation_summaries = {}

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Loads a YAML file from the same directory."""
        path = os.path.join(os.path.dirname(__file__), filename)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _validate_and_sanitize_input(self, input_text: str) -> str:
        return input_text

    def _sanitize_output(self, output_text: str) -> str:
        # PII sanitization was removed. This is now a pass-through.
        return output_text

    def _clean_json_response(self, raw_response: str) -> str:
        """
        Cleans the raw LLM response to isolate the JSON object.
        Removes markdown fences (```json ... ```) and other surrounding text.
        """
        # Find the start of the JSON object
        start_brace = raw_response.find('{')
        # Find the end of the JSON object
        end_brace = raw_response.rfind('}')
        
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            # Extract the content between the first '{' and the last '}'
            return raw_response[start_brace:end_brace+1]
        
        # If no JSON object is found, return the original string to let the parser fail
        return raw_response

    async def _get_conversation_summary(self, chat_title: str, chat_history: List[Dict[str, Any]]) -> str:
        SUMMARY_THRESHOLD = 10
        if len(chat_history) < SUMMARY_THRESHOLD:
            return self.conversation_summaries.get(chat_title, "")

        messages_to_summarize = chat_history[:-5]
        summary_prompt = self.prompts["tasks"]["summarize_conversation"]
        
        summary_messages = [{"role": "system", "content": summary_prompt}]
        summary_messages.extend(messages_to_summarize)

        try:
            summary = await self.conn.generate_chat_completion(
                messages=summary_messages,
                temperature=0.3,
                max_tokens=250,
            )
            self.conversation_summaries[chat_title] = summary
            print(f"Generated new summary for chat '{chat_title}': {summary}")
            return summary
        except Exception as e:
            print(f"Error generating conversation summary: {e}")
            return self.conversation_summaries.get(chat_title, "")

    async def generate_intro_and_topics(
        self,
        context_description: str,
        extra_instructions: Optional[str] = None,
        role: Optional[str] = None
    ) -> Tuple[str, List[str], List[str]]:
        context_description = self._validate_and_sanitize_input(context_description)
        extra_instructions = self._validate_and_sanitize_input(extra_instructions) if extra_instructions else ""
        
        default_behavior = self.prompts["default_instructions"]
        role_prompt = self.prompts["roles"].get(role, self.prompts["roles"]["default"])
        prompt_template = self.prompts["tasks"]["generate_intro_and_topics"]

        # Fix: Use the correct placeholder names that match the template
        prompt_content = prompt_template.format(
            context_description=context_description,
            role_prompt=role_prompt,
            default_behavior=default_behavior,
            extra_instructions=extra_instructions
        )
        messages = [{"role": "user", "content": prompt_content}]
        
        try:
            llm_raw_response = await self.conn.generate_chat_completion(
                messages=messages,
                temperature=0.5,
                max_tokens=800,
                json_mode=True
            )
            # ** NEW: Clean the response before parsing **
            cleaned_response = self._clean_json_response(llm_raw_response)
            parsed = json.loads(cleaned_response)
            
            greeting = self._sanitize_output(parsed.get("greeting", "Hello!"))
            topics = [self._sanitize_output(t) for t in parsed.get("topics", [])]
            question = self._sanitize_output(parsed.get("concluding_question", "Shall we start?"))
            suggestions = [self._sanitize_output(s) for s in parsed.get("suggestions", [])]
            
            intro_message = f"{greeting}\n\nHere are the topics we'll explore:\n- " + "\n- ".join(topics) + f"\n\n{question}"
            return (intro_message, topics, suggestions)
        except Exception as e:
            print(f"Error in generate_intro_and_topics: {e}")
            fallback_intro = "Hello! I'm your mentor, ready to guide you.\n\nHere are some topics:\n- Introduction\n- Core Concepts\n- Advanced Topics\n\nShall we start?"
            return fallback_intro, ["Introduction", "Core Concepts", "Advanced Topics"], ["What should I focus on first?", "Can you explain the first topic?", "How does this relate to my goal?", "Can you quiz me on a topic?"]

    async def chat(
        self,
        chat_history: List[Dict[str, Any]],
        user_id: str,
        chat_title: str,
        learning_goal: Optional[str],
        skills: List[str],
        difficulty: str,
        role: str,
        mentor_topics: Optional[List[str]] = None,
        current_topic: Optional[str] = None,
        completed_topics: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        if not chat_history:
            return "Please start the conversation with a message.", []

        summary = await self._get_conversation_summary(chat_title, chat_history)
        recent_history = chat_history[-6:]

        system_prompt = self._build_system_context(learning_goal, skills, difficulty, role, mentor_topics, current_topic, completed_topics)
        
        if summary:
            user_prompt_wrapper = self.prompts["tasks"]["chat"]["user_prompt_wrapper"]
            system_prompt += "\n\n" + user_prompt_wrapper.format(summary=summary)

        try:
            llm_raw_response = await self.conn.generate_chat_completion(
                messages=recent_history,
                system_instruction=system_prompt,
                temperature=0.7,
                max_tokens=1500,
                json_mode=True
            )
            # ** NEW: Clean the response before parsing **
            cleaned_response = self._clean_json_response(llm_raw_response)
            parsed = json.loads(cleaned_response)

            reply = self._sanitize_output(parsed.get("reply", "I'm sorry, I couldn't form a proper reply."))
            suggestions = [self._sanitize_output(s) for s in parsed.get("suggestions", [])]
            return reply, suggestions
        except json.JSONDecodeError as e:
            print(f"CRITICAL: LLM failed to produce valid JSON. Error: {e}. Raw Response: {llm_raw_response}")
            return "I seem to be having trouble formatting my thoughts. Please try rephrasing your question.", []
        except Exception as e:
            print(f"Error in chat: {e}")
            return "I'm sorry, I couldn't understand your question. Could you please rephrase it?", []

    def _build_system_context(
        self, learning_goal, skills, difficulty, role, mentor_topics, current_topic, completed_topics
    ) -> str:
        context_lines = [f"Role: {role}"]
        if learning_goal: context_lines.append(f"Learning Goal: {learning_goal}")
        if skills: context_lines.append(f"Skills: {', '.join(skills)}")
        context_lines.append(f"Difficulty: {difficulty}")
        if mentor_topics: context_lines.append(f"Topics: {', '.join(mentor_topics)}")
        if current_topic: context_lines.append(f"Current Topic: {current_topic}")
        if completed_topics: context_lines.append(f"Completed Topics: {', '.join(completed_topics)}")
        
        role_instruction = self.prompts["roles"].get(role, self.prompts["roles"]["default"])
        default_instruction = self.prompts["default_instructions"]
        json_output_instruction = self.prompts["shared_components"]["json_output_format"]
        system_prompt_template = self.prompts["tasks"]["chat"]["system_prompt"]

        return system_prompt_template.format(
            context_summary="\n".join(context_lines),
            role_instruction=role_instruction,
            default_instruction=default_instruction,
            json_output_instruction=json_output_instruction
        )
    
    async def generate_topic_prompts(
        self,
        topic: str,
        context_description: str = "",
        role: Optional[str] = None
    ) -> list:
        topic = self._validate_and_sanitize_input(topic)
        context_description = self._validate_and_sanitize_input(context_description)
        
        role_prompt = self.prompts["roles"].get(role, self.prompts["roles"]["default"])
        prompt_template = self.prompts["tasks"]["generate_topic_prompts"]
        
        prompt_content = prompt_template.format(
            topic=topic,
            role_prompt=role_prompt,
            context_description=context_description
        )
        messages = [{"role": "user", "content": prompt_content}]
        
        try:
            llm_response = await self.conn.generate_chat_completion(
                messages=messages,
                temperature=0.5,
                max_tokens=500,
                json_mode=True
            )
            # ** NEW: Clean the response before parsing **
            cleaned_response = self._clean_json_response(llm_response)
            prompts = json.loads(cleaned_response)

            return [self._sanitize_output(p) for p in prompts]
        except Exception as e:
            print(f"Error in generate_topic_prompts: {e}")
            return [f"What are the basics of {topic}?", f"Give me an example of {topic}", f"How to apply {topic}?", f"Common mistakes in {topic}?"]
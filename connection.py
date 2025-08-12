# mentor/core/engine/Connection.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

class Connection:
    def __init__(self):
        """Initializes the connection to the Google Gemini API."""
        load_dotenv() # Load environment variables from .env file

        self.api_key = os.getenv("GOOGLE_API_KEY")
        # Use Gemini 1.5 Flash as requested, allow override via .env
        self.model_name = os.getenv("GEMINI_MODEL_NAME")

        if not self.api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment variables. Please check your .env file.")

        # Configure the Gemini client
        genai.configure(api_key=self.api_key)

        # This client object represents your "LLM" connection.
        self.client = genai.GenerativeModel(self.model_name)

    def get_llm(self) -> genai.GenerativeModel:
        """
        Returns the initialized GenerativeModel client instance.
        """
        return self.client

    def get_llm_deployment_name(self) -> str:
        """
        Returns the name of the Gemini model (e.g., "gemini-2.5-flash").
        """
        return self.model_name

    async def generate_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 2024,
        json_mode: bool = False,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Makes a chat completion call using the configured Gemini client.
        This method encapsulates the actual API interaction.
        """
        # Map OpenAI-style message roles to Gemini's format.
        chat_history = []
        effective_system_instruction = system_instruction

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "assistant":
                role = "model"
            
            if role in ["user", "model"]:
                chat_history.append({"role": role, "parts": [content]})
            elif role == "system" and effective_system_instruction is None:
                # If a system message is in the history and not passed separately,
                # use its content as the system instruction.
                effective_system_instruction = content

        # Configure the generation parameters for the API call
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if json_mode:
            # Instruct Gemini to output a JSON object
            generation_config.response_mime_type = "application/json"

        # Configure safety settings to be less restrictive
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]

        # Use a model instance with the system instruction, if provided
        model_to_use = self.client
        if effective_system_instruction:
            model_to_use = genai.GenerativeModel(
                self.model_name,
                system_instruction=effective_system_instruction,
                safety_settings=safety_settings
            )
        else:
            model_to_use = genai.GenerativeModel(
                self.model_name,
                safety_settings=safety_settings
            )

        try:
            response = await model_to_use.generate_content_async(
                contents=chat_history,
                generation_config=generation_config,
            )
            
            # Enhanced error handling for blocked responses
            if not response.candidates:
                feedback = response.prompt_feedback
                print(f"❌ Gemini API: Response was blocked. Reason: {feedback.block_reason}")
                if feedback.safety_ratings:
                    for rating in feedback.safety_ratings:
                        print(f"  - {rating.category}: {rating.probability}")
                return "I apologize, but I need to be careful with my response. Could you please rephrase your request or try asking about the topic in a different way?"
            
            # Check if the candidate has content
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                # Handle safety-filtered responses
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    if finish_reason == 2:  # SAFETY
                        print(f"❌ Gemini API: Response blocked due to safety filters")
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            for rating in candidate.safety_ratings:
                                print(f"  - {rating.category}: {rating.probability}")
                        return "I need to be careful with my response. Let me try to help you in a different way. Could you rephrase your question?"
                    elif finish_reason == 3:  # MAX_TOKENS
                        print(f"❌ Gemini API: Response truncated due to max tokens limit")
                        return "My response was too long. Could you ask for a shorter explanation or break your question into smaller parts?"
                    else:
                        print(f"❌ Gemini API: Response blocked. Finish reason: {finish_reason}")
                        return "I encountered an issue generating a response. Please try rephrasing your question."
                
                return "I'm having trouble generating a response right now. Please try again with a different question."

            return response.text.strip()
            
        except Exception as e:
            print(f"❌ Gemini API error in generate_chat_completion: {e}")
            # Return a user-friendly fallback instead of raising
            return "I'm experiencing some technical difficulties. Please try again or rephrase your question."


# Example usage (updated for Gemini)
if __name__ == "__main__":
    import asyncio

    async def test():
        try:
            conn = Connection()
            # Get the LLM client and model name
            llm_model = conn.get_llm()
            llm_model_name = conn.get_llm_deployment_name()

            print(f"Obtained LLM Client: {type(llm_model)}")
            print(f"Using Model Name: {llm_model_name}")

            # Prepare messages in the standard format
            messages_to_send: List[Dict[str, Any]] = [{"role": "user", "content": "Tell me a short, funny story about a talking parrot."}]

            print(f"\nAttempting to generate chat completion...")
            # Use the helper method to get the completion
            reply = await conn.generate_chat_completion(messages=messages_to_send,json_mode=True)
            print("✅ LLM Response:\n", reply)

        except ValueError as ve:
            print(f"Configuration Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during test: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test())
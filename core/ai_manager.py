import openai
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import re
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

class AIManager:
    def __init__(self):
        self.use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
        if self.use_azure:
            self.api_key, self.azure_endpoint, self.deployment_name = (os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
            if not all([self.api_key, self.azure_endpoint, self.deployment_name]): raise ValueError("For Azure, please set all required environment variables.")
            self.client = openai.AzureOpenAI(api_key=self.api_key, api_version="2024-02-01", azure_endpoint=self.azure_endpoint)
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.model = "gpt-4o"
            if not self.api_key: raise ValueError("For standard OpenAI, please set OPENAI_API_KEY.")
            self.client = openai.OpenAI(api_key=self.api_key)

    def _call_llm(self, messages: List[Dict], temperature: float = 0.5, json_mode: bool = False) -> str:
        try:
            model_to_use = self.deployment_name if self.use_azure else self.model
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            response = self.client.chat.completions.create(model=model_to_use, messages=messages, temperature=temperature, response_format=response_format, max_tokens=4000)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error calling OpenAI API: {e}")
            return ""

    def _robust_json_parser(self, json_string: str) -> Optional[Dict]:
        try:
            cleaned_string = re.sub(r'```json\s*|\s*```', '', json_string, flags=re.DOTALL)
            return json.loads(cleaned_string)
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON. Error: {e}\nRaw response:\n{json_string}")
            return None

    # In ai_manager.py, REPLACE the summarize_actions method with this:

    def summarize_actions(self, steps_with_data: List[Dict]) -> List[Dict]:
        """
        UPGRADED: This method now instructs the AI to group short, related events
        into longer, more meaningful user actions (approx. 8-15 seconds).
        """
        print("🤖 AI Pass 1: Grouping events into meaningful actions...")
        
        system_prompt = (
            "You are an expert video editor and user interface analyst. Your task is to analyze a log of short, sequential user events "
            "and **group them into larger, meaningful user actions.** An 'action' should represent a complete, logical task, "
            "like 'Creating a new page' or 'Exploring the basics menu'."
        )

        all_grouped_steps = []
        chunk_size = 20 # We can use a slightly larger chunk size for this task
        
        for i in range(0, len(steps_with_data), chunk_size):
            chunk = steps_with_data[i:i + chunk_size]
            print(f"  -> Analyzing chunk {i//chunk_size + 1} for actions...")

            user_prompt = (
                "Based on the following log of short events, group consecutive related events into logical user actions. "
                "Each action you create should have a duration of roughly 8-15 seconds. Merge multiple short events if they belong to the same logical action. "
                "For each new, longer action you identify, provide a concise 'description', a 'start_time' from the *first* event in the group, and an 'end_time' from the *last* event in the group. "
                "Also, make sure to correct any obvious OCR errors (e.g., 'Rasics' should be 'Basics').\n\n"
                "Your output MUST be a JSON object with a 'steps' key, containing a list of these new, longer actions.\n\n"
                f"EVENT LOG CHUNK:\n{json.dumps(chunk, indent=2)}"
            )
            
            response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.5, json_mode=True)
            structured_summary = self._robust_json_parser(response_str)

            if structured_summary and "steps" in structured_summary:
                all_grouped_steps.extend(structured_summary["steps"])
            else:
                print(f"⚠️ Failed to process chunk {i//chunk_size + 1}. Skipping.")
        
        if all_grouped_steps:
            # Re-number the steps sequentially for consistency in subsequent stages
            for idx, step in enumerate(all_grouped_steps):
                step['step_number'] = idx + 1
            print("✅ Events successfully grouped into longer actions.")
            return all_grouped_steps
        else:
            print("⚠️ Failed to group any events into actions.")
            return []

    def select_relevant_segments(self, structured_summary: List[Dict], user_prompt: str) -> List[Dict]:
        print("🤖 AI Pass 2: Selecting relevant segments...")
        system_prompt = "You are an intelligent video segment selector. Respond ONLY with a JSON object."
        user_prompt_for_selection = (f"Here is a summary of actions:\n\n{json.dumps(structured_summary, indent=2)}\n\nThe user wants a video guide about: '{user_prompt}'\n\nSelect all step numbers relevant to the user's request. Return a JSON object with a single key 'relevant_steps' containing a list of integers (the step numbers).")
        response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_selection}], temperature=0.2, json_mode=True)
        selection_data = self._robust_json_parser(response_str)
        if selection_data and "relevant_steps" in selection_data:
            selected_indices = selection_data.get("relevant_steps", [])
            selected_steps = [step for step in structured_summary if step.get('step_number') in selected_indices]
            print(f"✅ Selected {len(selected_steps)} relevant steps.")
            return selected_steps
        return []

    # def generate_narrations(self, production_plan: List[Dict]) -> Dict[str, str]:
    #     print("🤖 AI Pass 3: Generating narrations for each shot...")
    #     system_prompt = ("You are a professional scriptwriter for short, clear tutorial videos. For each shot description, write a single, concise narration sentence (max 15 words). Respond ONLY with a valid JSON object.")
    #     shots_info = [f"Shot {shot['shot_number']} ({shot['output_filename']}): Prompt is '{shot['prompt']}'" for shot in production_plan]
    #     user_prompt_for_script = ("Based on the shot list, generate a single sentence of voiceover narration for each one. The narration should be in the second person ('you'). Your response must be a single JSON object where keys are the 'output_filename' and values are the narration strings.\n\n" f"SHOT LIST:\n{json.dumps(shots_info, indent=2)}")
    #     response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_script}], temperature=0.7, json_mode=True)
    #     narrations = self._robust_json_parser(response_str)
    #     if narrations and isinstance(narrations, dict):
    #         print("✅ Narrations generated for all shots.")
    #         return narrations
    #     print("⚠️ Failed to generate narration dictionary.")
    #     return {}

    # In ai_manager.py, REPLACE the generate_narrations method

    # def generate_narrations(self, production_plan: List[Dict]) -> Dict[str, str]:
    #     """
    #     MODIFIED: Generates extremely concise narrations to fit within short video clips.
    #     """
    #     print("🤖 AI Pass 3: Generating concise narrations for each shot...")
    #     system_prompt = (
    #         "You are a professional scriptwriter for fast-paced, cinematic tech tutorials. "
    #         "Your job is to write one, extremely concise sentence of narration for each shot. "
    #         "CRITICAL: Each sentence MUST be easily spoken in **under 4 seconds** (ideally 8-10 words). "
    #         "Respond ONLY with a valid JSON object."
    #     )

    #     shots_info = [f"Shot {shot['shot_number']}: The user {shot['prompt'].lower()}" for shot in production_plan]
        
    #     user_prompt_for_script = (
    #         "Based on the following shot list, generate an ultra-concise voiceover narration for each one. "
    #         "Each narration must be speakable in under 4 seconds. Use the second person ('you'). "
    #         "Your response must be a single JSON object where keys are the 'output_filename' for each shot, "
    #         "and values are the short narration strings.\n\n"
    #         f"SHOT LIST:\n{json.dumps(shots_info, indent=2)}"
    #     )
        
    #     response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_script}], temperature=0.7, json_mode=True)
    #     narrations = self._robust_json_parser(response_str)
        
    #     if narrations and isinstance(narrations, dict):
    #         print("✅ Concise narrations generated for all shots.")
    #         return narrations
        
    #     print("⚠️ Failed to generate narration dictionary.")
    #     return {}
        
    # In ai_manager.py, REPLACE the generate_narrations method with this:

    def generate_narrations(self, production_plan: List[Dict]) -> List[Dict]:
        """
        MODIFIED: Generates narrations as a list of objects for better reliability.
        """
        print("🤖 AI Pass 3: Generating narrations for each shot...")
        system_prompt = (
            "You are a professional scriptwriter for short, clear tutorial videos. For each shot description I provide, write a single, "
            "concise narration sentence (max 15 words). Your response MUST be a valid JSON object containing a 'narrations' key, "
            "which holds a list of objects."
        )

        shots_info = [{"shot_number": shot['shot_number'], "output_filename": shot['output_filename'], "prompt": shot['prompt']} for shot in production_plan]
        
        user_prompt_for_script = (
            "Based on the following list of shots, generate a single, concise voiceover sentence for each one. The narration should be in the second person ('you'). "
            "Avoid mentioning vague things mentioned in the script like description of animations or something vague that shouldn't be in the guide video"
            "Your response must be a single JSON object with a 'narrations' key. This key should contain a list of objects, "
            "where each object has two keys: 'output_filename' and 'narration'.\n\n"
            f"SHOT LIST:\n{json.dumps(shots_info, indent=2)}"
        )
        
        response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_script}], temperature=0.7, json_mode=True)
        narrations_data = self._robust_json_parser(response_str)
        
        if narrations_data and "narrations" in narrations_data:
            print("✅ Narrations generated for all shots.")
            return narrations_data["narrations"]
        
        print("⚠️ Failed to generate narration list.")
        return []
        
    def create_voiceover_for_shot(self, text: str, output_path: str):
        if not text: return
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(output_path)
        except Exception as e:
            print(f"⚠️ Could not create voiceover for shot: {e}")
            
    # In ai_manager.py, REPLACE the create_production_plan method with this:

    def create_production_plan(self, selected_steps: List[Dict], user_prompt: str) -> List[Dict]:
        """
        MODIFIED: The AI Director prompt now includes a specific example of the
        desired JSON output to ensure the structure is always correct.
        """
        print("🤖 AI Director: Creating a Video-to-Video production plan...")
        system_prompt = ("You are an expert AI Video Director. Your job is to create a 'shot list' for a human to follow using RunwayML. Your output MUST be a valid JSON object with the exact structure requested.")
        available_tools = """
        - tool: 'veo3' (Input: Text, Output: Video) - Use for generating new clips from scratch.
        - tool: 'gen4_aleph' (Input: Video + Text, Output: Video) - Use for transforming existing video clips.
        """
        
        # --- NEW: Added a clear example to the prompt ---
        example_output_format = """
        "production_plan": [
            {
                "shot_number": 1,
                "model_to_use": "veo3",
                "prompt": "A cinematic, abstract animation of connected ideas and nodes forming a beautiful collaborative workspace, clean and modern aesthetic, purple and blue tones.",
                "original_step_number": null,
                "output_filename": "shot_01_intro.mp4"
            },
            {
                "shot_number": 2,
                "model_to_use": "gen4_aleph",
                "prompt": "Subtly highlight the 'Create new page' button with a soft glow as the mouse clicks it. Keep the rest of the interface clean.",
                "original_step_number": 12,
                "output_filename": "shot_02_create_page.mp4"
            }
        ]
        """

        user_prompt_for_plan = (
            f"## GOAL\nCreate a production plan for a tutorial about: '{user_prompt}'.\n\n## AVAILABLE TOOLS\n{available_tools}\n\n## KEY SCENES\n{json.dumps(selected_steps, indent=2)}\n\n"
            "## YOUR TASK\nCreate a JSON shot list called 'production_plan'. Create an optional intro shot with 'veo3' and a transformation shot with 'gen4_aleph' for EACH key scene. Follow the JSON output format exactly.\n\n"
            f"## EXAMPLE OUTPUT FORMAT\nYour final JSON object must follow this structure:\n```json\n{example_output_format}\n```\n\n"
            "## JSON OUTPUT KEYS\nFor each shot, you must provide these exact keys: 'shot_number', 'model_to_use', 'prompt', 'original_step_number', 'output_filename'."
        )
        response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_plan}], temperature=0.8, json_mode=True)
        plan_data = self._robust_json_parser(response_str)
        if plan_data and "production_plan" in plan_data:
            # Additional check to ensure the list contains dictionaries
            if isinstance(plan_data["production_plan"], list) and all(isinstance(item, dict) for item in plan_data["production_plan"]):
                print("✅ Production plan generated successfully.")
                return plan_data["production_plan"]
        
        print("⚠️ AI Director failed to generate a valid, structured production plan.")
        return []

        def create_intro_plan(self, user_prompt: str) -> Dict:
        """Generates a plan for a single, creative intro video."""
        print("🤖 AI Director: Creating intro plan...")
        system_prompt = "You are an AI Video Director. Create a single, creative prompt for the 'veo3' model to generate a cinematic intro video. Respond ONLY with a valid JSON object."
        user_prompt_for_plan = (
            f"The final video is a tutorial about: '{user_prompt}'. "
            "Write one creative, exciting prompt for a 4-second intro video. "
            "The prompt should be visually descriptive and related to the topic. "
            "Return a JSON object with a single key 'intro_prompt' containing the prompt string."
        )
        response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_plan}], temperature=0.8, json_mode=True)
        plan_data = self._robust_json_parser(response_str)
        if plan_data and "intro_prompt" in plan_data:
            return {"prompt": plan_data["intro_prompt"]}
        return {"prompt": "A cinematic intro about the video topic."} # Fallback
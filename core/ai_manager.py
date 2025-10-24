# import openai
# import json
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# import os
# import re
# from gtts import gTTS
# from dotenv import load_dotenv

# load_dotenv()

# class AIManager:
#     def __init__(self):
#         self.use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
#         if self.use_azure:
#             print("🔧 Configuring for Azure OpenAI...")
#             self.api_key, self.azure_endpoint, self.deployment_name = (
#                 os.getenv("AZURE_OPENAI_API_KEY"),
#                 os.getenv("AZURE_OPENAI_ENDPOINT"),
#                 os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
#             )
#             if not all([self.api_key, self.azure_endpoint, self.deployment_name]):
#                 raise ValueError("For Azure, please set all required environment variables.")
#             self.client = openai.AzureOpenAI(
#                 api_key=self.api_key, api_version="2024-02-01", azure_endpoint=self.azure_endpoint
#             )
#             print("✅ Azure OpenAI Client configured.")
#         else:
#             print("🔧 Configuring for standard OpenAI...")
#             self.api_key = os.getenv("OPENAI_API_KEY")
#             self.model = "gpt-4o"
#             if not self.api_key:
#                 raise ValueError("For standard OpenAI, please set OPENAI_API_KEY.")
#             self.client = openai.OpenAI(api_key=self.api_key)
#             print("✅ Standard OpenAI Client configured.")

#     def _call_llm(self, messages: List[Dict], temperature: float = 0.5, json_mode: bool = False) -> str:
#         try:
#             model_to_use = self.deployment_name if self.use_azure else self.model
#             response_format = {"type": "json_object"} if json_mode else {"type": "text"}
#             response = self.client.chat.completions.create(
#                 model=model_to_use,
#                 messages=messages,
#                 temperature=temperature,
#                 response_format=response_format,
#                 max_tokens=4000
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"❌ Error calling OpenAI API: {e}")
#             return ""

#     def _robust_json_parser(self, json_string: str) -> Optional[Dict]:
#         try:
#             cleaned_string = re.sub(r'```json\s*|\s*```', '', json_string, flags=re.DOTALL)
#             return json.loads(cleaned_string)
#         except json.JSONDecodeError as e:
#             print(f"⚠️ Failed to parse JSON. Error: {e}\nRaw response:\n{json_string}")
#             return None
#     '''
#     def summarize_actions(self, steps_with_data: List[Dict]) -> List[Dict]:
#         """
#         CORRECTED: Analyzes pre-defined steps (with start/end times and text changes)
#         and generates a high-level summary description for each one.
#         """
#         print("🤖 AI Pass 1: Summarizing action steps...")
#         system_prompt = (
#             "You are an expert user interface analyst. Your task is to analyze a list of user interaction steps, "
#             "each defined by a start time, end time, and the text that appeared or disappeared on screen. "
#             "For each step, you will write a concise, high-level description of the user's action. "
#             "Your output MUST be a JSON object containing a 'steps' key, which is a list of these analyzed steps. "
#             "Crucially, you MUST preserve the original 'step_number', 'start_time', and 'end_time' for each step."
#         )
#         user_prompt = (
#             "Based on the following log of steps, generate a concise 'description' for each one. "
#             "Focus on the primary user action (e.g., 'User navigates to the dashboard', 'User searches for a template'). "
#             "Do not list every single text change in your description. "
#             "Return the complete list of steps, including your new description and the original timing data.\n\n"
#             f"STEPS LOG:\n{json.dumps(steps_with_data, indent=2)}"
#         )
        
#         response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.2, json_mode=True)
        
#         structured_summary = self._robust_json_parser(response_str)
#         if structured_summary and "steps" in structured_summary:
#             print("✅ High-level summary generated for all steps.")
#             return structured_summary.get("steps", [])
#         print("⚠️ Failed to generate structured summary or 'steps' key was missing.")
#         return []

#     def select_relevant_segments(self, structured_summary: List[Dict], user_prompt: str) -> List[Dict]:
#         """
#         Pass 2: Selects segments relevant to the user's prompt.
#         """
#         print("🤖 AI Pass 2: Selecting relevant segments...")
#         system_prompt = "You are an intelligent video segment selector. Your goal is to choose enough relevant steps to create a coherent guide video. Respond ONLY with a JSON object."
        
#         user_prompt_for_selection = (
#             f"Here is a summary of actions:\n\n{json.dumps(structured_summary, indent=2)}\n\nThe user wants a video guide about: '{user_prompt}'\n\n"
#             "Analyze the summary and select all step numbers that are thematically or directly relevant to the user's request. "
#             "It is better to include a partially relevant step than to miss a potentially important one. "
#             "Return your answer as a JSON object with a single key 'relevant_steps' containing a list of integers (the step numbers)."
#         )
        
#         response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_selection}], temperature=0.2, json_mode=True)
        
#         selection_data = self._robust_json_parser(response_str)
#         if selection_data and "relevant_steps" in selection_data:
#             selected_indices = selection_data.get("relevant_steps", [])
#             selected_steps = [step for step in structured_summary if step.get('step_number') in selected_indices]
#             total_duration = sum(step.get('end_time', 0) - step.get('start_time', 0) for step in selected_steps)
#             print(f"✅ Selected {len(selected_steps)} steps with a total duration of {total_duration:.1f} seconds.")
#             return selected_steps
#         return []

#     def generate_script(self, selected_steps: List[Dict], user_prompt: str) -> str:
#         """
#         Pass 3: Generates the final voiceover script from the selected steps.
#         """
#         print("🤖 AI Pass 3: Generating final voiceover script...")
#         system_prompt = (
#             "You are a professional scriptwriter for tutorial videos. Your tone is clear, friendly, and professional. "
#             "Your output must be ONLY the voiceover script itself, perfectly formatted for text-to-speech."
#         )
        
#         formatted_actions = "".join([f"- Step {step.get('step_number')}: {step['description']}\n" for step in selected_steps])

#         user_prompt_for_script = (
#             "Write a professional voiceover script for a video tutorial about: "
#             f"'{user_prompt}'.\n\n"
#             "The script must be based on these key steps:\n"
#             f"{formatted_actions}\n\n"
#             "The script MUST have three parts:\n"
#             "1. Intro: A brief, welcoming sentence (10-15 words) that introduces the topic.\n"
#             "2. Body: A step-by-step narration of the key actions provided above. Refer to the actions smoothly.\n"
#             "3. Outro: A short concluding sentence (10-15 words) that wraps up the tutorial.\n\n"
#             "Your response must contain ONLY the complete, spoken script. Do not include any extra text, titles, or labels like 'Intro:' or 'Body:'."
#         )
        
#         script = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_script}], temperature=0.7)
#         print("✅ Script generated.")
#         return script
#     '''
#     def create_voiceover(self, script: str, output_path: str):
#         """Generates an MP3 voiceover file from the script text."""
#         if not script:
#             print("⚠️ Script is empty. Skipping voiceover generation.")
#             return
#         try:
#             tts = gTTS(text=script, lang='en')
#             tts.save(output_path)
#             print(f"🎤 Voiceover saved successfully to {output_path}")
#         except Exception as e:
#             print(f"⚠️ Could not create voiceover: {e}")


#     def summarize_actions(self, action_log: List[Dict]) -> List[Dict]:
#         """
#         UPDATED: Pass 1 with a prompt that demands more detail in the summary.
#         """
#         print("🤖 AI Pass 1: Creating detailed structured summary...")
#         system_prompt = (
#             "You are a meticulous assistant that analyzes a user action log from a screen recording. "
#             "Your task is to convert this raw log into a structured JSON object of chronological steps. "
#             "The description for each step MUST be detailed and include the key UI elements or specific text "
#             "that appeared or disappeared, such as button names, menu items, or page titles."
#         )
#         user_prompt = f"Based on the following JSON action log, create a detailed, structured summary. The output MUST be a JSON object with a 'steps' key, containing a list of objects, each with 'step_number', 'start_time', 'end_time', and a detailed 'description'.\n\nACTION LOG:\n{json.dumps(action_log, indent=2)}"
        
#         response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.1, json_mode=True)
        
#         structured_summary = self._robust_json_parser(response_str)
#         if structured_summary:
#             print("✅ Detailed summary generated.")
#             return structured_summary.get("steps", [])
#         return []

#     def select_relevant_segments(self, structured_summary: List[Dict], user_prompt: str) -> List[Dict]:
#         """
#         UPDATED: Pass 2 with a prompt that encourages selecting more content.
#         """
#         print("🤖 AI Pass 2: Selecting relevant segments...")
#         system_prompt = "You are an intelligent video segment selector. Your goal is to choose enough relevant steps to create a guide video that is approximately 60 seconds long. Respond ONLY with a JSON object containing a list of the relevant step numbers."
        
#         user_prompt_for_selection = (
#             f"Here is a structured summary of actions:\n\n{json.dumps(structured_summary, indent=2)}\n\nThe user wants a video guide about: '{user_prompt}'\n\n"
#             "Analyze the summary and select all step numbers that are thematically or directly relevant to the user's request. "
#             "It is better to include a partially relevant step than to miss a potentially important one. "
#             "Your selection should result in a total video duration of around 60 seconds. "
#             "Return your answer as a JSON object with a single key 'relevant_steps' containing a list of integers."
#         )
        
#         response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_selection}], temperature=0.2, json_mode=True)
        
#         selection_data = self._robust_json_parser(response_str)
#         if selection_data:
#             selected_indices = selection_data.get("relevant_steps", [])
#             selected_steps = [step for step in structured_summary if step['step_number'] in selected_indices]
#             total_duration = sum(step['end_time'] - step['start_time'] for step in selected_steps)
#             print(f"✅ Selected {len(selected_steps)} actions with a total duration of {total_duration:.1f} seconds.")
#             return selected_steps
#         return []

#     def generate_script(self, selected_steps: List[Dict], user_prompt: str) -> str:
#         """
#         UPDATED: Pass 3 with a prompt that enforces a proper script structure.
#         """
#         print("🤖 AI Pass 3: Generating final voiceover script...")
#         system_prompt = (
#             "You are a professional scriptwriter for tutorial videos. Your tone is clear, friendly, and professional. "
#             "Your output must be ONLY the voiceover script itself, perfectly formatted for text-to-speech."
#         )
        
#         formatted_actions = "".join([f"- {step['description']}\n" for step in selected_steps])

#         user_prompt_for_script = (
#         "Write a clear, natural-sounding professional voiceover script for a tutorial video about: "
#         f"'{user_prompt}'.\n\n"
#         "The script should directly guide viewers through the process using an engaging and confident tone — "
#         "as if a skilled instructor is explaining each step.\n\n"
#         "Base the narration strictly on the following key steps:\n"
#         f"{formatted_actions}\n\n"
#         "The script MUST include three parts:\n"
#         "1. Intro – A concise (10–15 words) welcome that clearly introduces the topic and goal of the tutorial.\n"
#         "2. Body – A detailed, step-by-step explanation that matches the provided actions. "
#         "Each step should be described in practical, instructional language that clearly tells the viewer what to do and why.\n"
#         "3. Outro – A short (10–15 words) closing line that reinforces what was learned and ends confidently.\n\n"
#         "Guidelines:\n"
#         "- Avoid vague or filler phrases like 'Next, we do something' or 'You can now proceed'.\n"
#         "- Use active, instructive language (e.g., 'Click', 'Select', 'Open', 'Enter', 'Save').\n"
#         "- Keep the flow conversational and natural for voiceover delivery.\n"
#         "- Do NOT include any titles, stage directions, labels (like 'Intro:'), or non-spoken text.\n\n"
#         "Return ONLY the complete, ready-to-record spoken script."
#         )

        
#         script = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_script}], temperature=0.7)
#         print("✅ Script generated.")
#         return script

    
'''
File: run_pipeline_direct.py
MODIFIED:
- Added Step 3.5 to generate an intro plan and provide manual instructions.
- Updated the call to editor.create_guide_video to pass the intro_clip_path.
'''
'''
File: core/ai_manager2.py
DESCRIPTION:
- Manages all 3 AI passes: Summarize, Select, Script.
- Generates a plan for a manual veo3 intro clip.
- Uses Azure TTS for high-quality voiceover generation.
'''

import openai
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import re
import requests  # Required for Azure TTS
from dotenv import load_dotenv

load_dotenv()

class AIManager:
    def __init__(self):
        # --- OpenAI LLM Config ---
        self.use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
        if self.use_azure:
            print("🔧 Configuring for Azure OpenAI (LLM)...")
            self.api_key, self.azure_endpoint, self.deployment_name = (
                os.getenv("AZURE_OPENAI_API_KEY"),
                os.getenv("AZURE_OPENAI_ENDPOINT"),
                os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            )
            if not all([self.api_key, self.azure_endpoint, self.deployment_name]):
                raise ValueError("For Azure LLM, please set all required environment variables.")
            self.client = openai.AzureOpenAI(
                api_key=self.api_key, api_version="2024-02-01", azure_endpoint=self.azure_endpoint
            )
            print("✅ Azure OpenAI (LLM) Client configured.")
        else:
            print("🔧 Configuring for standard OpenAI (LLM)...")
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.model = "gpt-4o"
            if not self.api_key:
                raise ValueError("For standard OpenAI (LLM), please set OPENAI_API_KEY.")
            self.client = openai.OpenAI(api_key=self.api_key)
            print("✅ Standard OpenAI (LLM) Client configured.")

        # --- Azure TTS Config ---
        print("🔧 Configuring for Azure TTS...")
        self.azure_tts_endpoint = os.getenv("AZURE_TTS_ENDPOINT", "https://vastai-openai-westus3.openai.azure.com/openai/deployments/tts-glimpsebyte-voiceover/audio/speech?api-version=2025-03-01-preview")
        self.azure_tts_key = os.getenv("AZURE_TTS_API_KEY")
        
        if not self.azure_tts_key:
            print("⚠️ WARNING: AZURE_TTS_API_KEY not set. Voiceover generation will fail.")
        else:
            print("✅ Azure TTS Client configured.")

    def _call_llm(self, messages: List[Dict], temperature: float = 0.5, json_mode: bool = False) -> str:
        try:
            model_to_use = self.deployment_name if self.use_azure else self.model
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
                max_tokens=4000
            )
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

    def summarize_actions(self, action_log: List[Dict]) -> List[Dict]:
        """
        AI Pass 1: Analyzes the raw action log and creates a detailed, 
        structured summary of steps.
        """
        print("🤖 AI Pass 1: Creating detailed structured summary...")
        system_prompt = (
            "You are a meticulous assistant that analyzes a user action log from a screen recording. "
            "Your task is to convert this raw log into a structured JSON object of chronological steps. "
            "The description for each step MUST be detailed and include the key UI elements or specific text "
            "that appeared or disappeared, such as button names, menu items, or page titles."
        )
        user_prompt = f"Based on the following JSON action log, create a detailed, structured summary. The output MUST be a JSON object with a 'steps' key, containing a list of objects, each with 'step_number', 'start_time', 'end_time', and a detailed 'description'.\n\nACTION LOG:\n{json.dumps(action_log, indent=2)}"
        
        response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.1, json_mode=True)
        
        structured_summary = self._robust_json_parser(response_str)
        if structured_summary:
            print("✅ Detailed summary generated.")
            return structured_summary.get("steps", [])
        return []

    def select_relevant_segments(self, structured_summary: List[Dict], user_prompt: str) -> List[Dict]:
        """
        AI Pass 2: Selects which steps from the summary are relevant
        to the user's video prompt.
        """
        print("🤖 AI Pass 2: Selecting relevant segments...")
        system_prompt = "You are an intelligent video segment selector. Your goal is to choose enough relevant steps to create a guide video that is approximately 60 seconds long. Respond ONLY with a JSON object containing a list of the relevant step numbers."
        
        user_prompt_for_selection = (
            f"Here is a structured summary of actions:\n\n{json.dumps(structured_summary, indent=2)}\n\nThe user wants a video guide about: '{user_prompt}'\n\n"
            "Analyze the summary and select all step numbers that are thematically or directly relevant to the user's request. "
            "It is better to include a partially relevant step than to miss a potentially important one. "
            "Your selection should result in a total video duration of around 60 seconds. "
            "Return your answer as a JSON object with a single key 'relevant_steps' containing a list of integers."
        )
        
        response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_selection}], temperature=0.2, json_mode=True)
        
        selection_data = self._robust_json_parser(response_str)
        if selection_data:
            selected_indices = selection_data.get("relevant_steps", [])
            selected_steps = [step for step in structured_summary if step['step_number'] in selected_indices]
            total_duration = sum(step['end_time'] - step['start_time'] for step in selected_steps)
            print(f"✅ Selected {len(selected_steps)} actions with a total duration of {total_duration:.1f} seconds.")
            return selected_steps
        return []

    def generate_script(self, selected_steps: List[Dict], user_prompt: str) -> str:
        """
        AI Pass 3: Generates a single, cohesive voiceover script
        based on the selected steps.
        """
        print("🤖 AI Pass 3: Generating final voiceover script...")
        system_prompt = (
            "You are a professional scriptwriter for tutorial videos. Your tone is clear, friendly, and professional. "
            "Your output must be ONLY the voiceover script itself, perfectly formatted for text-to-speech."
        )
        
        formatted_actions = "".join([f"- {step['description']}\n" for step in selected_steps])

        user_prompt_for_script = (
        "Write a clear, natural-sounding professional voiceover script for a tutorial video about: "
        f"'{user_prompt}'.\n\n"
        "The script should directly guide viewers through the process using an engaging and confident tone — "
        "as if a skilled instructor is explaining each step.\n\n"
        "Base the narration strictly on the following key steps:\n"
        f"{formatted_actions}\n\n"
        "The script MUST include three parts:\n"
        "1. Intro – A concise (10–15 words) welcome that clearly introduces the topic and goal of the tutorial.\n"
        "2. Body – A detailed, step-by-step explanation that matches the provided actions. "
        "Each step should be described in practical, instructional language that clearly tells the viewer what to do and why.\n"
        "3. Outro – A short (10–15 words) closing line that reinforces what was learned and ends confidently.\n\n"
        "Guidelines:\n"
        "- Avoid vague or filler phrases like 'Next, we do something' or 'You can now proceed'.\n"
        "- Use active, instructive language (e.g., 'Click', 'Select', 'Open', 'Enter', 'Save').\n"
        "- Keep the flow conversational and natural for voiceover delivery.\n"
        "- Do NOT include any titles, stage directions, labels (like 'Intro:'), or non-spoken text.\n\n"
        "Return ONLY the complete, ready-to-record spoken script."
        )

        
        script = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_script}], temperature=0.7)
        print("✅ Script generated.")
        return script

    def create_intro_plan(self, user_prompt: str) -> str:
        """
        Generates a text prompt for the user to manually create
        a 'veo3' intro video.
        """
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
            return plan_data["intro_prompt"]
        return f"A cinematic intro about {user_prompt}." # Fallback

    def generate_intro_video(self, prompt: str, output_path: Path):
        """
        Placeholder for Veo3 generation.
        This function provides the user with the prompt and expected output path.
        """
        print("\n" + "="*50)
        print("🚨 MANUAL STEP REQUIRED 🚨")
        print("="*50)
        print("🤖 Please use Veo3 to generate your intro clip with the following prompt:")
        print(f"\nPROMPT: \"{prompt}\"\n")
        print(f"👉 Please save the generated video to this exact path:")
        print(f"{output_path.resolve()}")
        print("="*50 + "\n")
        
        # Ensure the directory exists so the user can save the file there
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # We return the path for the pipeline to use.
        return str(output_path)

    def create_voiceover(self, script: str, output_path: str):
        """Generates an MP3 voiceover file from the script text using Azure TTS."""
        if not self.azure_tts_key:
            print("❌ Cannot create voiceover: AZURE_TTS_API_KEY is not set.")
            return
        if not script:
            print("⚠️ Script is empty. Skipping voiceover generation.")
            return

        headers = {
            "api-key": self.azure_tts_key,
            "Content-Type": "application/json"
        }
        # Using 'nova' voice as a high-quality default.
        body = {
            "input": script,
            "voice": "nova" 
        }
        
        try:
            print(f"🔊 Calling Azure TTS endpoint to generate voiceover...")
            response = requests.post(self.azure_tts_endpoint, headers=headers, json=body, stream=True)
            
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=4096):
                        f.write(chunk)
                print(f"🎤 Voiceover saved successfully to {output_path}")
            else:
                print(f"⚠️ Could not create voiceover. Status: {response.status_code}, Response: {response.text}")
        
        except requests.exceptions.RequestException as e:
            print(f"❌ 'requests' library failed. Error: {e}")
        except Exception as e:
            print(f"⚠️ An error occurred during voiceover creation: {e}")
'''import openai
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import re
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

class AIManager:
    # __init__ and _robust_json_parser are unchanged
    def __init__(self):
        self.use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
        if self.use_azure:
            print("🔧 Configuring for Azure OpenAI...")
            self.api_key, self.azure_endpoint, self.deployment_name = (
                os.getenv("AZURE_OPENAI_API_KEY"),
                os.getenv("AZURE_OPENAI_ENDPOINT"),
                os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            )
            if not all([self.api_key, self.azure_endpoint, self.deployment_name]):
                raise ValueError("For Azure, please set all required environment variables.")
            self.client = openai.AzureOpenAI(
                api_key=self.api_key, api_version="2024-02-01", azure_endpoint=self.azure_endpoint
            )
            print("✅ Azure OpenAI Client configured.")
        else:
            print("🔧 Configuring for standard OpenAI...")
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.model = "gpt-4o"
            if not self.api_key:
                raise ValueError("For standard OpenAI, please set OPENAI_API_KEY.")
            self.client = openai.OpenAI(api_key=self.api_key)
            print("✅ Standard OpenAI Client configured.")

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
        FINAL VERSION: Pass 1 with a prompt for a high-level, concise summary.
        This will prevent the response from being too long and getting cut off.
        """
        print("🤖 AI Pass 1: Creating high-level summary...")
        system_prompt = (
            "You are an expert user interface analyst. Your task is to analyze a log of on-screen text changes "
            "and summarize the user's primary action in each step. Be concise and focus on the main goal of the user."
        )
        user_prompt = (
            "Based on the following action log, create a structured JSON summary. The 'description' for each step "
            "should be a short, high-level summary of the user's main action (e.g., 'User navigates to the main dashboard', "
            "'User opens the File menu', 'User searches for a template'). DO NOT list every single text change.\n\n"
            f"ACTION LOG:\n{json.dumps(action_log, indent=2)}"
        )
        
        response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.2, json_mode=True)
        
        structured_summary = self._robust_json_parser(response_str)
        if structured_summary:
            print("✅ High-level summary generated.")
            return structured_summary.get("steps", [])
        return []

    def select_relevant_segments(self, structured_summary: List[Dict], user_prompt: str) -> List[Dict]:
        """
        FINAL VERSION: Pass 2 with a prompt that encourages selecting more content.
        """
        print("🤖 AI Pass 2: Selecting relevant segments...")
        system_prompt = "You are an intelligent video segment selector. Your goal is to choose enough relevant steps to create a guide video that is approximately 60 seconds long. Respond ONLY with a JSON object."
        
        user_prompt_for_selection = (
            f"Here is a summary of actions:\n\n{json.dumps(structured_summary, indent=2)}\n\nThe user wants a video guide about: '{user_prompt}'\n\n"
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
            total_duration = sum(step.get('end_time', 0) - step.get('start_time', 0) for step in selected_steps)
            print(f"✅ Selected {len(selected_steps)} actions with a total duration of {total_duration:.1f} seconds.")
            return selected_steps
        return []

    def generate_script(self, selected_steps: List[Dict], user_prompt: str) -> str:
        """
        FINAL VERSION: Pass 3 with a prompt that enforces a proper script structure.
        """
        print("🤖 AI Pass 3: Generating final voiceover script...")
        system_prompt = (
            "You are a professional scriptwriter for tutorial videos. Your tone is clear, friendly, and professional. "
            "Your output must be ONLY the voiceover script itself, perfectly formatted for text-to-speech."
        )
        
        formatted_actions = "".join([f"- {step['description']}\n" for step in selected_steps])

        user_prompt_for_script = (
            "Write a professional voiceover script for a video tutorial about: "
            f"'{user_prompt}'.\n\n"
            "The script must be based on these key steps:\n"
            f"{formatted_actions}\n\n"
            "The script MUST have three parts:\n"
            "1. Intro: A brief, welcoming sentence (10-15 words) that introduces the topic, like 'Welcome! In this guide...'.\n"
            "2. Body: A step-by-step narration of the key actions provided above.\n"
            "3. Outro: A short concluding sentence (10-15 words) that wraps up the tutorial.\n\n"
            "Your response must contain ONLY the complete, spoken script. Do not include any extra text, titles, or labels like 'Intro:' or 'Body:'."
        )
        
        script = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_script}], temperature=0.7)
        print("✅ Script generated.")
        return script

    def create_voiceover(self, script: str, output_path: str):
        try:
            tts = gTTS(text=script, lang='en')
            tts.save(output_path)
            print(f"🎤 Voiceover saved successfully to {output_path}")
        except Exception as e:
            print(f"⚠️ Could not create voiceover: {e}")

##########################
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
            print("🔧 Configuring for Azure OpenAI...")
            self.api_key, self.azure_endpoint, self.deployment_name = (
                os.getenv("AZURE_OPENAI_API_KEY"),
                os.getenv("AZURE_OPENAI_ENDPOINT"),
                os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            )
            if not all([self.api_key, self.azure_endpoint, self.deployment_name]):
                raise ValueError("For Azure, please set all required environment variables.")
            self.client = openai.AzureOpenAI(
                api_key=self.api_key, api_version="2024-02-01", azure_endpoint=self.azure_endpoint
            )
            print("✅ Azure OpenAI Client configured.")
        else:
            print("🔧 Configuring for standard OpenAI...")
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.model = "gpt-4o"
            if not self.api_key:
                raise ValueError("For standard OpenAI, please set OPENAI_API_KEY.")
            self.client = openai.OpenAI(api_key=self.api_key)
            print("✅ Standard OpenAI Client configured.")

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
        """More robust JSON parser that handles markdown code blocks."""
        try:
            cleaned_string = re.sub(r'```json\s*|\s*```', '', json_string, flags=re.DOTALL)
            return json.loads(cleaned_string)
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON. Error: {e}\nRaw response:\n{json_string}")
            return None

    def summarize_actions(self, concise_log: List[Dict]) -> List[Dict]:
        print("🤖 AI Pass 1: Creating high-level summary...")
        system_prompt = (
            "You are an expert user interface analyst. Your task is to analyze a log of on-screen text changes "
            "and summarize the user's primary action in each step. Be concise and focus on the main goal of the user."
        )
        user_prompt = (
            "Based on the following CONCISE action log, create a structured JSON summary. The 'description' for each step "
            "should be a short, high-level summary of the user's main action (e.g., 'User navigates to the main dashboard', "
            "'User opens the File menu', 'User searches for a template'). DO NOT list every single text change.\n\n"
            f"ACTION LOG:\n{json.dumps(concise_log, indent=2)}"
        )
        
        response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.2, json_mode=True)
        
        structured_summary = self._robust_json_parser(response_str)
        if structured_summary:
            print("✅ High-level summary generated.")
            return structured_summary.get("steps", [])
        return []

    def select_relevant_segments(self, structured_summary: List[Dict], user_prompt: str) -> List[Dict]:
        print("🤖 AI Pass 2: Selecting relevant segments...")
        system_prompt = "You are an intelligent video segment selector. Your goal is to choose enough relevant steps to create a guide video that is approximately 60 seconds long. Respond ONLY with a JSON object."
        
        user_prompt_for_selection = (
            f"Here is a summary of actions:\n\n{json.dumps(structured_summary, indent=2)}\n\nThe user wants a video guide about: '{user_prompt}'\n\n"
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
            total_duration = sum(step.get('end_time', 0) - step.get('start_time', 0) for step in selected_steps)
            print(f"✅ Selected {len(selected_steps)} actions with a total duration of {total_duration:.1f} seconds.")
            return selected_steps
        return []

    def generate_script(self, selected_steps: List[Dict], user_prompt: str) -> str:
        print("🤖 AI Pass 3: Generating final voiceover script...")
        system_prompt = (
            "You are a professional scriptwriter for tutorial videos. Your tone is clear, friendly, and professional. "
            "Your output must be ONLY the voiceover script itself, perfectly formatted for text-to-speech."
        )
        
        formatted_actions = "".join([f"- {step['description']}\n" for step in selected_steps])

        user_prompt_for_script = (
            "Write a professional voiceover script for a video tutorial about: "
            f"'{user_prompt}'.\n\n"
            "The script must be based on these key steps:\n"
            f"{formatted_actions}\n\n"
            "The script MUST have three parts:\n"
            "1. Intro: A brief, welcoming sentence (10-15 words) that introduces the topic, like 'Welcome! In this guide...'.\n"
            "2. Body: A step-by-step narration of the key actions provided above.\n"
            "3. Outro: A short concluding sentence (10-15 words) that wraps up the tutorial.\n\n"
            "Your response must contain ONLY the complete, spoken script. Do not include any extra text, titles, or labels like 'Intro:' or 'Body:'."
        )
        
        script = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_script}], temperature=0.7)
        print("✅ Script generated.")
        return script

    def create_voiceover(self, script: str, output_path: str):
        try:
            tts = gTTS(text=script, lang='en')
            tts.save(output_path)
            print(f"🎤 Voiceover saved successfully to {output_path}")
        except Exception as e:
            print(f"⚠️ Could not create voiceover: {e}")

'''

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
            print("🔧 Configuring for Azure OpenAI...")
            self.api_key, self.azure_endpoint, self.deployment_name = (
                os.getenv("AZURE_OPENAI_API_KEY"),
                os.getenv("AZURE_OPENAI_ENDPOINT"),
                os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            )
            if not all([self.api_key, self.azure_endpoint, self.deployment_name]):
                raise ValueError("For Azure, please set all required environment variables.")
            self.client = openai.AzureOpenAI(
                api_key=self.api_key, api_version="2024-02-01", azure_endpoint=self.azure_endpoint
            )
            print("✅ Azure OpenAI Client configured.")
        else:
            print("🔧 Configuring for standard OpenAI...")
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.model = "gpt-4o"
            if not self.api_key:
                raise ValueError("For standard OpenAI, please set OPENAI_API_KEY.")
            self.client = openai.OpenAI(api_key=self.api_key)
            print("✅ Standard OpenAI Client configured.")

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
    '''
    def summarize_actions(self, steps_with_data: List[Dict]) -> List[Dict]:
        """
        CORRECTED: Analyzes pre-defined steps (with start/end times and text changes)
        and generates a high-level summary description for each one.
        """
        print("🤖 AI Pass 1: Summarizing action steps...")
        system_prompt = (
            "You are an expert user interface analyst. Your task is to analyze a list of user interaction steps, "
            "each defined by a start time, end time, and the text that appeared or disappeared on screen. "
            "For each step, you will write a concise, high-level description of the user's action. "
            "Your output MUST be a JSON object containing a 'steps' key, which is a list of these analyzed steps. "
            "Crucially, you MUST preserve the original 'step_number', 'start_time', and 'end_time' for each step."
        )
        user_prompt = (
            "Based on the following log of steps, generate a concise 'description' for each one. "
            "Focus on the primary user action (e.g., 'User navigates to the dashboard', 'User searches for a template'). "
            "Do not list every single text change in your description. "
            "Return the complete list of steps, including your new description and the original timing data.\n\n"
            f"STEPS LOG:\n{json.dumps(steps_with_data, indent=2)}"
        )
        
        response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.2, json_mode=True)
        
        structured_summary = self._robust_json_parser(response_str)
        if structured_summary and "steps" in structured_summary:
            print("✅ High-level summary generated for all steps.")
            return structured_summary.get("steps", [])
        print("⚠️ Failed to generate structured summary or 'steps' key was missing.")
        return []

    def select_relevant_segments(self, structured_summary: List[Dict], user_prompt: str) -> List[Dict]:
        """
        Pass 2: Selects segments relevant to the user's prompt.
        """
        print("🤖 AI Pass 2: Selecting relevant segments...")
        system_prompt = "You are an intelligent video segment selector. Your goal is to choose enough relevant steps to create a coherent guide video. Respond ONLY with a JSON object."
        
        user_prompt_for_selection = (
            f"Here is a summary of actions:\n\n{json.dumps(structured_summary, indent=2)}\n\nThe user wants a video guide about: '{user_prompt}'\n\n"
            "Analyze the summary and select all step numbers that are thematically or directly relevant to the user's request. "
            "It is better to include a partially relevant step than to miss a potentially important one. "
            "Return your answer as a JSON object with a single key 'relevant_steps' containing a list of integers (the step numbers)."
        )
        
        response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_selection}], temperature=0.2, json_mode=True)
        
        selection_data = self._robust_json_parser(response_str)
        if selection_data and "relevant_steps" in selection_data:
            selected_indices = selection_data.get("relevant_steps", [])
            selected_steps = [step for step in structured_summary if step.get('step_number') in selected_indices]
            total_duration = sum(step.get('end_time', 0) - step.get('start_time', 0) for step in selected_steps)
            print(f"✅ Selected {len(selected_steps)} steps with a total duration of {total_duration:.1f} seconds.")
            return selected_steps
        return []

    def generate_script(self, selected_steps: List[Dict], user_prompt: str) -> str:
        """
        Pass 3: Generates the final voiceover script from the selected steps.
        """
        print("🤖 AI Pass 3: Generating final voiceover script...")
        system_prompt = (
            "You are a professional scriptwriter for tutorial videos. Your tone is clear, friendly, and professional. "
            "Your output must be ONLY the voiceover script itself, perfectly formatted for text-to-speech."
        )
        
        formatted_actions = "".join([f"- Step {step.get('step_number')}: {step['description']}\n" for step in selected_steps])

        user_prompt_for_script = (
            "Write a professional voiceover script for a video tutorial about: "
            f"'{user_prompt}'.\n\n"
            "The script must be based on these key steps:\n"
            f"{formatted_actions}\n\n"
            "The script MUST have three parts:\n"
            "1. Intro: A brief, welcoming sentence (10-15 words) that introduces the topic.\n"
            "2. Body: A step-by-step narration of the key actions provided above. Refer to the actions smoothly.\n"
            "3. Outro: A short concluding sentence (10-15 words) that wraps up the tutorial.\n\n"
            "Your response must contain ONLY the complete, spoken script. Do not include any extra text, titles, or labels like 'Intro:' or 'Body:'."
        )
        
        script = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_script}], temperature=0.7)
        print("✅ Script generated.")
        return script
    '''
    def create_voiceover(self, script: str, output_path: str):
        """Generates an MP3 voiceover file from the script text."""
        if not script:
            print("⚠️ Script is empty. Skipping voiceover generation.")
            return
        try:
            tts = gTTS(text=script, lang='en')
            tts.save(output_path)
            print(f"🎤 Voiceover saved successfully to {output_path}")
        except Exception as e:
            print(f"⚠️ Could not create voiceover: {e}")


    def summarize_actions(self, action_log: List[Dict]) -> List[Dict]:
        """
        UPDATED: Pass 1 with a prompt that demands more detail in the summary.
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
        UPDATED: Pass 2 with a prompt that encourages selecting more content.
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
        UPDATED: Pass 3 with a prompt that enforces a proper script structure.
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

    
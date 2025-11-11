import openai
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import re
import requests
from dotenv import load_dotenv

# --- NEW IMPORTS FOR RUNWAYML ---
from runwayml import RunwayML, TaskFailedError
# --------------------------------

load_dotenv()

class AIManager:
    # --- __init__, _call_llm, _robust_json_parser ---
    # (These are unchanged)
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

    # --- NEW "SMART" SINGLE-PASS FUNCTION ---
    # (This is your chosen function, unchanged)
    def create_clip_plan(self, action_log: List[Dict], user_prompt: str) -> List[Dict]:
        """
        AI Pass 1 (and only): Converts the raw event log directly into a
        final, timed clip plan based on the user's prompt.
        """
        print("🤖 AI Pass 1: Generating final clip plan from raw events...")
        
        system_prompt = (
            "You are an expert AI video editor. Your job is to create a perfect 'clip plan' "
            "by analyzing a raw UI event log and a user's prompt. "
            "You MUST scan the *entire* log to find the *best* matching events. "
            "Your output MUST be a JSON object with a single key 'clip_plan'."
        )

        user_prompt_for_plan = f'''Here is the raw UI event log from a screen recording:
{json.dumps(action_log, indent=2)}

Here is the user's request:
"{user_prompt}"

**Your Task (Follow exactly):**
1.  **Analyze Request:** Identify the key actions in the user's request (e.g., "create a new page", "share a workspace").
2.  **Scan and Match:** Read the *entire* event log from start to finish. Find the *specific events* that match these key actions. Look for `text_added` or `text_removed` fields that literally contain the keywords (e.g., "Create new page", "Share", "Untitled", etc.).
3.  **IGNORE ALL NOISE (CRITICAL):** The log is noisy. You MUST ignore all events *not* directly related to the user's request.
    * **IGNORE:** Clicks on "Welcome", "Check the Basics", "Get Inspired", "Next Steps", "Send Feedback", "Recycle bin".
    * **IGNORE:** Random scrolling or clicks on existing text.
    * **ONLY MATCH:** The specific actions requested, such as the *actual click* on "Create new page" (around 43s) or the "Share" button (around 103s).
4.  **Create Clips & Apply "Smart Timings":**
    For *only* the matched, relevant events, create a list of clip objects. Each object must have `start_time`, `end_time`, and `description`.
    
    * **Rule A (Simple Click):** For a simple click action (like 'User clicks Create new page' or 'User clicks Share'):
        * `start_time`: Must be **1.0 second *before*** the event's 'timestamp'.
        * `end_time`: Must be **2.0 seconds *after*** the event's 'timestamp'.
        * `description`: A clear description (e.g., "User clicks 'Create new page'").
    
    * **Rule B (Click + Type):** If a click is immediately followed by typing (like 'User clicks page title' and 'User types...'):
        * `start_time`: **1.0 second *before*** the *click* event.
        * `end_time`: Timestamp of the *last* typing event.
        * `description`: A summary (e.g., "User types the new page title 'GlimpseByte Updates'").

5.  **Return JSON:** Return a JSON object with a single key "clip_plan" containing the final, ordered list of clip objects.
'''
        
        response_str = self._call_llm(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_plan}],
            temperature=0.1, json_mode=True
        )
        
        parsed_data = self._robust_json_parser(response_str)
        if parsed_data and "clip_plan" in parsed_data:
            print(f"✅ AI successfully generated a clip plan with {len(parsed_data['clip_plan'])} steps.")
            return parsed_data["clip_plan"]
        
        print("⚠️ AI failed to generate a valid clip plan.")
        return []

    # --- generate_narration (Unchanged) ---
    def generate_narration(
        self,
        main_prompt: str,
        step_description: str,
        target_duration: float,
        is_intro: bool = False,
        is_outro: bool = False
    ) -> str:
        
        system_prompt = (
            "You are a professional, clear scriptwriter for 'how-to' tutorial videos. "
            "You write in a helpful, instructional tone, telling the user what to do. "
            "Your response must be ONLY the voiceover script itself."
            "**CRITICAL**: Do NOT include any calls to action like 'please like, share, or subscribe' or any similar phrases."
        )
        
        task_prompt = ""
        if is_intro:
            task_prompt = (
                f"Write a brief, engaging intro for a video about '{main_prompt}'. "
                f"The intro clip is {target_duration:.1f} seconds long. The script MUST be readable in this time."
            )
        elif is_outro:
            task_prompt = (
                f"Write a brief, concluding outro for a video about '{main_prompt}'. "
                f"The action in this final clip is: '{step_description}'. "
                f"The clip is {target_duration:.1f} seconds long. Conclude the tutorial (e.g., 'And that's it! Thanks for watching.'). The script MUST be readable in this time."
            )
        else:
            task_prompt = (
                f"The user wants to learn about: '{main_prompt}'.\n"
                f"The current step is: '{step_description}'.\n"
                f"The video clip for this step is {target_duration:.1f} seconds long.\n\n"
                "Write a concise, instructional script that *guides the user* through this step. "
                "Speak in the second person (e.g., 'First, you'll click...'). "
                "**CRITICAL**: Your entire script MUST be readable in {target_duration:.1f} seconds. Be extremely concise."
            )

        script = self._call_llm(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": task_prompt}],
            temperature=0.7
        )
        return script

    # --- polish_scripts (Unchanged) ---
    def polish_scripts(self, script_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("🤖 AI Pass 3: Polishing scripts for smooth narration...")
        if not script_data:
            return []
            
        system_prompt = (
            "You are an expert video script editor. You will receive a JSON list of script 'segments', each with a script and its duration. "
            "Your job is to rewrite the scripts to flow together as one continuous, professional narration. "
            "Add transition words (e.g., 'Next,', 'After that,', 'Now that you've...'). "
            "**CRITICAL**: You MUST respect the `target_duration` for each segment. Do NOT make a script longer than its duration allows. "
            "Do NOT add any intro or outro text, just polish the body scripts."
            "Respond with a JSON object with a single key 'polished_scripts', which is a list of the *new* script strings, in the same order."
        )

        user_prompt = (
            "Here is the list of script segments. Please polish them to flow together.\n\n"
            f"{json.dumps(script_data, indent=2)}\n\n"
            "Return a JSON object with a 'polished_scripts' key, containing only the list of new, polished script strings."
        )
        
        response_str = self._call_llm(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.5,
            json_mode=True
        )
        
        parsed_response = self._robust_json_parser(response_str)
        
        if parsed_response and "polished_scripts" in parsed_response:
            polished_scripts = parsed_response["polished_scripts"]
            if len(polished_scripts) == len(script_data):
                for i, segment in enumerate(script_data):
                    segment["script"] = polished_scripts[i]
                print("✅ Scripts successfully polished.")
                return script_data
            else:
                print("⚠️ AI returned a different number of scripts. Using original scripts.")
                return script_data 
        else:
            print("⚠️ Failed to parse polished scripts. Using original scripts.")
            return script_data
            
    # --- create_intro_plan (Unchanged) ---
    def create_intro_plan(self, user_prompt: str) -> str:
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

    # --- generate_intro_video (MODIFIED WITH RUNWAY API) ---
    def generate_intro_video(self, prompt: str, output_path: Path):
        print(f"\n🤖 Calling Runway API to generate intro video...")
        print(f"   Prompt: \"{prompt}\"")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not os.getenv("RUNWAYML_API_SECRET"):
            print("❌ Error: RUNWAYML_API_SECRET not set in .env file.")
            print("   Skipping automated intro generation.")
            return None
        
        try:
            client = RunwayML() 
            
            print(f"   Task started. Waiting for completion (this may take 1-3 minutes)...")
            completed_task = client.text_to_video.create(
                model='veo3.1_fast',
                prompt_text=prompt,
                ratio='720:1280',  # 9:16 aspect ratio
                duration=6       
            ).wait_for_task_output() 
            
            video_url = completed_task.output[0]
            print(f"   ✅ Generation complete. Downloading from {video_url}...")
            
            response = requests.get(video_url)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"   💾 Intro video saved to {output_path}")
                return str(output_path)
            else:
                print(f"   ❌ Failed to download video. Status: {response.status_code}")
                return None

        except TaskFailedError as e:
            print(f"   ❌ Runway task failed: {e.taskDetails}")
            return None
        except Exception as e:
            print(f"   ❌ An error occurred during intro generation: {e}")
            return None

    # --- create_voiceover (Unchanged) ---
    def create_voiceover(self, script: str, output_path: str, voice: str = "nova"):
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
        body = {"input": script, "voice": voice}
        
        try:
            print(f"🔊 Calling Azure TTS endpoint for clip (Voice: {voice})...")
            response = requests.post(self.azure_tts_endpoint, headers=headers, json=body, stream=True)
            
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=4096):
                        f.write(chunk)
                print(f"🎤 Clip voiceover saved to {output_path}")
            else:
                print(f"⚠️ Could not create voiceover. Status: {response.status_code}, Response: {response.text}")
        
        except requests.exceptions.RequestException as e:
            print(f"❌ 'requests' library failed. Error: {e}")
        except Exception as e:
            print(f"⚠️ An error occurred during voiceover creation: {e}")

     

# import openai
# import json
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# import os
# import re
# import requests
# from dotenv import load_dotenv

# load_dotenv()

# class AIManager:
#     # --- __init__, _call_llm, _robust_json_parser ---
#     # (These are unchanged from your file)
#     def __init__(self):
#         # --- OpenAI LLM Config ---
#         self.use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
#         if self.use_azure:
#             print("🔧 Configuring for Azure OpenAI (LLM)...")
#             self.api_key, self.azure_endpoint, self.deployment_name = (
#                 os.getenv("AZURE_OPENAI_API_KEY"),
#                 os.getenv("AZURE_OPENAI_ENDPOINT"),
#                 os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
#             )
#             if not all([self.api_key, self.azure_endpoint, self.deployment_name]):
#                 raise ValueError("For Azure LLM, please set all required environment variables.")
#             self.client = openai.AzureOpenAI(
#                 api_key=self.api_key, api_version="2024-02-01", azure_endpoint=self.azure_endpoint
#             )
#             print("✅ Azure OpenAI (LLM) Client configured.")
#         else:
#             print("🔧 Configuring for standard OpenAI (LLM)...")
#             self.api_key = os.getenv("OPENAI_API_KEY")
#             self.model = "gpt-4o"
#             if not self.api_key:
#                 raise ValueError("For standard OpenAI (LLM), please set OPENAI_API_KEY.")
#             self.client = openai.OpenAI(api_key=self.api_key)
#             print("✅ Standard OpenAI (LLM) Client configured.")

#         # --- Azure TTS Config ---
#         print("🔧 Configuring for Azure TTS...")
#         self.azure_tts_endpoint = os.getenv("AZURE_TTS_ENDPOINT", "https://vastai-openai-westus3.openai.azure.com/openai/deployments/tts-glimpsebyte-voiceover/audio/speech?api-version=2025-03-01-preview")
#         self.azure_tts_key = os.getenv("AZURE_TTS_API_KEY")
        
#         if not self.azure_tts_key:
#             print("⚠️ WARNING: AZURE_TTS_API_KEY not set. Voiceover generation will fail.")
#         else:
#             print("✅ Azure TTS Client configured.")

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

#     # --- NEW "SMART" SINGLE-PASS FUNCTION ---
#     def create_clip_plan(self, action_log: List[Dict], user_prompt: str) -> List[Dict]:
#         """
#         AI Pass 1 (and only): Converts the raw event log directly into a
#         final, timed clip plan based on the user's prompt.
#         """
#         print("🤖 AI Pass 1: Generating final clip plan from raw events...")
        
#         system_prompt = (
#             "You are an expert AI video editor. Your job is to create a perfect 'clip plan' "
#             "by analyzing a raw UI event log and a user's prompt. "
#             "You MUST scan the *entire* log to find the *best* matching events. "
#             "Your output MUST be a JSON object with a single key 'clip_plan'."
#         )

#         user_prompt_for_plan = f'''Here is the raw UI event log from a screen recording:
# {json.dumps(action_log, indent=2)}

# Here is the user's request:
# "{user_prompt}"

# **Your Task (Follow exactly):**
# 1.  **Analyze Request:** Identify the key actions in the user's request (e.g., "create a new page", "share a workspace").
# 2.  **Scan and Match:** Read the *entire* event log from start to finish. Find the *specific events* that match these key actions. Look for `text_added` or `text_removed` fields that literally contain the keywords (e.g., "Create new page", "Share", "Untitled", etc.).
# 3.  **IGNORE ALL NOISE (CRITICAL):** The log is noisy. You MUST ignore all events *not* directly related to the user's request.
#     * **IGNORE:** Clicks on "Welcome", "Check the Basics", "Get Inspired", "Next Steps", "Send Feedback", "Recycle bin".
#     * **IGNORE:** Random scrolling or clicks on existing text.
#     * **ONLY MATCH:** The specific actions requested, such as the *actual click* on "Create new page" (around 43s) or the "Share" button (around 103s).
# 4.  **Create Clips & Apply "Smart Timings":**
#     For *only* the matched, relevant events, create a list of clip objects. Each object must have `start_time`, `end_time`, and `description`.
    
#     * **Rule A (Simple Click):** For a simple click action (like 'User clicks Create new page' or 'User clicks Share'):
#         * `start_time`: Must be **1.0 second *before*** the event's 'timestamp'.
#         * `end_time`: Must be **2.0 seconds *after*** the event's 'timestamp'.
#         * `description`: A clear description (e.g., "User clicks 'Create new page'").
    
#     * **Rule B (Click + Type):** If a click is immediately followed by typing (like 'User clicks page title' and 'User types...'):
#         * `start_time`: **1.0 second *before*** the *click* event.
#         * `end_time`: Timestamp of the *last* typing event.
#         * `description`: A summary (e.g., "User types the new page title 'GlimpseByte Updates'").

# 5.  **Return JSON:** Return a JSON object with a single key "clip_plan" containing the final, ordered list of clip objects.
# '''
        
#         response_str = self._call_llm(
#             [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_plan}],
#             temperature=0.1, json_mode=True
#         )
        
#         parsed_data = self._robust_json_parser(response_str)
#         if parsed_data and "clip_plan" in parsed_data:
#             print(f"✅ AI successfully generated a clip plan with {len(parsed_data['clip_plan'])} steps.")
#             return parsed_data["clip_plan"]
        
#         print("⚠️ AI failed to generate a valid clip plan.")
#         return []

#     # --- generate_narration (Unchanged from your file) ---
#     def generate_narration(
#         self,
#         main_prompt: str,
#         step_description: str,
#         target_duration: float,
#         is_intro: bool = False,
#         is_outro: bool = False
#     ) -> str:
        
#         system_prompt = (
#             "You are a professional, clear scriptwriter for 'how-to' tutorial videos. "
#             "You write in a helpful, instructional tone, telling the user what to do. "
#             "Your response must be ONLY the voiceover script itself."
#             "**CRITICAL**: Do NOT include any calls to action like 'please like, share, or subscribe' or any similar phrases."
#         )
        
#         task_prompt = ""
#         if is_intro:
#             task_prompt = (
#                 f"Write a brief, engaging intro for a video about '{main_prompt}'. "
#                 f"The intro clip is {target_duration:.1f} seconds long. The script MUST be readable in this time."
#             )
#         elif is_outro:
#             task_prompt = (
#                 f"Write a brief, concluding outro for a video about '{main_prompt}'. "
#                 f"The action in this final clip is: '{step_description}'. "
#                 f"The clip is {target_duration:.1f} seconds long. Conclude the tutorial (e.g., 'And that's it! Thanks for watching.'). The script MUST be readable in this time."
#             )
#         else:
#             task_prompt = (
#                 f"The user wants to learn about: '{main_prompt}'.\n"
#                 f"The current step is: '{step_description}'.\n"
#                 f"The video clip for this step is {target_duration:.1f} seconds long.\n\n"
#                 "Write a concise, instructional script that *guides the user* through this step. "
#                 "Speak in the second person (e.g., 'First, you'll click...'). "
#                 "**CRITICAL**: Your entire script MUST be readable in {target_duration:.1f} seconds. Be extremely concise."
#             )

#         script = self._call_llm(
#             [{"role": "system", "content": system_prompt}, {"role": "user", "content": task_prompt}],
#             temperature=0.7
#         )
#         return script

#     # --- polish_scripts (Unchanged from your file) ---
#     def polish_scripts(self, script_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         print("🤖 AI Pass 3: Polishing scripts for smooth narration...")
#         if not script_data:
#             return []
            
#         system_prompt = (
#             "You are an expert video script editor. You will receive a JSON list of script 'segments', each with a script and its duration. "
#             "Your job is to rewrite the scripts to flow together as one continuous, professional narration. "
#             "Add transition words (e.g., 'Next,', 'After that,', 'Now that you've...'). "
#             "**CRITICAL**: You MUST respect the `target_duration` for each segment. Do NOT make a script longer than its duration allows. "
#             "Do NOT add any intro or outro text, just polish the body scripts."
#             "Respond with a JSON object with a single key 'polished_scripts', which is a list of the *new* script strings, in the same order."
#         )

#         user_prompt = (
#             "Here is the list of script segments. Please polish them to flow together.\n\n"
#             f"{json.dumps(script_data, indent=2)}\n\n"
#             "Return a JSON object with a 'polished_scripts' key, containing only the list of new, polished script strings."
#         )
        
#         response_str = self._call_llm(
#             [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
#             temperature=0.5,
#             json_mode=True
#         )
        
#         parsed_response = self._robust_json_parser(response_str)
        
#         if parsed_response and "polished_scripts" in parsed_response:
#             polished_scripts = parsed_response["polished_scripts"]
#             if len(polished_scripts) == len(script_data):
#                 for i, segment in enumerate(script_data):
#                     segment["script"] = polished_scripts[i]
#                 print("✅ Scripts successfully polished.")
#                 return script_data
#             else:
#                 print("⚠️ AI returned a different number of scripts. Using original scripts.")
#                 return script_data 
#         else:
#             print("⚠️ Failed to parse polished scripts. Using original scripts.")
#             return script_data
            
#     # --- create_intro_plan & generate_intro_video (Unchanged from your file) ---
#     def create_intro_plan(self, user_prompt: str) -> str:
#         print("🤖 AI Director: Creating intro plan...")
#         system_prompt = "You are an AI Video Director. Create a single, creative prompt for the 'veo3' model to generate a cinematic intro video. Respond ONLY with a valid JSON object."
#         user_prompt_for_plan = (
#             f"The final video is a tutorial about: '{user_prompt}'. "
#             "Write one creative, exciting prompt for a 4-second intro video. "
#             "The prompt should be visually descriptive and related to the topic. "
#             "Return a JSON object with a single key 'intro_prompt' containing the prompt string."
#         )
#         response_str = self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_for_plan}], temperature=0.8, json_mode=True)
#         plan_data = self._robust_json_parser(response_str)
#         if plan_data and "intro_prompt" in plan_data:
#             return plan_data["intro_prompt"]
#         return f"A cinematic intro about {user_prompt}." # Fallback

#     def generate_intro_video(self, prompt: str, output_path: Path):
#         print("\n" + "="*50)
#         print("🚨 MANUAL STEP REQUIRED 🚨")
#         print("="*50)
#         print("🤖 Please use Veo3 to generate your intro clip with the following prompt:")
#         print(f"\nPROMPT: \"{prompt}\"\n")
#         print(f"👉 Please save the generated video to this exact path:")
#         print(f"{output_path.resolve()}")
#         print("="*50 + "\n")
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         return str(output_path)

#     # --- create_voiceover (Unchanged from your file) ---
#     def create_voiceover(self, script: str, output_path: str):
#         if not self.azure_tts_key:
#             print("❌ Cannot create voiceover: AZURE_TTS_API_KEY is not set.")
#             return
#         if not script:
#             print("⚠️ Script is empty. Skipping voiceover generation.")
#             return

#         headers = {
#             "api-key": self.azure_tts_key,
#             "Content-Type": "application/json"
#         }
#         body = {"input": script, "voice": "nova"}
        
#         try:
#             print(f"🔊 Calling Azure TTS endpoint for clip...")
#             response = requests.post(self.azure_tts_endpoint, headers=headers, json=body, stream=True)
            
#             if response.status_code == 200:
#                 with open(output_path, "wb") as f:
#                     for chunk in response.iter_content(chunk_size=4096):
#                         f.write(chunk)
#                 print(f"🎤 Clip voiceover saved to {output_path}")
#             else:
#                 print(f"⚠️ Could not create voiceover. Status: {response.status_code}, Response: {response.text}")
        
#         except requests.exceptions.RequestException as e:
#             print(f"❌ 'requests' library failed. Error: {e}")
#         except Exception as e:
#             print(f"⚠️ An error occurred during voiceover creation: {e}")



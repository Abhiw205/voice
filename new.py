import cohere
import os
import json
import logging
import sounddevice as sd
from scipy.io.wavfile import write
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from datetime import datetime
import difflib
import time
import random

import random

VOCAB_SYNONYMS = [
    {"word": "happy", "type": "synonym", "hint": "joyful, cheerful"},
    {"word": "fast", "type": "synonym", "hint": "quick, speedy"},
    {"word": "smart", "type": "synonym", "hint": "intelligent, clever"},
    {"word": "cold", "type": "synonym", "hint": "chilly, freezing"}
]

VOCAB_ANTONYMS = [
    {"word": "strong", "type": "antonym", "hint": "weak, feeble"},
    {"word": "easy", "type": "antonym", "hint": "hard, difficult"},
    {"word": "hot", "type": "antonym", "hint": "cold, chilly"},
    {"word": "tall", "type": "antonym", "hint": "short, tiny"}
]

AUDIO_FILENAME = "input_audio.wav"
DEFAULT_DURATION = 5
SIMILARITY_THRESHOLD = 0.90

FIELD_RECORDING_DURATION = {
    "summary": 7  # Duration for summary recording
}

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION", "eastus")

co = cohere.Client(api_key=COHERE_API_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModuleRunner:
    def __init__(self, config_path, gui_mode=False, ui_callback=None):
        self.config = self.load_config(config_path)
        self.fields = self.config.get("fields", [])
        self.filled_fields = {}
        self.expected_summary = ""
        self.field_results = {}
        self.field_scores = {}
        self.last_prompt = ""
        self.last_hint = ""
        self.last_feedback = ""
        self.last_result = {}
        self.gui_mode = gui_mode
        self.ui_callback = ui_callback
        self.current_field_index = 0
        self.attempts = {}

    def emit_welcome_and_prompt(self):
        self.current_field_index = 0
        if "welcome" in self.config:
            self.notify_ui("welcome", self.config["welcome"])
        self.prompt_current_field()
    def emit_closing_if_needed(self):
        if "closing" in self.config:
            self.notify_ui("closing", self.config["closing"])
    def emit_static_closing(self):
        closing_message = "This module is completed. Please select the next one."
        logging.info(f"Emitting static closing message: {closing_message}")
        self.notify_ui("closing", closing_message)
    def prompt_current_field(self):
        if self.current_field_index >= len(self.fields):
            self.perform_validation_and_feedback()
            self.emit_static_closing()
            return

        field = self.fields[self.current_field_index]
        key = field["key"]
        self.last_prompt = field.get("prompt", "")
        self.last_hint = field.get("hint", "")
        self.notify_ui("prompt", self.last_prompt, duration=field.get("duration", DEFAULT_DURATION))

        if self.last_hint:
            self.notify_ui("hint", self.last_hint)

        self.notify_ui("status", "recording")

    def submit_response(self, text, field_key=None):
        if self.current_field_index >= len(self.fields):
            return

        field = self.fields[self.current_field_index]
        key = field["key"]

        if field_key and field_key != key:
            logging.warning(f"Unexpected field key: {field_key} (expected {key})")

        feedback = self.last_result.get("feedback")
        if feedback:
            self.notify_ui("feedback", feedback)

        self.notify_ui("status", "validating")
        self.notify_ui("transcript", text)

        result = self.handle_llm_behavior(field, text)
        if result:
            self.filled_fields[key] = result
        else:
            self.filled_fields[key] = "[Unrecognized or skipped]"

        self.last_result["feedback"] = self.filled_fields.get("feedback")

        # Early check for field_scores set by LLM validation
        passed = self.field_scores.get(key, False)
        if passed:
            self.current_field_index += 1
            self.last_result = {}
            self.prompt_current_field()
            return

        max_attempts = field.get("max_attempts", 2)
        attempts = self.attempts.get(key, 0) + 1
        self.attempts[key] = attempts

        if self.field_scores.get(key, False):
            self.current_field_index += 1
            self.last_result = {}
            self.prompt_current_field()
        elif attempts < max_attempts:
            self.notify_ui("retry", "Hmm, that wasn't quite right. Let's try again.")
            self.last_result = {}
            self.prompt_current_field()
        else:
            self.current_field_index += 1
            self.last_result = {}
            self.prompt_current_field()

    def perform_validation_and_feedback(self):
        if self.config.get("validation_logic"):
            self.notify_ui("status", "validating")
            self.run_validation_logic()
        else:
            self.notify_ui("status", "Module complete — no validation logic.")

        filename = f"{self.config['module']}_output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(filename, 'w') as f:
            json.dump({**self.filled_fields, "field_results": self.field_results}, f, indent=2)
        self.notify_ui("status", f"Results saved to {filename}")
        # self.emit_static_closing()
    def mark_audio_ready(self):
        self.audio_ready = True
    def notify_ui(self, label, message, **kwargs):
        if self.gui_mode and self.ui_callback:
            payload = {"label": label, "message": message}
            payload.update(kwargs)
            self.ui_callback(payload)
        else:
            print(f"[{label.upper()}]: {message}")
            if label in ("prompt", "retry", "feedback"):
                # self.speak(message)
                pass

    def load_config(self, path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "configs", path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)

        if config.get("random_prompt_pool"):
            prompt_fields = config.get("fields", [])
            for field in prompt_fields:
                if "prompt_pool" in field:
                    field["prompt"] = random.choice(field["prompt_pool"])

        if config.get("dynamic_vocab"):
            vocab_fields = config.get("fields", [])
            for field in vocab_fields:
                if field.get("llm_behavior") == "validate_synonym_antonym":
                    word_type = field.get("word_type", "synonym")
                    if word_type == "synonym":
                        vocab = random.choice(VOCAB_SYNONYMS)
                    else:
                        vocab = random.choice(VOCAB_ANTONYMS)
                    field["word"] = vocab["word"]
                    field["hint"] = vocab["hint"]
                    field["prompt"] = f"Give a {word_type} for '{vocab['word']}'."
        return config

    # def speak(self, text):
    #     try:
    #         speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    #         speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"
    #         synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    #         synthesizer.speak_text_async(text).get()
    #         time.sleep(0.5)
    #     except Exception as e:
    #         logging.error(f"TTS failed: {e}")

    # def record_audio(self, duration):
    #     logging.info(f"Recording for {duration} seconds...")
    #     recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
    #     sd.wait()
    #     write(AUDIO_FILENAME, 16000, recording)
    #     return AUDIO_FILENAME

    # def transcribe_audio(self, filename=AUDIO_FILENAME):
    #     try:
    #         speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    #         audio_input = speechsdk.AudioConfig(filename=filename)
    #         recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
    #         result = recognizer.recognize_once()
    #         if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    #             logging.info(f"Recognized Text: {result.text}")
    #             return result.text.strip()
    #     except Exception as e:
    #         logging.error(f"Transcription failed: {e}")
    #     return ""

    def normalize(self, text):
        return ''.join(text.lower().strip().split())

    def is_valid_spelling(self, text):
        clean = ''.join([c for c in text.upper() if c.isalpha()])
        return clean.isalpha() and len(clean) >= 3

    def extract_with_cohere(self, field_key, text, system_prompt_override=None):
        base_prompt = (
            f"You are a silent extraction assistant. Extract only the value for {field_key.upper()} from this input. "
            "Return only the extracted phrase."
        )

        prompt = system_prompt_override or base_prompt

        try:
            resp = co.chat(
                chat_history=[{"role": "system", "message": prompt}],
                message=text,
                temperature=0.3,
                max_tokens=30
            )
            extracted = resp.text.strip()
            # Handle edge cases when LLM gives explanation instead of value
            if any(phrase in extracted.lower() for phrase in ["no value", "does not contain", "implied"]):
                return ""
            return extracted
        except Exception as e:
            logging.error(f"Cohere extraction failed: {e}")
            return ""

    def generate_natural_acknowledgment(self, field_key, value, custom_prompt=None):
        if not value:
            return f"There is no value for {field_key.upper()} provided."

        if custom_prompt:
            system_prompt = custom_prompt.replace("{value}", value)
        else:
            system_prompt = (
                f"You are a friendly English-speaking assistant. Please respond with one short sentence to warmly "
                f"acknowledge the user's {field_key}. For example, if the {field_key} is '{{value}}', say something like its a beautiful place."
                f"encouraging or friendly using it in context."
            )

        system_prompt += " Limit your response to 10 words."

        try:
            message = f"{field_key}: {value}"
            resp = co.chat(chat_history=[{"role": "system", "message": system_prompt}], message=message)
            return resp.text.strip()
        except Exception as e:
            logging.error(f"Acknowledgment generation failed: {e}")
            return ""
    
    def validate_svo_pattern(self, text, field):
        if not isinstance(text, str) or not text.strip():
            logging.warning("Empty or invalid input for SVO validation.")
            return False

        system_prompt = (
            "You are an English grammar teacher. Evaluate whether the following sentence follows the Subject-Verb-Object (SVO) pattern. "
            "Only reply 'Yes' or 'No'."
        )

        try:
            resp = co.chat(
                chat_history=[{"role": "system", "message": system_prompt}],
                message=text.strip(),  # ✅ user input goes here
                temperature=0.2,
                max_tokens=10
            )
            answer = resp.text.strip().lower()
            logging.info(f"SVO validation result: {answer}")
            return "yes" in answer
        except Exception as e:
            logging.error(f"SVO validation failed: {e}")
            return False

    def validate_clause_expansion(self, text, field):
        system_prompt = (
            "You are a grammar tutor helping students expand sentences meaningfully.\n"
            "Check if the student's response logically expands the sentence: 'I met a girl.'\n"
            "It must include a clause such as 'who', 'where', 'when', or 'because', and maintain the original meaning (that the person met a girl).\n"
            f"Student's response: \"{text}\"\n"
            "Reply only 'Yes' if the response maintains the base meaning and adds a valid clause. Reply 'No' otherwise."
        )
        try:
            resp = co.chat(
                chat_history=[{"role": "system", "message": system_prompt}],
                message=text,
                temperature=0.2,
                max_tokens=10
            )
            result = resp.text.strip().lower()
            logging.info(f"Clause expansion validation result: {result}")
            return "yes" in result
        except Exception as e:
            logging.error(f"Clause expansion validation failed: {e}")
            return False
    
    def validate_yesno_question(self, text):
        if not isinstance(text, str) or not text.strip():
            return False

        system_prompt = (
            "You are an English grammar teacher. Is the following sentence a correctly formed Yes/No question? "
            "Answer only 'Yes' or 'No'."
        )

        try:
            resp = co.chat(
                chat_history=[{"role": "system", "message": system_prompt}],
                message=text.strip(),
                temperature=0.2,
                max_tokens=10
            )
            result = resp.text.strip().lower()
            logging.info(f"Yes/No question check result: {result}")
            return "yes" in result
        except Exception as e:
            logging.error(f"Yes/No validation failed: {e}")
            return False
    
    def validate_family_answer(self, user_input, field):
        expected = field.get("hint", "")
        system_prompt = (
            f"You are a language tutor. Check if the following student answer correctly answers the prompt."
            f"\nPrompt: {field['prompt']}\nHint (Expected Answer): {expected}\nStudent Answer: {user_input}"
            "\nOnly reply with Yes or No."
        )
        try:
            resp = co.chat(chat_history=[{"role": "system", "message": system_prompt}], message=user_input)
            result = resp.text.strip().lower()
            logging.info(f"Family validation result: {result}")
            return "yes" in result
        except Exception as e:
            logging.error(f"Validation failed for {field['key']}: {e}")
            return False  

    def validate_adjective_description(self, text, field):
        expected = field.get("hint", "")
        system_prompt = (
            f"You are a helpful language teacher. Decide if the student's response correctly answers the prompt.\n"
            f"Prompt: {field.get('prompt', '')}\n"
            f"Hint/Expected: {expected}\n"
            f"Student Answer: {text}\n\n"
            "Reply only 'Yes' if the answer is contextually appropriate, otherwise 'No'."
        )
        try:
            resp = co.chat(
                chat_history=[{"role": "system", "message": system_prompt}],
                message=text,
                temperature=0.2,
                max_tokens=10
            )
            result = resp.text.strip().lower()
            logging.info(f"Adjective validation result: {result}")
            return "yes" in result
        except Exception as e:
            logging.error(f"Adjective description validation failed: {e}")
            return False
    
    def validate_negative_sentence(self, text):
        if not isinstance(text, str) or not text.strip():
            return False

        system_prompt = (
            "You are an English grammar teacher. Is the following sentence a correct negative sentence in English? "
            "Answer only 'Yes' or 'No'."
        )

        try:
            resp = co.chat(
                chat_history=[{"role": "system", "message": system_prompt}],
                message=text.strip(),
                temperature=0.2,
                max_tokens=10
            )
            result = resp.text.strip().lower()
            logging.info(f"Negative sentence check result: {result}")
            return "yes" in result
        except Exception as e:
            logging.error(f"Negative validation failed: {e}")
            return False

    def extract_number_or_fallback(self, field_key, text):
        digits = ''.join(filter(str.isdigit, text))
        if digits and 3 <= int(digits) <= 120:
            return digits
        # fallback to cohere if STT fails
        return self.extract_with_cohere(field_key, text)
    
    def capture_summary_with_retry(self):
        max_attempts = 2
        attempts = 0
        while attempts < max_attempts:
            duration = FIELD_RECORDING_DURATION.get("summary", DEFAULT_DURATION)
            self.record_audio(duration=duration)
            user_summary = self.transcribe_audio()
            assessment, similarity = self.assess_summary(user_summary)

            self.filled_fields["user_summary_spoken"] = user_summary
            self.filled_fields["expected_summary"] = self.expected_summary
            self.filled_fields["similarity_score"] = round(similarity, 2)

            if assessment == "PASS":
                self.filled_fields["assessment"] = "PASS"
                return
            elif assessment == "RETRY":
                retry_msg = "Almost there! Try repeating the sentence once more."
            else:
                retry_msg = "Hmm, that wasn’t quite right. Let’s try again!"

            attempts += 1
            if attempts < max_attempts:
                #self.speak(retry_msg)
                #print(f"\nCHATBOT: {retry_msg}")
                self.notify_ui("retry", retry_msg)
            else:
                self.filled_fields["assessment"] = "FAIL"

    def handle_field_interaction(self, field):
        key = field["key"]
        prompt = field["prompt"].format(**self.filled_fields)
        duration = field.get("duration", DEFAULT_DURATION)
        behavior = field.get("llm_behavior")
        max_attempts = field.get("max_attempts", 2)

        value = ""
        attempts = 0

        while not value and attempts < max_attempts:
            self.notify_ui("prompt", prompt, duration=duration)
            if field.get("hint"):
                self.notify_ui("hint", field["hint"])

            self.notify_ui("status", "recording")

            # Awaiting frontend text submission via /submit-response

            self.notify_ui("status", "validating")
            # text = self.transcribe_audio()
            text = ""  # Placeholder: frontend should provide text

            # Remove the audio file after transcription in GUI mode
            # if self.gui_mode and self.audio_path and os.path.exists(self.audio_path):
            #     os.remove(self.audio_path)

            if text:
                self.notify_ui("transcript", text)
                value = self.handle_llm_behavior(field, text)

                if field.get("validate_spell") and not self.is_valid_spelling(value):
                    self.notify_ui("retry", "Hmm, that didn't sound like a spelling. Let's try again.")
                    value = ""
                    attempts += 1
                    continue

                if not value and field.get("retry_on_empty"):
                    self.notify_ui("retry", "Hmm, I didn’t get that. Could you please try again?")
                    value = ""
                    attempts += 1
                    continue

            attempts += 1
        self.filled_fields[key] = value if value else "[Unrecognized or skipped]"

    def handle_llm_behavior(self, field, user_input):
        behavior = field.get("llm_behavior")

        def _set_result(field_key, is_correct):
            self.field_scores[field_key] = is_correct
            self.field_results[field_key] = "PASS" if is_correct else "FAIL"
                
        if behavior == "extract_value":
            extract_type = field.get("extract_type")
            if extract_type == "number":
                value = self.extract_number_or_fallback(field["key"], user_input)
            else:
                value = self.extract_with_cohere(field["key"], user_input)

            if value and "The input does not" in value:
                logging.warning(f"LLM fallback text detected for {field['key']}")
                value = "[Unrecognized]"

            if field.get("acknowledge"):
                ack_prompt = field.get("ack_prompt")
                ack = self.generate_natural_acknowledgment(field["key"], value, custom_prompt=ack_prompt)
                self.notify_ui("acknowledge", ack)
            return value

        elif behavior == "acknowledge":
            ack = self.generate_natural_acknowledgment(field["key"], user_input)
            self.notify_ui("feedback", ack)
            self.last_result["feedback"] = ack
            return ack

        elif behavior == "validate_svo":
            is_correct = self.validate_svo_pattern(user_input, field)
            _set_result(field["key"], is_correct)
            return user_input

        elif behavior == "validate_family_answers":
            is_correct = self.validate_family_answer(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input
        
        elif behavior == "yesno_question_check":
            is_correct = self.validate_yesno_question(user_input)
            _set_result(field["key"], is_correct)
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "negative_sentence_check":
            is_correct = self.validate_negative_sentence(user_input)
            _set_result(field["key"], is_correct)
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_clause_expansion":
            is_correct = self.validate_clause_expansion(user_input, field)
            _set_result(field["key"], is_correct)
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input
        
        elif behavior == "validate_pronunciation":
            expected_text = field.get("expected")
            if expected_text:
                result, score = self.validate_pronunciation(expected_text, user_input)
                self.field_scores[field["key"]] = result == "PASS"
                self.filled_fields[f"{field['key']}_pron_score"] = round(score, 2)
                self.filled_fields[f"{field['key']}_pron_result"] = result
                feedback = field.get("feedback_pass") if result == "PASS" else field.get("feedback_fail")
                if feedback:
                    self.notify_ui("feedback", feedback)
                    self.last_result["feedback"] = feedback
            return user_input
        
        elif behavior == "validate_stress_pattern":
            result = self.validate_stress_pattern(user_input, field.get("expected"))
            self.field_scores[field["key"]] = result
            feedback = field.get("feedback_pass") if result else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_rising_intonation":
            result = self.validate_rising_intonation(user_input)
            self.field_scores[field["key"]] = result
            feedback = field.get("feedback_pass") if result else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_adjective_description":
            is_correct = self.validate_adjective_description(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_modal_verb_usage":
            is_correct = self.validate_modal_verb_usage(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_first_conditional":
            is_correct = self.validate_first_conditional(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_phrasal_verb":
            is_correct = self.validate_phrasal_verb(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_falling_intonation":
            result = self.validate_falling_intonation(user_input)
            self.field_scores[field["key"]] = result
            feedback = field.get("feedback_pass") if result else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_command":
            is_correct = self.validate_command_sentence(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input
        
        elif behavior == "validate_place_description":
            is_correct = self.validate_place_description(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_contextual_dialogue":
            is_correct = self.validate_contextual_dialogue(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_synonym_antonym":
            is_correct = self.validate_synonym_antonym(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input
        
        elif behavior == "validate_meaningful_response_strict":
            is_correct = self.validate_meaningful_response_strict(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_mindmap_ideas":
            is_correct = self.validate_mindmap_ideas(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_grammar_correction":
            is_correct = self.validate_grammar_correction(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_passive_voice":
            is_correct = self.validate_passive_voice(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_reported_speech":
            is_correct = self.validate_reported_speech(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input

        elif behavior == "validate_discourse_marker":
            is_correct = self.validate_discourse_marker(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input
        
        elif behavior == "validate_debate_response":
            is_correct = self.validate_debate_response(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input
        
        elif behavior == "validate_comparative_expression":
            is_correct = self.validate_comparative_expression(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input
        
        elif behavior == "validate_past_tense_response":
            is_correct = self.validate_past_tense_response(user_input, field)
            self.field_scores[field["key"]] = is_correct
            self.field_results[field["key"]] = "PASS" if is_correct else "FAIL"
            feedback = field.get("feedback_pass") if is_correct else field.get("feedback_fail")
            if feedback:
                self.notify_ui("feedback", feedback)
                self.last_result["feedback"] = feedback
            return user_input
        
        return user_input
    
    def run(self):
        if "welcome" in self.config:
            welcome_message = self.config["welcome"]
            #print(f"\nCHATBOT: {welcome_message}")
            #self.speak(welcome_message)
            self.notify_ui("welcome", welcome_message)
            # Step 1: Ask and process all fields
            for field in self.fields:
                key = field["key"]
                attempts = 0
                max_attempts = field.get("max_attempts", 2)
                while attempts < max_attempts:
                    self.handle_field_interaction(field)
                    behavior = field.get("llm_behavior")
                    if behavior and key in self.field_scores:
                        if self.field_scores[key]:
                            break  # success
                        attempts += 1
                        if attempts < max_attempts:
                            retry_msg = "Hmm, that wasn't quite right. Let's try again."
                            self.notify_ui("retry", retry_msg)
                    else:
                        break

        # Step 2: Validation logic
        feedback = ""
        if self.config.get("validation_logic"):
            logic = self.config["validation_logic"]
            logic_type = logic.get("type", "pattern_match")
            correct = 0
            total = 0

            if logic_type == "pattern_match":
                patterns = logic.get("pattern", [])
                if isinstance(patterns, str):
                    patterns = [patterns]

                for field in self.fields:
                    if field.get("validate_pattern") == logic.get("pattern"):
                        total += 1
                        if self.validate_svo_pattern(self.filled_fields[field["key"]], field):
                            correct += 1

            elif logic_type == "family_validation":
                for key, is_correct in self.field_scores.items():
                    total += 1
                    if is_correct:
                        correct += 1

            elif logic_type == "custom_svo_block":
                for key in logic.get("target_fields", []):
                    total += 1
                    if self.validate_svo_pattern(self.filled_fields.get(key, ""), {"key": key}):
                        correct += 1

            elif logic_type == "custom_clause":
                total = correct = 0
                for field in self.fields:
                    if field.get("llm_behavior") == "validate_clause_expansion":
                        total += 1
                        if self.validate_clause_expansion(self.filled_fields.get(field["key"], ""), field):
                            correct += 1

            elif logic_type == "yesno_or_negative":
                total = len(self.field_scores)
                correct = sum(1 for v in self.field_scores.values() if v)

            elif logic_type == "yesno_pattern_matching":
                patterns = logic.get("pattern", [])
                if isinstance(patterns, str):
                    patterns = [patterns]

                for field in self.fields:
                    if field.get("validate_pattern") in patterns:
                        total += 1
                        key = field["key"]
                        text = self.filled_fields.get(key, "")
                        pattern_type = field.get("validate_pattern")

                        if pattern_type == "yesno":
                            if self.validate_yesno_question(text):
                                correct += 1
                        elif pattern_type == "negative":
                            if self.validate_negative_sentence(text):
                                correct += 1

            elif logic_type == "pronunciation_check":
                for field in self.fields:
                    if field.get("llm_behavior") == "validate_pronunciation":
                        total += 1
                        score_key = f"{field['key']}_pron_score"
                        result_key = f"{field['key']}_pron_result"
                        result = self.filled_fields.get(result_key, "FAIL")
                        if result == "PASS":
                            correct += 1

            elif logic_type == "synonym_antonym_check":
                for field in self.fields:
                    if field.get("llm_behavior") == "validate_synonym_antonym":
                        total += 1
                        key = field["key"]
                        if self.field_scores.get(key, False):
                            correct += 1

            elif logic_type == "meaningful_response":
                for field in self.fields:
                    if field.get("llm_behavior") == "validate_meaningful_response":
                        total += 1
                        if self.field_scores.get(field["key"], False):
                            correct += 1

            elif logic_type == "stress_and_intonation":
                total = len(self.field_scores)
                correct = sum(1 for v in self.field_scores.values() if v)

            elif logic_type == "verb_constructs_validation":
                for field in self.fields:
                    key = field["key"]
                    behavior = field.get("llm_behavior", "")
                    if behavior in ["validate_modal_verb_usage", "validate_first_conditional", "validate_phrasal_verb"]:
                        total += 1
                        if self.field_scores.get(key, False):
                            correct += 1
            elif logic_type == "custom_command_block":
                for key in logic.get("target_fields", []):
                    total += 1
                    if self.field_scores.get(key):
                        correct += 1

            elif logic_type == "command_validation":
                for field in self.fields:
                    if field.get("llm_behavior") == "validate_command":
                        total += 1
                        key = field["key"]
                        if self.field_scores.get(key, False):
                            correct += 1

            elif logic_type == "place_description_check":
                for field in self.fields:
                    if field.get("llm_behavior") == "validate_place_description":
                        total += 1
                        if self.field_scores.get(field["key"]):
                            correct += 1

            elif logic_type == "custom_meaningful":
                for key in logic.get("target_fields", []):
                    total += 1
                    if self.field_scores.get(key):
                        correct += 1

            elif logic_type == "price_check":
                for field in self.fields:
                    if field.get("llm_behavior") == "validate_price_question":
                        total += 1
                        key = field["key"]
                        if self.field_scores.get(key, False):
                            correct += 1

            elif logic_type == "contextual_dialogue_check":
                for field in self.fields:
                    if field.get("llm_behavior") == "validate_contextual_dialogue":
                        total += 1
                        if self.field_scores.get(field["key"], False):
                            correct += 1

            elif logic_type == "mindmap_check":
                for field in self.fields:
                    if field.get("llm_behavior") == "validate_mindmap_ideas":
                        total += 1
                        if self.field_scores.get(field["key"], False):
                            correct += 1

            elif logic_type == "grammar_constructs_check":
                for field in self.fields:
                    if field["llm_behavior"] in ["validate_passive_voice", "validate_reported_speech", "validate_discourse_marker"]:
                        total += 1
                        if self.field_scores.get(field["key"], False):
                            correct += 1

            elif logic_type == "custom_literature_analysis":
                for key in logic.get("target_fields", []):
                    total += 1
                    if self.field_scores.get(key):
                        correct += 1

            elif logic_type == "custom_debate_analysis":
                for field in self.fields:
                    if field["key"] in logic.get("target_fields", []):
                        total += 1
                        if self.field_scores.get(field["key"], False):
                            correct += 1
                            
            elif logic_type == "group_discussion_turns":
                for key in logic.get("target_fields", []):
                    total += 1
                    if self.field_scores.get(key):
                        correct += 1

            if total > 0:
                score = correct / total
                passed = score >= (logic.get("minimum_correct", total) / total)
                feedback = logic.get("feedback_pass", "Great job!") if passed else logic.get("feedback_fail", "Please try again.")
                #print(f"\nCHATBOT FEEDBACK: {feedback}")
                #self.speak(feedback)
                self.notify_ui("feedback", feedback)
                self.filled_fields.update({
                    "score": round(score * 100, 2),
                    "feedback": feedback
                })
            else:
                fallback_feedback = "No applicable validation fields."
                self.notify_ui("feedback", fallback_feedback) 
                self.filled_fields.update({
                    "score": "N/A",
                    "feedback": fallback_feedback
                })
            if total == 0:
                score = 0
    
    # Step 3: Summary Section
# Step 3: Summary Section
        if self.config.get("summary_template") and all(
            v and "[Unrecognized" not in str(v) for v in self.filled_fields.values()
        ):
            self.expected_summary = self.config["summary_template"].format(**self.filled_fields)

            msg_intro = "Let's try a full sentence!"
            msg_example = f"(example): {self.expected_summary}"

            self.notify_ui("prompt", msg_intro)
            self.notify_ui("example", self.expected_summary)

            self.capture_summary_with_retry()
        else:
            self.notify_ui("status", "Debug mode on.")
            self.filled_fields.update({
                "expected_summary": "[Skipped]",
                "user_summary_spoken": "[Skipped]",
                "similarity_score": 0,
                "assessment": "SKIPPED"
            })

        # ➤ New: post-summary feedback
        if "post_summary_feedback" in self.config:
            outcome = self.filled_fields.get("assessment", "SKIPPED")
            msg = self.config["post_summary_feedback"].get(outcome)
            if msg:
                #print(f"\nCHATBOT: {msg}")
                #self.speak(msg)
                self.notify_ui("feedback", msg)
                
        if "closing" in self.config:
            self.notify_ui("closing", self.config["closing"])

        filename = f"{self.config['module']}_output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(filename, 'w') as f:
            json.dump({**self.filled_fields, "field_results": self.field_results}, f, indent=2)
        self.notify_ui("status", f"Results saved to {filename}")
        self.emit_static_closing()
        
    def assess_summary(self, user_spoken):
        norm_user = self.normalize(user_spoken)
        norm_expected = self.normalize(self.expected_summary)
        score = difflib.SequenceMatcher(None, norm_user, norm_expected).ratio()

        if score >= 0.90:
            return "PASS", score
        elif score >= 0.80:
            return "RETRY", score
        else:
            return "FAIL", score

    def validate_pronunciation(self, expected: str, spoken: str) -> tuple[str, float]:
        norm_expected = self.normalize(expected)
        norm_spoken = self.normalize(spoken)
        similarity = difflib.SequenceMatcher(None, norm_expected, norm_spoken).ratio()
        result = "PASS" if similarity >= 0.85 else "RETRY" if similarity >= 0.7 else "FAIL"
        return result, similarity
    
    def validate_stress_pattern(self, text, expected_word):
        system_prompt = (
            f"Does the pronunciation of the word '{expected_word}' in the user's response '{text}' show correct syllable stress? "
            "Reply only with 'Yes' or 'No'."
        )
        try:
            resp = co.chat(
                chat_history=[{"role": "system", "message": system_prompt}],
                message=text,
                temperature=0.2,
                max_tokens=10
            )
            answer = resp.text.strip().lower()
            logging.info(f"Stress validation result: {answer}")
            return "yes" in answer
        except Exception as e:
            logging.error(f"Stress validation failed: {e}")
            return False
        
    def validate_rising_intonation(self, text):
        system_prompt = (
            "Does the following sentence demonstrate rising intonation (as in a Yes/No question)? "
            "Only answer 'Yes' or 'No'."
        )
        try:
            resp = co.chat(
                chat_history=[{"role": "system", "message": system_prompt}],
                message=text,
                temperature=0.2,
                max_tokens=10
            )
            answer = resp.text.strip().lower()
            logging.info(f"Rising intonation result: {answer}")
            return "yes" in answer
        except Exception as e:
            logging.error(f"Rising intonation validation failed: {e}")
            return False
    def validate_meaningful_response(self, text, field):
        prompt = (
            f"Does the following response appropriately and meaningfully answer the question: '{field.get('prompt')}'?\n"
            f"Answer: '{text}'\n\nReply only 'Yes' or 'No'."
        )
        try:
            resp = co.chat(chat_history=[{"role": "system", "message": prompt}], message=text)
            result = resp.text.strip().lower()
            logging.info(f"Meaningful response check result: {result}")
            return "yes" in result
        except Exception as e:
            logging.error(f"Meaningful response validation failed: {e}")
            return False
    
    def validate_modal_verb_usage(self, text, field):
        prompt = (
            f"You are a grammar teacher. Does the following sentence use a modal verb correctly?\n"
            f"Sentence: '{text}'\nExpected: a correct use of 'must', 'should', or 'might'.\nReply only 'Yes' or 'No'."
        )
        return self._validate_yes_no_prompt(prompt, text, "Modal verb")

    def validate_first_conditional(self, text, field):
        prompt = (
            f"You're an English grammar teacher. Does the sentence '{text}' follow this 1st conditional structure:\n"
            "If + present simple, will + base verb?\nReply only 'Yes' or 'No'."
        )
        return self._validate_yes_no_prompt(prompt, text, "First Conditional")

    def validate_phrasal_verb(self, text, field):
        prompt = (
            f"You are an English tutor. Does this sentence use a common phrasal verb like 'run out of', 'run into', or 'give up'?\n"
            f"Sentence: '{text}'\nReply only 'Yes' or 'No'."
        )
        return self._validate_yes_no_prompt(prompt, text, "Phrasal Verb")

    def _validate_yes_no_prompt(self, system_prompt, text, label=""):
        try:
            resp = co.chat(chat_history=[{"role": "system", "message": system_prompt}], message=text)
            result = resp.text.strip().lower()
            logging.info(f"{label} validation result: {result}")
            return "yes" in result
        except Exception as e:
            logging.error(f"{label} validation failed: {e}")
            return False

    def validate_falling_intonation(self, text):
        system_prompt = (
            "Does the following sentence demonstrate falling intonation (as in a wh-question or statement)? "
            "Only answer 'Yes' or 'No'."
        )
        try:
            resp = co.chat(
                chat_history=[{"role": "system", "message": system_prompt}],
                message=text,
                temperature=0.2,
                max_tokens=10
            )
            answer = resp.text.strip().lower()
            logging.info(f"Falling intonation result: {answer}")
            return "yes" in answer
        except Exception as e:
            logging.error(f"Falling intonation validation failed: {e}")
            return False
        
    def validate_command_sentence(self, text, field):
        system_prompt = (   
            "You're an English teacher. Does the following sentence provide a clear, grammatically correct instruction using an imperative form? "
            f"Sentence: '{text}'\nReply only 'Yes' or 'No'."
        )
        return self._validate_yes_no_prompt(system_prompt, text, "Command Validation")
    
    def validate_place_description(self, text, field):
        prompt = (
            f"You are a language tutor. Does this response use spatial prepositions like 'near', 'next to', or 'behind' "
            f"to describe a place?\nResponse: \"{text}\"\nReply only 'Yes' or 'No'."
        )
        return self._validate_yes_no_prompt(prompt, text, "Place Description")
    
    def validate_contextual_dialogue(self, text, field):
        context = field.get("context", "")
        system_prompt = (
            f"You are simulating a roleplay practice. Based on this scene: '{context}', "
            f"does the user response fit this context?\n"
            f"Response: '{text}'\n"
            "Reply only 'Yes' or 'No'."
        )
        return self._validate_yes_no_prompt(system_prompt, text, label="Contextual Dialogue")
    def validate_synonym_antonym(self, text, field):
        word_type = field.get("word_type", "synonym")  # synonym or antonym
        expected_word = field.get("word", "")

        system_prompt = (
            f"You're a vocabulary expert. Is '{text}' a valid {word_type} for the word '{expected_word}'?\n"
            f"Only respond 'Yes' or 'No'."
        )
        return self._validate_yes_no_prompt(system_prompt, text, f"{word_type.title()} Validation")
    def validate_meaningful_response_strict(self, text, field):
        prompt = (
            f"You are a language tutor. Check if the student's answer meaningfully and directly answers this prompt.\n"
            f"Prompt: {field.get('prompt')}\n"
            f"Expected/Hints: {field.get('hint', '')}\n"
            f"Student's Answer: {text}\n"
            "Reply only with 'Yes' if it is clearly on-topic. Reply 'No' otherwise."
        )
        return self._validate_yes_no_prompt(prompt, text, "Meaningful Response Strict")  
    def validate_mindmap_ideas(self, text, field):
        system_prompt = (
            "You are helping a student build a mind map.\n"
            "Check if the following response contains at least three relevant ideas about the topic 'Planning a Birthday Party'.\n"
            "Acceptable concepts include: venue, guest list, cake, games, decorations, food, music.\n"
            f"Student's response: \"{text}\"\n"
            "Reply only 'Yes' if the response includes 3 or more related ideas. Otherwise reply 'No'."
        )
        return self._validate_yes_no_prompt(system_prompt, text, label="Mindmap Ideas")  
    def validate_grammar_correction(self, text, field):
        incorrect = field.get("incorrect", "")
        prompt = (
            f"You are a grammar teacher. Does this response correct the error in:\n"
            f"Incorrect Sentence: \"{incorrect}\"\n"
            f"Student Answer: \"{text}\"\n\n"
            "Reply only 'Yes' or 'No'."
        )
        return self._validate_yes_no_prompt(prompt, text, label="Grammar Correction")

    def validate_passive_voice(self, text, field):
        prompt = (
            f"Does the sentence '{text}' correctly use passive voice based on the active sentence: '{field.get('prompt')}'?\n"
            "Reply only 'Yes' or 'No'."
        )
        return self._validate_yes_no_prompt(prompt, text, "Passive Voice")
    def validate_reported_speech(self, text, field):
        prompt = (
            f"Does the response '{text}' convert the direct speech correctly into reported speech from: '{field.get('prompt')}'?\n"
            "Reply only 'Yes' or 'No'."
        )
        return self._validate_yes_no_prompt(prompt, text, "Reported Speech")
    def validate_discourse_marker(self, text, field):
        prompt = (
            f"Does this sentence use discourse markers like 'however', 'therefore', or 'meanwhile' to join two ideas?\n"
            f"Sentence: '{text}'\nReply only 'Yes' or 'No'."
        )
        return self._validate_yes_no_prompt(prompt, text, "Discourse Marker")
    
    def validate_debate_response(self, user_input: str, field: dict) -> bool:
        prompt = (
            f"Check if this response presents a coherent argument or opinion with justification: \"{user_input}\".\n"
            "Reply only with yes or no."
        )
        try:
            result = self.call_llm(prompt)
            is_valid = "yes" in result.lower()
            logging.info(f"Debate validation result: {'yes' if is_valid else 'no'}")
            return is_valid
        except Exception as e:
            logging.error(f"Debate validation failed: {e}")
            return False
        
    def validate_comparative_expression(self, text, field):
        prompt = (
            "You're a grammar teacher. Does the response use a correct comparative or superlative form (e.g., 'better', 'healthier', 'the best')?\n"
            f"Response: \"{text}\"\nReply only 'Yes' or 'No'."
        )
        return self._validate_yes_no_prompt(prompt, text, "Comparative/Superlative")
    def validate_past_tense_response(self, text, field):
        prompt = (
            "You're an English grammar teacher. Check if the following sentence is written in correct past tense "
            "and describes a past event or experience.\n"
            f"Sentence: \"{text}\"\n\n"
            "Only reply 'Yes' if it's grammatically in past tense and contextually about the past. Otherwise reply 'No'."
        )
        return self._validate_yes_no_prompt(prompt, text, "Past Tense Validation")
    def run_validation_logic(self):
        logic = self.config.get("validation_logic", {})
        minimum_correct = logic.get("minimum_correct", 1)
        feedback_pass = logic.get("feedback_pass", "Well done!")
        feedback_fail = logic.get("feedback_fail", "Let's try again.")

        passed_fields = [k for k, v in self.field_scores.items() if v]
        passed_count = len(passed_fields)

        if passed_count >= minimum_correct:
            self.notify_ui("feedback", feedback_pass)
            self.filled_fields["feedback"] = feedback_pass
        else:
            self.notify_ui("feedback", feedback_fail)
            self.filled_fields["feedback"] = feedback_fail
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to module config JSON")
    args = parser.parse_args()

    runner = ModuleRunner(args.config)
    runner.run()

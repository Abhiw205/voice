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

# === Constants ===
AUDIO_FILENAME = "input_audio.wav"
MAX_RETRIES = 2
REQUIRED_FIELDS = ["name", "location", "age"]

FIELD_RECORDING_DURATION = {
    "name": 4,
    "location": 4,
    "age": 3,
    "summary": 8
}

SYSTEM_PROMPT_EXTRACTION = (
    "You are a silent information extraction assistant embedded inside a voice conversation. "
    "Your only task is to extract exactly the required entity (NAME, LOCATION, or AGE) from the user's input without changing the conversation flow. "
    "Return only the clean extracted value (word or number), no greetings, no explanations, no extra words."
)

# === Load environment variables ===
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION", "eastus")

# === Initialize clients and logging ===
co = cohere.Client(api_key=COHERE_API_KEY)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class VoiceAssistant:
    def __init__(self):
        self.filled_fields = {}
        self.expected_summary = ""

    def speak(self, text):
        try:
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
            speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            synthesizer.speak_text_async(text).get()
        except Exception as e:
            logging.error(f"Text-to-speech failed: {e}")

    def record_audio(self, filename=AUDIO_FILENAME, duration=5, fs=16000):
        logging.info(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        write(filename, fs, recording)
        logging.info(f"Audio recorded and saved as {filename}")
        return filename

    def transcribe_audio(self, filename=AUDIO_FILENAME):
        try:
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
            audio_input = speechsdk.AudioConfig(filename=filename)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

            logging.info("Transcribing audio...")
            result = recognizer.recognize_once()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logging.info(f"Recognized Text: {result.text}")
                return result.text.strip()
            else:
                logging.warning("Could not recognize speech.")
                return ""
        except Exception as e:
            logging.error(f"Speech recognition failed: {e}")
            return ""

    def extract_with_cohere(self, field_type, user_input):
        system_instruction = SYSTEM_PROMPT_EXTRACTION.replace("(NAME, LOCATION, or AGE)", field_type.upper())

        try:
            response = co.chat(
                chat_history=[{"role": "system", "message": system_instruction}],
                message=user_input,
                temperature=0.2,
                max_tokens=30
            )
            extracted_value = response.text.strip()
            logging.info(f"Extracted {field_type}: {extracted_value}")
            return extracted_value
        except Exception as e:
            logging.error(f"Extraction failed for {field_type}: {e}")
            return user_input

    def validate_not_empty(self, text):
        return bool(text.strip())

    def ask_and_capture(self, prompt_text, field_key, duration_sec):
        self.speak(prompt_text)
        print(f"\nCHATBOT: {prompt_text}")
        self.record_audio(duration=duration_sec)
        user_input = self.transcribe_audio()
        if self.validate_not_empty(user_input):
            clean_input = self.extract_with_cohere(field_key, user_input)
            self.filled_fields[field_key] = clean_input
            return clean_input
        else:
            return None

    def run_conversation(self):
        name = None
        name_prompt = "Hello! What's your name?"
        while not name:
            name = self.ask_and_capture(name_prompt, "name", FIELD_RECORDING_DURATION["name"])

        location = None
        location_prompt = f"Nice to meet you, {self.filled_fields['name']}! Where are you from?"
        while not location:
            location = self.ask_and_capture(location_prompt, "location", FIELD_RECORDING_DURATION["location"])

        repeat_location = f"You said you are from {self.filled_fields['location']}."
        self.speak(repeat_location)
        print(f"\nCHATBOT: {repeat_location}")

        age = None
        age_prompt = "And how old are you?"
        while not age:
            age = self.ask_and_capture(age_prompt, "age", FIELD_RECORDING_DURATION["age"])

        repeat_age = f"You said you are {self.filled_fields['age']} years old."
        self.speak(repeat_age)
        print(f"\nCHATBOT: {repeat_age}")

        self.expected_summary = (
            f"My name is {self.filled_fields['name']}. "
            f"I am from {self.filled_fields['location']}. "
            f"I am {self.filled_fields['age']} years old."
        )

        self.speak("Great! Let's say it all together:")
        print("\nCHATBOT: Great! Let's say it all together:")
        self.speak(self.expected_summary)
        print(f"\nCHATBOT (example): {self.expected_summary}")

        self.capture_summary_with_retry()

    def capture_summary_with_retry(self):
        max_attempts = 2
        attempts = 0
        while attempts < max_attempts:
            self.record_audio(duration=FIELD_RECORDING_DURATION["summary"])
            user_summary = self.transcribe_audio()
            assessment, similarity = self.assess_summary(user_summary)

            self.filled_fields["user_summary_spoken"] = user_summary
            self.filled_fields["expected_summary"] = self.expected_summary
            self.filled_fields["similarity_score"] = round(similarity, 2)

            if assessment == "PASS":
                self.filled_fields["assessment"] = "PASS"
                return
            else:
                attempts += 1
                if attempts < max_attempts:
                    retry_msg = "Hmm, that wasn’t quite right. Let’s try again!"
                    self.speak(retry_msg)
                    print(f"\nCHATBOT: {retry_msg}")
                else:
                    self.filled_fields["assessment"] = "FAIL"

    def assess_summary(self, user_spoken):
        def normalize(text):
            return ''.join(text.lower().strip().split())

        normalized_user = normalize(user_spoken)
        normalized_expected = normalize(self.expected_summary)

        similarity = difflib.SequenceMatcher(None, normalized_user, normalized_expected).ratio()
        logging.info(f"Similarity Score: {similarity:.2f}")

        if similarity >= 0.90:
            return "PASS", similarity
        else:
            return "FAIL", similarity

    def save_results(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"module_output_{timestamp}.json"
        with open(output_filename, "w") as f:
            json.dump(self.filled_fields, f, indent=2)
        print(f"\n📁 Results saved to {output_filename}")

    def run(self):
        self.run_conversation()
        self.save_results()


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
# engine_config_runner.py

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

AUDIO_FILENAME = "input_audio.wav"
DEFAULT_DURATION = 5
SIMILARITY_THRESHOLD = 0.90

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION", "eastus")

co = cohere.Client(api_key=COHERE_API_KEY)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ConfigModuleRunner:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
        self.activities = self.config.get("activities", [])
        self.module_name = self.config.get("module_name", "module")

    def speak(self, text):
        try:
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
            speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            synthesizer.speak_text_async(text).get()
        except Exception as e:
            logging.error(f"TTS failed: {e}")

    def record_audio(self, duration):
        logging.info(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
        sd.wait()
        write(AUDIO_FILENAME, 16000, recording)
        return AUDIO_FILENAME

    def transcribe_audio(self, filename=AUDIO_FILENAME):
        try:
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
            audio_input = speechsdk.AudioConfig(filename=filename)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
            result = recognizer.recognize_once()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logging.info(f"Recognized Text: {result.text}")
                return result.text.strip()
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
        return ""

    def extract_with_cohere(self, field_key, text):
        prompt = f"You are a silent extraction assistant. Extract only the value for {field_key.upper()} from this input. Return only the extracted phrase."
        try:
            resp = co.chat(chat_history=[{"role": "system", "message": prompt}], message=text)
            return resp.text.strip()
        except Exception as e:
            logging.error(f"Cohere extraction failed: {e}")
            return text

    def normalize(self, text):
        return ''.join(text.lower().strip().split())

    def assess_summary(self, expected, spoken):
        norm_user = self.normalize(spoken)
        norm_expected = self.normalize(expected)
        score = difflib.SequenceMatcher(None, norm_user, norm_expected).ratio()
        return ("PASS" if score >= SIMILARITY_THRESHOLD else "FAIL", score)

    def run_activity(self, activity):
        filled_fields = {}
        for field in activity["fields"]:
            prompt = field["prompt"].format(**filled_fields)
            duration = field.get("duration", DEFAULT_DURATION)
            value = ""
            while not value:
                self.speak(prompt)
                print(f"\nCHATBOT: {prompt}")
                self.record_audio(duration)
                raw = self.transcribe_audio()
                if raw:
                    value = self.extract_with_cohere(field["key"], raw)
                    filled_fields[field["key"]] = value

        summary = activity.get("summary_template", "").format(**filled_fields)
        self.speak("Let's try a full sentence!")
        print("\nCHATBOT: Let's try a full sentence!")
        self.speak(summary)
        print(f"\nCHATBOT (example): {summary}")

        self.record_audio(DEFAULT_DURATION + 2)
        spoken = self.transcribe_audio()
        status, score = self.assess_summary(summary, spoken)

        return {
            **filled_fields,
            "expected_summary": summary,
            "user_summary_spoken": spoken,
            "similarity_score": round(score, 2),
            "assessment": status
        }

    def run(self):
        for activity in self.activities:
            print(f"\n=== {activity['title']} ===")
            result = self.run_activity(activity)
            filename = f"{self.module_name}_{activity['title'].replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"📁 Results saved to {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to module config JSON")
    args = parser.parse_args()

    runner = ConfigModuleRunner(args.config)
    runner.run()
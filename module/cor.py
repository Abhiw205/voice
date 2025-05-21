import cohere
import os
import json
import logging
import sounddevice as sd
from scipy.io.wavfile import write
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION", "eastus")

# === Init clients ===
co = cohere.Client(api_key=COHERE_API_KEY)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REQUIRED_FIELDS = {
    "name": "The user's name",
    "location": "Where the user is from",
    "age": "The user's age",
    "summary": "A sentence combining name, location, and age"
}

filled_fields = {}
retry_count = {}
MAX_RETRIES = 2
AUDIO_FILENAME = "input_audio.wav"

def get_prompt_for_field(field):
    examples = {
        "name": "Ask the user their name in a natural and friendly way.",
        "location": "Ask the user where they are from.",
        "age": "Ask the user how old they are.",
        "summary": "Ask the user to say a full sentence like: 'My name is Sarah. I am from Delhi. I am 22 years old.'"
    }
    return f"You are a helpful English-speaking assistant. {examples[field]}"

def validate_input(field, text):
    text = text.lower().strip()
    if field == "name":
        return "name" in text or text.replace(" ", "").isalpha()
    if field == "location":
        return "from" in text or len(text.split()) <= 3
    if field == "age":
        digits = ''.join(filter(str.isdigit, text))
        return digits.isdigit() and 0 < int(digits) < 120
    if field == "summary":
        return all(word in text for word in ["name", "from", "year"])
    return False

def ask_field_question(field):
    response = co.generate(
        prompt=get_prompt_for_field(field),
        max_tokens=60,
        temperature=0.6
    )
    return response.generations[0].text.strip()

def record_audio(filename="input_audio.wav", duration=5, fs=16000):
    logging.info("Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    logging.info(f"Audio recorded and saved as {filename}")
    return filename

def transcribe_audio(filename="input_audio.wav"):
    try:
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
        audio_input = speechsdk.AudioConfig(filename=filename)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

        logging.info("Transcribing audio...")
        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            logging.info(f"Recognized Text: {result.text}")
            return result.text
        else:
            logging.warning("Could not recognize speech.")
            return ""
    except Exception as e:
        logging.error(f"Speech recognition failed: {e}")
        return ""

def speak(text):
    try:
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
        speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        synthesizer.speak_text_async(text).get()
    except Exception as e:
        logging.error(f"Text-to-speech failed: {e}")

def main():
    for field in REQUIRED_FIELDS:
        retry_count[field] = 0

        while retry_count[field] < MAX_RETRIES:
            question = ask_field_question(field)
            print(f"\nCHATBOT: {question}")
            speak(question)

            record_audio(AUDIO_FILENAME)
            user_input = transcribe_audio(AUDIO_FILENAME)

            if validate_input(field, user_input):
                filled_fields[field] = user_input
                break
            else:
                retry_count[field] += 1
                if retry_count[field] < MAX_RETRIES:
                    retry_msg = "Hmm, that didn’t seem right. Can you try again?"
                    print(f"CHATBOT: {retry_msg}")
                    speak(retry_msg)
                else:
                    filled_fields[field] = "[User skipped or invalid input]"
                    skip_msg = "No worries, we’ll skip that for now."
                    print(f"CHATBOT: {skip_msg}")
                    speak(skip_msg)

    complete_msg = "Thanks! That completes our module."
    print(f"\nCHATBOT: {complete_msg}")
    speak(complete_msg)

    print("\n Module complete. Here’s what we gathered:")
    print(filled_fields)

    with open("module_output.json", "w") as f:
        json.dump(filled_fields, f, indent=2)
        print("📁 Results saved to module_output.json")

if __name__ == "__main__":
    main()
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from new import ModuleRunner
import os
from flask_cors import CORS
import tempfile
import json
import logging
import subprocess
import wave

# Setup Flask and Socket.IO
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Logging setup for debugging
logging.basicConfig(level=logging.INFO)

# Base config path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")

def ui_callback(payload):
    logging.info(f"UI Callback Payload: {json.dumps(payload, indent=2)}")
    socketio.emit("ui_event", payload)

def convert_to_wav(input_path, output_path):
    try:
        cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg conversion failed: {e}")
        raise

def log_wav_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            logging.info(f"⏱️ WAV duration: {duration:.2f}s")
    except Exception as e:
        logging.error(f"Failed to calculate duration: {e}")


# Split /run-module into /start-module and /submit-audio

@app.route("/start-module", methods=["POST"])
def start_module():
    global runner_instance
    data = request.get_json()
    config_name = data.get("config")

    if not config_name:
        return jsonify({"error": "No config provided"}), 400

    config_path = os.path.join(CONFIG_DIR, config_name)
    if not os.path.exists(config_path):
        return jsonify({"error": f"Config file not found: {config_path}"}), 400

    runner_instance = ModuleRunner(
        config_path=config_path,
        gui_mode=True,
        ui_callback=ui_callback
    )
    runner_instance.emit_welcome_and_prompt()

    logging.info("✅ /start-module endpoint successfully invoked.")
    return jsonify({"status": "started"})


@app.route("/list-modules", methods=["GET"])
def list_modules():
    categories = [
        os.path.join(CONFIG_DIR, "energizer"),
        os.path.join(CONFIG_DIR, "refresher"),
        os.path.join(CONFIG_DIR, "achiver"),
    ]
    module_list = []

    for category in categories:
        if os.path.exists(category):
            for f in os.listdir(category):
                if f.endswith(".json"):
                    module_list.append(os.path.join(os.path.basename(category), f))

    return jsonify({"modules": module_list})
# Global variable to hold the runner instance
runner_instance = None

@socketio.on("audio_ready")
def handle_audio_ready():
    global runner_instance
    if runner_instance:
        runner_instance.mark_audio_ready()
        logging.info("Audio is ready for processing.")


# --- Azure Speech TTS and STT endpoints ---
import azure.cognitiveservices.speech as speechsdk
from flask import send_file

@app.route("/speak", methods=["GET"])
def speak():
    text = request.args.get("text", "")
    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_KEY"),
            region=os.getenv("AZURE_REGION", "eastus")
        )
        speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"

        output_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return send_file(output_path, mimetype="audio/wav")
        else:
            return jsonify({"error": "TTS synthesis failed"}), 500
    except Exception as e:
        logging.error(f"TTS failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe", methods=["POST"])
@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No audio file uploaded"}), 400

    try:
        webm_path = os.path.join(tempfile.gettempdir(), "input.webm")
        wav_path = os.path.join(tempfile.gettempdir(), "input_converted.wav")
        audio.save(webm_path)

        # 🔁 Convert WebM to valid PCM WAV using ffmpeg
        cmd = ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 🔊 Transcribe
        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_KEY"),
            region=os.getenv("AZURE_REGION", "eastus")
        )
        audio_config = speechsdk.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return jsonify({"transcript": result.text})
        else:
            return jsonify({"error": "Speech not recognized"}), 400
    except Exception as e:
        logging.error(f"STT failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/submit-response", methods=["POST"])
def submit_response():
    global runner_instance
    data = request.get_json()
    logging.info(f"📥 Received response data: {data}")

    if not data or "text" not in data or "field_key" not in data:
        logging.error("❌ Missing 'text' or 'field_key' in response")
        return jsonify({"error": "Missing text or field_key"}), 400

    text = data["text"]
    field_key = data["field_key"]

    try:
        result = runner_instance.submit_response(text, field_key)
        score = runner_instance.field_scores.get(field_key)
        result_text = runner_instance.filled_fields.get(field_key)
        logging.info(f"✅ LLM Eval Score: {score} | Extracted Response: {result_text}")
        return jsonify({"result": result})
    except Exception as e:
        logging.exception("❌ Error during response handling")
        return jsonify({"error": str(e)}), 500

# --- New /get-summary endpoint ---
@app.route("/get-summary", methods=["GET"])
def get_summary():
    global runner_instance
    if not runner_instance:
        return jsonify({"error": "Runner not initialized"}), 400
    return jsonify({
        "filled_fields": runner_instance.filled_fields,
        "field_scores": runner_instance.field_scores
    })
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=3000, debug=True)

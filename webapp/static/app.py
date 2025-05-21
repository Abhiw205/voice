from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from new import ModuleRunner
app = Flask(__name__)

CONFIG_BASE = "configs"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    folders = [f for f in os.listdir(CONFIG_BASE) if os.path.isdir(os.path.join(CONFIG_BASE, f))]
    return render_template("index.html", folders=folders)

@app.route("/get-configs", methods=["POST"])
def get_configs():
    folder = request.json.get("folder")
    path = os.path.join(CONFIG_BASE, folder)
    configs = []

    if os.path.exists(path):
        configs = [f for f in os.listdir(path) if f.endswith(".json")]
    return jsonify(configs)

@app.route("/run", methods=["POST"])
def run():
    audio = request.files.get("audio")
    config_path = request.form.get("config")

    if not audio or not config_path:
        return jsonify({"error": "Missing audio or config"}), 400

    config_full = os.path.join(CONFIG_BASE, config_path)
    if not os.path.exists(config_full):
        return jsonify({"error": f"Config file not found: {config_full}"}), 400

    audio_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio.filename))
    audio.save(audio_path)

    runner = ModuleRunner(config=config_full, audio_path=audio_path)
    runner.run()

    result = dict(runner.filled_fields)
    # Extract the first field to show prompt/hint on frontend
    if runner.fields:
        first = runner.fields[0]
        result["prompt"] = first.get("prompt", "")
        result["hint"] = first.get("hint", "")
        result["feedback"] = runner.filled_fields.get("feedback", "")

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
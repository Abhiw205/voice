from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import os
from werkzeug.utils import secure_filename
import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from new import ModuleRunner

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

runner = None

def ui_callback(payload):
    """Emit UI feedback to frontend using Socket.IO."""
    socketio.emit("ui_event", payload)

@app.route("/")
def index():
    folders = [f for f in os.listdir(CONFIG_BASE) if os.path.isdir(os.path.join(CONFIG_BASE, f))]
    return render_template("index.html", folders=folders)

@app.route("/refresher")
def refresher():
    return render_template("refresher.html")

@app.route("/energizer")
def energizer():
    return render_template("energizer.html")

@app.route("/achiver")
def achiver():
    return render_template("achiver.html")

@app.route("/get-configs", methods=["POST"])
def get_configs():
    folder = request.json.get("folder")
    config_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    path = os.path.join(config_base, folder)
    configs = []
    if os.path.exists(path):
        configs = [{"id": idx, "name": f} for idx, f in enumerate(os.listdir(path)) if f.endswith(".json")]
    return jsonify(configs)

@app.route("/get-module", methods=["GET"])
def get_configs_query():
    folder = request.args.get("type")
    config_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    path = os.path.join(config_base, folder)
    configs = []
    if folder and os.path.exists(path):
        configs = [{"id": idx, "name": f} for idx, f in enumerate(os.listdir(path)) if f.endswith(".json")]
    return jsonify(configs)

# --- New endpoints for event-driven module running ---
@app.route("/start-module", methods=["POST"])
def start_module():
    global runner
    data = request.get_json()
    config_name = data.get("config")
    if not config_name:
        return jsonify({"error": "Missing config name"}), 400

    runner = ModuleRunner(config_name, gui_mode=True, ui_callback=ui_callback)
    runner.emit_welcome_and_prompt()
    return jsonify({"status": "started"})


# --- SocketIO Events ---
@socketio.on("connect")
def on_connect():
    print("Client connected")

@socketio.on("disconnect")
def on_disconnect():
    print("Client disconnected")

if __name__ == "__main__":
    socketio.run(app, debug=True)

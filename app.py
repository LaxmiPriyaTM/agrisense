"""
AgriSense - Flask REST API
===========================
Main entry point for the AgriSense application.
Serves the web dashboard and REST API endpoints.

API Endpoints:
  GET  /              → Web dashboard
  GET  /health        → Health check
  POST /predict       → Disease detection
  POST /chat          → LLM advisory chat
  GET  /history       → Session scan history
  POST /clear-history → Clear session history

Author: SEAI Individual Project
SDG: SDG 2 - Zero Hunger
"""

import os
import uuid
import json
import datetime
from flask import (Flask, request, jsonify, render_template,
                   session, send_from_directory)
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())
CORS(app)  # Enable CORS for cross-origin requests

# Configuration
UPLOAD_FOLDER = "./uploads"
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max upload
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ─────────────────────────────────────────────
# LAZY IMPORT (Load model only when first needed)
# ─────────────────────────────────────────────
_model_loaded = False

def get_predictor():
    """Lazy-load the ML model to avoid startup delay."""
    global _model_loaded
    from model.predict import predict, validate_image, ModelManager
    if not _model_loaded:
        try:
            ModelManager.get_instance()
            _model_loaded = True
        except Exception as e:
            app.logger.warning(f"Model not yet loaded: {e}")
    return predict, validate_image

def get_advisor():
    """Lazy-load the LLM advisor."""
    from llm.advisor import get_disease_advisory, chat_with_advisor
    return get_disease_advisory, chat_with_advisor


# ─────────────────────────────────────────────
# ① GET / — Web Dashboard
# ─────────────────────────────────────────────
@app.route("/")
def index():
    """Serves the main web dashboard."""
    return render_template("index.html")


# ─────────────────────────────────────────────
# ② GET /health — Health Check
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint for Docker/K8s readiness probes.
    
    Response:
    {
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": true,
        "timestamp": "2025-01-01T00:00:00"
    }
    """
    model_status = "loaded" if _model_loaded else "not_loaded"
    
    return jsonify({
        "status": "healthy",
        "service": "AgriSense API",
        "version": "1.0.0",
        "model_status": model_status,
        "groq_configured": bool(os.environ.get("GROQ_API_KEY")),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "sdg_alignment": "SDG 2 - Zero Hunger"
    }), 200


# ─────────────────────────────────────────────
# ③ POST /predict — Disease Detection
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_disease():
    """
    Main disease detection endpoint.
    
    Request:
        Content-Type: multipart/form-data
        Body:
            image (file)         ← Required: leaf image
            get_advisory (bool)  ← Optional: also get LLM advice (default: true)
            farmer_preference    ← Optional: "organic"/"chemical"/"balanced"
    
    Response:
    {
        "success": true,
        "request_id": "uuid",
        "prediction": {
            "primary": { "crop": "Tomato", "disease": "Late Blight", "confidence": 98.5 },
            "top_3": [...],
            "confidence_level": "HIGH"
        },
        "advisory": { ... },  // if get_advisory=true
        "timestamp": "..."
    }
    """
    request_id = str(uuid.uuid4())[:8]
    
    # ── Validate request
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file uploaded. Use field name 'image'.",
            "request_id": request_id
        }), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty filename. Please select a valid image.",
            "request_id": request_id
        }), 400
    
    # ── Read file bytes
    file_bytes = file.read()
    
    # ── Validate image
    predict_fn, validate_fn = get_predictor()
    is_valid, error_msg = validate_fn(file_bytes, file.filename)
    if not is_valid:
        return jsonify({
            "success": False,
            "error": error_msg,
            "request_id": request_id
        }), 400
    
    # ── Run ML inference
    try:
        result = predict_fn(file_bytes, top_k=3)
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({
            "success": False,
            "error": f"Model inference failed: {str(e)}",
            "request_id": request_id
        }), 500
    
    # ── Optionally get LLM advisory
    advisory_result = None
    get_advisory = request.form.get("get_advisory", "true").lower() == "true"
    farmer_preference = request.form.get("farmer_preference", "balanced")
    
    if get_advisory:
        try:
            primary = result["primary_prediction"]
            get_advisory_fn, _ = get_advisor()
            advisory_result = get_advisory_fn(
                crop=primary["crop"],
                disease=primary["disease"],
                class_key=primary["class_key"],
                confidence=primary["confidence"],
                farmer_preference=farmer_preference
            )
        except Exception as e:
            app.logger.warning(f"Advisory generation failed: {e}")
            advisory_result = {"error": str(e), "source": "failed"}
    
    # ── Build response
    primary = result["primary_prediction"]
    
    response_data = {
        "success": True,
        "request_id": request_id,
        "prediction": {
            "primary": {
                "crop": primary["crop"],
                "disease": primary["disease"],
                "class_key": primary["class_key"],
                "confidence": primary["confidence"],
                "confidence_formatted": primary["confidence_formatted"],
                "is_healthy": "healthy" in primary["disease"].lower()
            },
            "top_3": result["top_predictions"],
            "confidence_level": result["confidence_level"],
            "confidence_message": result["confidence_message"]
        },
        "advisory": advisory_result,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "model_version": result["model_version"]
    }
    
    # ── Save to session history
    if "scan_history" not in session:
        session["scan_history"] = []
    
    history_entry = {
        "id": request_id,
        "crop": primary["crop"],
        "disease": primary["disease"],
        "confidence": primary["confidence"],
        "timestamp": response_data["timestamp"],
        "is_healthy": "healthy" in primary["disease"].lower()
    }
    session["scan_history"].insert(0, history_entry)
    session["scan_history"] = session["scan_history"][:10]  # Keep last 10
    session.modified = True
    
    return jsonify(response_data), 200


# ─────────────────────────────────────────────
# ④ POST /chat — LLM Advisory Chat
# ─────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    """
    Multi-turn LLM advisory chatbot endpoint.
    
    Request:
    {
        "message": "What organic treatment should I use?",
        "history": [...]  // Previous conversation turns
    }
    
    Response:
    {
        "success": true,
        "reply": "For organic treatment...",
        "history": [...]
    }
    """
    data = request.get_json()
    
    if not data or "message" not in data:
        return jsonify({
            "success": False,
            "error": "Request body must contain 'message' field"
        }), 400
    
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({
            "success": False,
            "error": "Message cannot be empty"
        }), 400
    
    if len(user_message) > 1000:
        return jsonify({
            "success": False,
            "error": "Message too long (max 1000 characters)"
        }), 400
    
    conversation_history = data.get("history", [])
    
    try:
        _, chat_fn = get_advisor()
        result = chat_fn(conversation_history, user_message)
        
        return jsonify({
            "success": True,
            "reply": result["reply"],
            "history": result["history"],
            "tokens_used": result.get("tokens_used", 0)
        }), 200
        
    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "reply": "Sorry, I'm experiencing technical difficulties. Please try again."
        }), 500


# ─────────────────────────────────────────────
# ⑤ GET /history — Scan History
# ─────────────────────────────────────────────
@app.route("/history", methods=["GET"])
def get_history():
    """Returns the user's session-based scan history."""
    history = session.get("scan_history", [])
    return jsonify({
        "success": True,
        "history": history,
        "count": len(history)
    }), 200


# ─────────────────────────────────────────────
# ⑥ POST /clear-history — Clear Scan History
# ─────────────────────────────────────────────
@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Clears the session scan history."""
    session.pop("scan_history", None)
    return jsonify({"success": True, "message": "History cleared"}), 200


# ─────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size is 10MB."
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found. Available: /predict, /chat, /history, /health"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error. Please try again."
    }), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    
    print("\n" + "="*55)
    print("  🌾 AgriSense API Server Starting...")
    print(f"  📡 URL: http://localhost:{port}")
    print(f"  🔧 Debug: {debug}")
    print(f"  🤖 Groq API: {'✅ Configured' if os.environ.get('GROQ_API_KEY') else '❌ Not set (fallback mode)'}")
    print("="*55 + "\n")
    
    app.run(host="0.0.0.0", port=port, debug=debug)

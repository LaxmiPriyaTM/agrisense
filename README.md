<p align="center">
  <img src="https://img.shields.io/badge/SDG%202-Zero%20Hunger-green?style=for-the-badge&logo=un&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2-orange?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker"/>
  <img src="https://img.shields.io/badge/Groq-Llama3-purple?style=for-the-badge"/>
</p>

<h1 align="center">🌾 AgriSense — AI Crop Disease Detection & Advisory System</h1>

<p align="center">
  <strong>An intelligent deep learning system for real-time crop disease detection with LLM-powered treatment advisory</strong><br>
  Built with EfficientNetB3 + Flask REST API + Groq Llama 3 | SEAI Individual Project 2025
</p>

---

## 📌 Overview

**AgriSense** is an end-to-end AI system that helps farmers detect crop diseases from leaf images and receive instant, actionable treatment recommendations powered by a Large Language Model.

> 🌍 **UN SDG Alignment:** SDG 2 (Zero Hunger) — Helping farmers protect their crops, reduce losses, and achieve food security.

### Key Features
- 🔬 **Disease Detection** — EfficientNetB3 trained on 54,000+ PlantVillage images (38 classes, ~99% accuracy)
- 🤖 **LLM Advisory** — Groq Llama 3-powered chatbot for treatment recommendations
- 🧠 **Agentic AI** — Autonomous decision-making based on confidence + severity
- 🌐 **REST API** — Clean Flask API with `/predict`, `/chat`, `/health` endpoints
- 🐳 **Docker Ready** — Fully containerized for easy deployment
- 📊 **Scan History** — Session-based tracking of past diagnoses

---

## 🏗️ Architecture

```
User (Browser)
     │
     ▼
Frontend (HTML/CSS/JS)
     │  POST /predict    POST /chat
     ▼
Flask REST API (app.py)
     │              │
     ▼              ▼
EfficientNetB3   Groq LLM (Llama3-70B)
(PyTorch)        + Agentic Decision Layer
     │              │
     └──────┬────────┘
            ▼
    disease_info.json (Knowledge Base)
```

---

## 🚀 Quick Start

### Option 1: Run with Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/agrisense.git
cd agrisense

# 2. Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Build and run
chmod +x run.sh
./run.sh build
./run.sh run

# App is live at http://localhost:5000
```

### Option 2: Run Locally (Python)

```bash
# 1. Clone & enter directory
git clone https://github.com/LaxmiPriyaTM/agrisense.git
cd agrisense

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# 5. Run (without trained model — fallback mode)
python app.py
```

---

## 📂 Project Structure

```
agrisense/
├── app.py                    ← Flask REST API (main entry point)
├── requirements.txt          ← Python dependencies
├── Dockerfile                ← Container definition
├── run.sh                    ← Docker management script
├── .env.example              ← Environment variable template
├── .gitignore
│
├── model/
│   ├── train.py              ← EfficientNetB3 training script
│   ├── predict.py            ← Inference module (singleton pattern)
│   ├── class_names.json      ← 38 PlantVillage class labels
│   └── efficientnet_b3.pth   ← Trained model weights (download separately)
│
├── llm/
│   └── advisor.py            ← Groq LLM integration + agentic layer
│
├── data/
│   └── disease_info.json     ← Disease knowledge base (38 entries)
│
├── templates/
│   └── index.html            ← Web dashboard
│
├── static/
│   ├── css/style.css         ← Dark-theme styling
│   └── js/app.js             ← Frontend logic
│
├── uploads/                  ← Temporary image uploads (gitignored)

```

---

## 🔌 API Reference

### `GET /health`
System health check.
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "groq_configured": true,
  "version": "1.0.0"
}
```

### `POST /predict`
Detect disease in a leaf image.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|---|---|---|---|
| `image` | File | ✅ | Leaf image (JPG/PNG/WEBP, max 10MB) |
| `get_advisory` | bool | ❌ | Get LLM advisory (default: true) |
| `farmer_preference` | string | ❌ | `organic` / `balanced` / `chemical` |

**Response:**
```json
{
  "success": true,
  "prediction": {
    "primary": {
      "crop": "Tomato",
      "disease": "Late Blight",
      "confidence": 98.5,
      "is_healthy": false
    },
    "top_3": [...],
    "confidence_level": "HIGH"
  },
  "advisory": {
    "advisory": "## 🌱 Disease Overview\n...",
    "decisions": { "escalate_to_agronomist": true }
  }
}
```

### `POST /chat`
Multi-turn LLM advisory chat.

**Request:**
```json
{
  "message": "What organic treatment should I use for Late Blight?",
  "history": []
}
```

**Response:**
```json
{
  "success": true,
  "reply": "For organic treatment of Late Blight...",
  "history": [...]
}
```

### `GET /history`
Get session scan history.

### `POST /clear-history`
Clear session scan history.

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | EfficientNetB3 (Transfer Learning) |
| Pretrained On | ImageNet (1.2M images) |
| Fine-tuned On | PlantVillage (54,309 images) |
| Classes | 38 (healthy + diseased) |
| Input Size | 224 × 224 × 3 RGB |
| Framework | PyTorch 2.2 |
| Expected Accuracy | ~99% (val set) |
| Optimizer | AdamW + OneCycleLR |
| Training Epochs | 15 |

---

## 🌿 Supported Crops & Diseases

| Crop | Diseases Detected |
|---|---|
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Spot, Spider Mites, Target Spot, TYLCV, Mosaic Virus, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍎 Apple | Apple Scab, Black Rot, Cedar Rust, Healthy |
| 🍇 Grape | Black Rot, Esca, Leaf Blight, Healthy |
| 🌽 Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Bell Pepper | Bacterial Spot, Healthy |
| 🍊 Orange | Citrus Greening (HLB) |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| + more | Cherry, Squash, Soybean, Raspberry, Blueberry |

---

## 🐳 Docker Commands

```bash
./run.sh build      # Build image
./run.sh run        # Run in background
./run.sh dev        # Run with live logs
./run.sh stop       # Stop container
./run.sh logs       # Follow logs
./run.sh push       # Push to DockerHub
```

**DockerHub:**
```bash
docker pull your-username/agrisense:latest
docker run -p 5000:5000 --env GROQ_API_KEY=your_key your-username/agrisense:latest
```

---

## 🔑 Getting Your Free Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up (free — no credit card needed)
3. Click **API Keys** → **Create API Key**
4. Copy the key and add it to your `.env` file:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
   ```

> **Note:** Without a Groq API key, AgriSense still works in **fallback mode** — disease detection works normally, and advisory is served from the local knowledge base.

---

## 📚 Training Your Own Model

```bash
# 1. Download PlantVillage dataset
# From: https://www.kaggle.com/datasets/emmarex/plantdisease
# Extract to: agrisense/dataset/PlantVillage/

# 2. Run training script
cd agrisense
python model/train.py

# Training will save the best model to:
# model/efficientnet_b3.pth
```

---

## 📊 Performance Metrics

| Metric | Value |
|---|---|
| Test Accuracy | ~99% |
| Inference Time | < 200ms (CPU) |
| API Response Time | < 2s (with LLM) |
| Docker Image Size | ~3GB |
| Memory Usage | ~2GB (model loaded) |

---

## 📄 Research Paper

See `paper/agrisense_ieee_paper.pdf` — 5-page IEEE-format research paper documenting:
- System architecture & methodology
- Experimental results & comparisons
- SDG impact analysis

---

## 🤝 Based On

This project is built upon and significantly enhances:
- **Original Repo:** [manthan89-py/Plant-Disease-Detection](https://github.com/manthan89-py/Plant-Disease-Detection)
- **Dataset:** [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) (Hughes & Salathé, 2015)
- **Reference Paper:** Mohanty et al., "Using Deep Learning for Image-Based Plant Disease Detection," *Frontiers in Plant Science*, 2016

---

## 📝 License

MIT License — See [LICENSE](LICENSE)

---

<p align="center">Made with ❤️ for farmers worldwide • 🌍 SDG 2: Zero Hunger</p>

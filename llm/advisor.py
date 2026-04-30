"""
AgriSense - LLM Advisory Engine
=================================
Integrates Groq API (Llama3-70B) to provide intelligent
crop disease treatment recommendations.

Author: SEAI Individual Project
"""

import os
import json
import re
from groq import Groq

# ─────────────────────────────────────────────
# SYSTEM PROMPT (Prompt Engineering)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are AgriSense AI — an expert agricultural advisor with deep knowledge 
of crop pathology, sustainable farming, and integrated pest management (IPM).

Your role:
- Help farmers understand crop diseases detected by our AI system
- Provide clear, actionable treatment recommendations  
- Balance organic and chemical approaches
- Prioritize farmer safety and environmental sustainability
- Align with UN SDG 2 (Zero Hunger) principles

Response guidelines:
- Use simple, clear language (farmers may not be technical experts)
- Always structure your response with clear sections
- Provide BOTH organic (preferred) AND chemical options
- Rate severity and urgency clearly
- For CRITICAL diseases, strongly recommend consulting a local agronomist
- Be empathetic and encouraging

Format your responses as follows:
## 🌱 Disease Overview
[Brief 2-3 sentence explanation of the disease]

## ⚠️ Severity Assessment
[LOW / MEDIUM / HIGH / CRITICAL with explanation]

## 🚨 Immediate Actions (Next 48 Hours)
[Numbered list of urgent actions]

## 💊 Treatment Options

### 🌿 Organic / Natural Methods
[Numbered list]

### 🧪 Chemical Methods (Use responsibly)
[Numbered list with dosages]

## 🛡️ Prevention for Future
[Numbered list]

## 👨‍🌾 Expert Recommendation
[Closing advice]"""


# ─────────────────────────────────────────────
# DISEASE KNOWLEDGE BASE LOADER
# ─────────────────────────────────────────────
def load_disease_db() -> dict:
    """Loads local disease knowledge base for context injection."""
    db_path = os.path.join(os.path.dirname(__file__), "../data/disease_info.json")
    try:
        with open(db_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


DISEASE_DB = load_disease_db()


# ─────────────────────────────────────────────
# GROQ LLM CLIENT
# ─────────────────────────────────────────────
def get_groq_client() -> Groq:
    """Initialize Groq client with API key from environment."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. "
            "Get a free key at https://console.groq.com and set it in .env"
        )
    return Groq(api_key=api_key)


# ─────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────
def build_disease_context(class_key: str) -> str:
    """
    Builds rich context from local disease DB to inject into prompt.
    This grounds the LLM with accurate data and reduces hallucination.
    """
    info = DISEASE_DB.get(class_key, {})
    if not info:
        return ""

    context = f"""
Local Knowledge Base Information:
- Disease: {info.get('disease', 'Unknown')}
- Crop: {info.get('crop', 'Unknown')}
- Severity: {info.get('severity', 'Unknown')}
- Key Symptoms: {info.get('symptoms', 'N/A')}
- Organic Treatments: {', '.join(info.get('organic_treatment', [])[:3])}
- Chemical Treatments: {', '.join(info.get('chemical_treatment', [])[:3])}
- Prevention: {', '.join(info.get('prevention', [])[:3])}
- Urgency: Act within {info.get('urgency_days', 7)} days
"""
    return context


# ─────────────────────────────────────────────
# AGENTIC DECISION LAYER
# ─────────────────────────────────────────────
def agentic_decision(confidence: float, class_key: str, 
                      farmer_preference: str = "balanced") -> dict:
    """
    Agentic AI layer — makes decisions based on detection context.
    
    This simulates an autonomous agent that:
    1. Decides urgency level based on confidence + disease severity
    2. Filters treatment recommendations by farmer preference
    3. Determines if expert escalation is needed
    """
    info = DISEASE_DB.get(class_key, {})
    severity = info.get("severity", "MEDIUM")
    urgency_days = info.get("urgency_days", 7)

    decisions = {
        "generate_full_report": confidence >= 70,
        "request_rescan": confidence < 70,
        "escalate_to_agronomist": severity == "CRITICAL",
        "preferred_treatment": farmer_preference,  # "organic", "chemical", "balanced"
        "urgency_days": urgency_days,
        "alert_level": severity,
        "actions_taken": []
    }

    # Agentic action logging
    if confidence >= 90:
        decisions["actions_taken"].append("High confidence — full treatment report generated")
    elif confidence >= 70:
        decisions["actions_taken"].append("Moderate confidence — report generated with rescan suggestion")
    else:
        decisions["actions_taken"].append("Low confidence — rescan recommended before treatment")

    if severity == "CRITICAL":
        decisions["actions_taken"].append("⚠️ CRITICAL disease — agronomist consultation recommended")

    if farmer_preference == "organic":
        decisions["actions_taken"].append("Organic preference set — highlighting natural remedies")

    return decisions


# ─────────────────────────────────────────────
# MAIN ADVISORY FUNCTION
# ─────────────────────────────────────────────
def get_disease_advisory(
    crop: str,
    disease: str,
    class_key: str,
    confidence: float,
    farmer_preference: str = "balanced"
) -> dict:
    """
    Generates LLM-powered advisory for detected disease.

    Args:
        crop: Crop name (e.g., "Tomato")
        disease: Disease name (e.g., "Late Blight")
        class_key: Dataset class key for knowledge base lookup
        confidence: Detection confidence (0-100)
        farmer_preference: "organic", "chemical", or "balanced"

    Returns:
        dict with advisory text, decisions, and metadata
    """
    # Run agentic decision first
    decisions = agentic_decision(confidence, class_key, farmer_preference)

    # If confidence too low, don't waste LLM tokens
    if decisions["request_rescan"]:
        return {
            "advisory": (
                "## 📸 Image Quality Issue\n\n"
                "The AI confidence is too low for a reliable diagnosis. "
                "Please:\n"
                "1. Retake the photo in natural daylight\n"
                "2. Focus on a single leaf showing symptoms\n"
                "3. Ensure the image is clear and not blurry\n"
                "4. Fill the frame with the leaf"
            ),
            "decisions": decisions,
            "source": "fallback",
            "tokens_used": 0
        }

    # Build context-enriched prompt
    disease_context = build_disease_context(class_key)
    is_healthy = "healthy" in disease.lower()

    if is_healthy:
        user_message = f"""
My {crop} plant has been scanned and it's HEALTHY! (Confidence: {confidence:.1f}%)

Please provide:
1. What this means for my plant
2. Preventive care routine to keep it healthy
3. Warning signs to watch for
4. Best practices for {crop} cultivation

{disease_context}
"""
    else:
        pref_note = {
            "organic": "The farmer STRONGLY PREFERS organic/natural treatments only.",
            "chemical": "The farmer is open to chemical treatments for fastest results.",
            "balanced": "Please provide both organic and chemical options."
        }.get(farmer_preference, "")

        user_message = f"""
I scanned my {crop} leaf and the AgriSense AI detected:
- Disease: {disease}
- Confidence Level: {confidence:.1f}%
- Treatment Preference: {farmer_preference}

{pref_note}

{disease_context}

Please provide complete advisory for treating this disease.
"""

    # Call Groq API
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1500,
            top_p=0.9
        )

        advisory_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        return {
            "advisory": advisory_text,
            "decisions": decisions,
            "source": "groq_llama3",
            "model_used": "llama3-70b-8192",
            "tokens_used": tokens_used
        }

    except EnvironmentError as e:
        # Fallback when no API key
        return {
            "advisory": _fallback_advisory(crop, disease, class_key),
            "decisions": decisions,
            "source": "fallback_db",
            "tokens_used": 0,
            "warning": str(e)
        }
    except Exception as e:
        return {
            "advisory": _fallback_advisory(crop, disease, class_key),
            "decisions": decisions,
            "source": "fallback_db",
            "tokens_used": 0,
            "error": str(e)
        }


# ─────────────────────────────────────────────
# FALLBACK (When no API key)
# ─────────────────────────────────────────────
def _fallback_advisory(crop: str, disease: str, class_key: str) -> str:
    """Returns structured advisory from local DB when LLM is unavailable."""
    info = DISEASE_DB.get(class_key, {})
    if not info:
        return f"Disease detected: {disease} on {crop}. Please consult your local agronomist."

    organic = "\n".join([f"{i+1}. {t}" for i, t in enumerate(info.get("organic_treatment", []))])
    chemical = "\n".join([f"{i+1}. {t}" for i, t in enumerate(info.get("chemical_treatment", []))])
    prevention = "\n".join([f"{i+1}. {t}" for i, t in enumerate(info.get("prevention", []))])

    return f"""## 🌱 Disease Overview
{info.get('disease', disease)} detected on your {crop} plant.
{info.get('symptoms', 'Please inspect the affected leaves carefully.')}

## ⚠️ Severity Assessment
**{info.get('severity', 'MEDIUM')}** — Act within {info.get('urgency_days', 7)} days.

## 🌿 Organic Treatment
{organic}

## 🧪 Chemical Treatment
{chemical}

## 🛡️ Prevention
{prevention}

## 👨‍🌾 Expert Recommendation
{'⚠️ This is a CRITICAL disease. Please contact your local agricultural extension office immediately.' 
  if info.get('severity') == 'CRITICAL' 
  else 'Monitor your crop daily after treatment. If symptoms worsen, consult an agronomist.'}

*Note: For personalized AI advisory, add your GROQ_API_KEY to the .env file.*"""


# ─────────────────────────────────────────────
# CHAT CONVERSATION FUNCTION
# ─────────────────────────────────────────────
def chat_with_advisor(conversation_history: list, user_message: str) -> dict:
    """
    Handles multi-turn conversation with the agricultural advisor.
    
    Args:
        conversation_history: List of {"role": ..., "content": ...} dicts
        user_message: New user message
    
    Returns:
        dict with response and updated history
    """
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    try:
        client = get_groq_client()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )

        reply = response.choices[0].message.content
        conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        return {
            "reply": reply,
            "history": conversation_history,
            "tokens_used": response.usage.total_tokens
        }

    except Exception as e:
        fallback = ("I'm sorry, I'm having trouble connecting to the AI service. "
                    "Please check your GROQ_API_KEY. "
                    "For urgent help, contact your local agricultural extension service.")
        conversation_history.append({"role": "assistant", "content": fallback})
        return {
            "reply": fallback,
            "history": conversation_history,
            "error": str(e)
        }

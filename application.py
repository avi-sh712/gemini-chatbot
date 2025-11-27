import os
import json
import uuid
import boto3
import requests
import logging
from time import time
from flask import Flask, request, jsonify, render_template, session
from botocore.exceptions import ClientError

# --- Configuration Constants ---
# DynamoDB table name (MUST match the table you create in AWS Console)
CONVERSATION_TABLE_NAME = os.environ.get('DYNAMODB_TABLE_NAME', 'ChatbotConversationHistory')

# Gemini API Model and Endpoint (Key will be set via Elastic Beanstalk Environment Variable)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# --- AWS & Flask Setup ---
# 'application' is the default name Elastic Beanstalk looks for
application = Flask(__name__)
# Flask requires a secret key for session management (used to track the user)
application.secret_key = str(uuid.uuid4())
logging.basicConfig(level=logging.INFO)

# Initialize DynamoDB client (boto3 will automatically use the IAM Role credentials from EC2)
try:
    # Set a default region if not provided by the EB environment
    region = os.environ.get('AWS_REGION', 'us-east-1')
    dynamodb = boto3.resource('dynamodb', region_name=region)
    conversation_table = dynamodb.Table(CONVERSATION_TABLE_NAME)
    logging.info(f"DynamoDB Table '{CONVERSATION_TABLE_NAME}' initialized in region {region}.")
except ClientError as e:
    logging.error(f"Error initializing DynamoDB: {e}")
    conversation_table = None


# --- DynamoDB Functions for Session State (Memory) ---

def load_conversation_history(session_id):
    """Loads chat history for a given session ID from DynamoDB."""
    if not conversation_table:
        return []

    try:
        response = conversation_table.get_item(Key={'SessionID': session_id})

        # ConversationData is stored as a JSON string, so we must parse it
        if 'Item' in response and 'ConversationData' in response['Item']:
            return json.loads(response['Item']['ConversationData'])

        return []
    except ClientError as e:
        logging.error(f"Failed to load history for {session_id}: {e}")
        return []


def save_conversation_history(session_id, history):
    """Saves the updated chat history back to DynamoDB."""
    if not conversation_table:
        return

    try:
        # Serialize the history (list of objects) to a JSON string before saving
        conversation_table.put_item(
            Item={
                'SessionID': session_id,
                'Timestamp': int(time()),
                # History is stored as an array of objects: [{"role": "user", "text": "..."}]
                'ConversationData': json.dumps(history)
            }
        )
    except ClientError as e:
        logging.error(f"Failed to save history for {session_id}: {e}")


# --- AI Agent Logic ---

def get_ai_response(user_prompt, history):
    """Calls the Gemini API with conversation history for stateful conversation."""
    if not GEMINI_API_KEY:
        return "ERROR: AI Key not configured. Check Elastic Beanstalk environment variables."

    # Map the stored history format to the API format
    api_history = []
    for turn in history:
        # Note: 'model' in DynamoDB maps to 'model' role for the API
        api_history.append({"role": turn["role"], "parts": [{"text": turn["text"]}]})

    # Add the new user prompt
    api_history.append({"role": "user", "parts": [{"text": user_prompt}]})

    # System instruction (The Agent's Persona)
    system_prompt = "You are a concise, helpful, and friendly chatbot running on an AWS EC2 instance. You must remember the user's name and previous questions."

    payload = {
        "contents": api_history,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}]  # Enable grounding for real-time information
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload),
            timeout=20
        )
        response.raise_for_status()
        result = response.json()

        # Safely extract text from the response
        if (result.get('candidates') and
                result['candidates'][0].get('content') and
                result['candidates'][0]['content'].get('parts') and
                result['candidates'][0]['content']['parts'][0].get('text')):

            generated_text = result['candidates'][0]['content']['parts'][0]['text']
            return generated_text
        else:
            logging.error(f"AI Response in unexpected format: {result}")
            return "Sorry, I received an unusual response from the AI model."

    except requests.exceptions.RequestException as e:
        logging.error(f"API Request Error: {e}")
        return f"Sorry, the AI service failed to respond due to a network error on the EC2 instance."
    except Exception as e:
        logging.error(f"AI Response Parsing Error: {e}")
        return "Sorry, I received an unreadable response from the AI model."


# --- Flask Routes ---

@application.route('/')
def home():
    """Initializes the session and displays the chat UI."""
    # Initialize a new session ID if one doesn't exist (used as the DynamoDB key)
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    return render_template('index.html', session_id=session['session_id'])


@application.route('/chat', methods=['POST'])
def chat():
    """Handles new chat messages, invokes the AI, and manages state."""
    data = request.json
    user_prompt = data.get('prompt')
    session_id = session.get('session_id')

    if not user_prompt or not session_id:
        return jsonify({'error': 'Missing prompt or session ID'}), 400

    # 1. Load History (DynamoDB Read)
    history = load_conversation_history(session_id)

    # 2. Get AI Response (API Call)
    ai_response_text = get_ai_response(user_prompt, history)

    # 3. Update History with current turn
    history.append({"role": "user", "text": user_prompt})
    history.append({"role": "model", "text": ai_response_text})

    # 4. Save History (DynamoDB Write)
    # This ensures the bot remembers this turn for the next query.
    save_conversation_history(session_id, history)

    return jsonify({'response': ai_response_text})


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000, debug=True)
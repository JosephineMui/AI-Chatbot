from flask import Flask, request
from flask_cors import CORS
from chatbot import generate_response
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/chatbot", methods=["POST"])
def handle_prompt():
    # Read prompt from the request body
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data.get("prompt", "")

    # Generate a response using the chatbot
    response = generate_response(input_text)

    return response


if __name__ == "__main__":
    app.run(debug=True)
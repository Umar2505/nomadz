import os
import getpass

# 1️⃣ Set the Groq key before any LangChain import
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq:")

from flask import Flask, request, jsonify
import traceback
import main

# Initialize the Flask application
app = Flask(__name__)


def serialize_messages(obj):
    """Convert LangChain messages or any object to JSON-serializable form."""
    if isinstance(obj, list):
        return [m.content if hasattr(m, "content") else str(m) for m in obj]
    if hasattr(obj, "content"):
        return obj.content
    return str(obj)


@app.route("/api", methods=["POST"])
def query():
    try:
        # Try to parse JSON data
        data = request.get_json(force=True)
        print("data received: ", data)
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400

        # Example variable to hold data
        prompt_value = main.template_main_data.invoke({"query": data["query"]})
        print("prompt_value type:", type(prompt_value))  # should be StringPromptValue

        prompt = prompt_value.to_string()
        print("prompt type after to_string:", type(prompt))  # should be str

        violations = main.violations
        print("violations received: ", violations)

        response = main.main_agent_executor.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        print("response received: ", response)

        # Serialize messages before returning
        response_serialized = serialize_messages(response)
        violations_serialized = serialize_messages(violations)

        return (
            jsonify(
                {"output": response_serialized, "violations": violations_serialized}
            ),
            200,
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        print("ERROR TRACEBACK:\n", error_trace)
        return (
            jsonify({"status": "error", "message": str(e), "traceback": error_trace}),
            500,
        )


@app.route("/api/test-bad", methods=["POST"])
def test_api_good():
    return (
        jsonify(
            {
                "output": "this is a bad answer to your query",
                "violations": ["violation 1", "violation 2"],
            }
        ),
        200,
    )


@app.route("/api/test-good", methods=["POST"])
def test_api_bad():
    return (
        jsonify(
            {
                "output": "this is a good answer to your query",
                "violations": [],
            }
        ),
        200,
    )


# Run the server
if __name__ == "__main__":
    app.run(debug=True)

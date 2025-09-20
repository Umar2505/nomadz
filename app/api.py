from flask import Flask, request, jsonify

import main

# Initialize the Flask application
app = Flask(__name__)


@app.route("/api", methods=["POST"])
def query():
    try:
        # Try to parse JSON data
        data = request.get_json(force=True)

        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400

        # Example variable to hold data

        prompt = main.template_main_data.invoke({"query": data["query"]})
        violations = main.violations
        response = main.main_agent_executor.invoke(prompt)

        # Example success response
        return jsonify({"output": response, "violations": violations}), 200

    except Exception as e:
        # Catch unexpected errors
        return jsonify({"status": "error", "message": str(e)}), 500


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

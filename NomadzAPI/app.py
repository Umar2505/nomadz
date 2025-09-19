import csv
import io
from flask import Flask, request, jsonify


# Initialize the Flask application
app = Flask(__name__)

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    # This function block is indented with 4 spaces
    # You can uncomment the print statement below for debugging if needed
    # print(f"DEBUG: request.files content is: {request.files}")

    # Check if a file is included in the request
    if 'file' not in request.files:
        # This block is indented with 8 spaces
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Check if the file is a CSV
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File is not a CSV"}), 400

    try:
        # Read the file in memory
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream)
        
        # Get header row
        header = next(csv_input)
        
        # Process the CSV into a list of dictionaries
        data = []
        for row in csv_input:
            data.append(dict(zip(header, row)))

        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Run the server
if __name__ == '__main__':
    app.run(debug=True)

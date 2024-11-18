import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from flask import Flask, request, render_template, flash, jsonify
import os
import datetime
import tempfile

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flashing messages

# Initialize the Vertex AI Model
vertexai.init(project="speech-app-436121", location="us-central1")
model = GenerativeModel("gemini-1.5-flash-002")

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

def analyze_audio(audio_data, mime_type):
    try:
        # Create a temporary file to store the audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix='.audio') as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        # Read the audio file for analysis
        with open(temp_path, 'rb') as f:
            audio_part = Part.from_data(f.read(), mime_type=mime_type)

        # Define the analysis prompt
        prompt = """Please provide:
        1. A complete transcript of the audio.
        2. Sentiment analysis with details on tone, emotional patterns, and voice characteristics."""

        # Perform the analysis using the Vertex AI model
        responses = model.generate_content(
            [prompt, audio_part],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

        # Collect the full response
        full_response = ""
        for response in responses:
            full_response += response.text

        # Save results to a text file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"analysis_results_{timestamp}.txt"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(full_response)

        # Clean up the temporary audio file
        os.remove(temp_path)

        return full_response, result_file

    except Exception as e:
        # Clean up in case of errors
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"Error analyzing audio: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    filename = None
    if request.method == 'POST':
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            if audio_file.filename == '':
                flash('No file selected')
                return render_template('index.html')

            if not audio_file.filename.lower().endswith(('.mp3', '.webm')):
                flash('Please upload an MP3 or WebM file')
                return render_template('index.html')

            try:
                mime_type = 'audio/mpeg' if audio_file.filename.lower().endswith('.mp3') else 'audio/webm'
                result, filename = analyze_audio(audio_file.read(), mime_type)
                flash(f'Analysis complete! Results saved to {filename}')
            except Exception as e:
                flash(f'Error: {str(e)}')
        else:
            flash('No audio data received')

    return render_template('index.html', result=result, filename=filename)

@app.route('/analyze_audio', methods=['POST'])
def analyze_recorded_audio():
    if 'audio_data' not in request.json:
        return jsonify({'error': 'No audio data received'}), 400

    audio_data = base64.b64decode(request.json['audio_data'].split(',')[1])
    try:
        result, filename = analyze_audio(audio_data, 'audio/webm')
        return jsonify({'result': result, 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8080))
        app.run(host='0.0.0.0', port=port, debug=False)

import os
import requests
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from moviepy.editor import VideoFileClip
import tempfile
import speech_recognition as sr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load API key from environment variable
EDENAI_API_KEY = os.getenv("EDENAI_API_KEY")
EDENAI_TEXT_URL = "https://api.edenai.run/v2/text/generation"

if EDENAI_API_KEY is None:
    raise EnvironmentError("EDENAI_API_KEY environment variable not set")

# Function to transcribe audio using SpeechRecognition library
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(file_path)
    
    with audio_file as source:
        audio = recognizer.record(source)
    
    try:
        transcript = recognizer.recognize_google(audio)
        return transcript
    except sr.RequestError as e:
        app.logger.error(f"Could not request results from Google Speech Recognition service; {e}")
    except sr.UnknownValueError:
        app.logger.error("Google Speech Recognition could not understand audio")

    return None

# Function to generate summary and MCQs using Eden AI
def generate_summary_and_mcqs(input_text):
    text = f"Generate 10 MCQ questions in the following format for {input_text}:"
    mcq_prompt = '''{
      "Question": "",
      "Answers": {
        "A": "",
        "B": "",
        "C": "",
        "D": ""
      },
      "Correct_Answer": ""
    }'''

    payload1 = {
        'providers': ['openai'],
        "response_as_dict": True,
        "attributes_as_list": False,
        "show_original_response": False,
        "temperature": 0.7,
        "max_tokens": 1500,
        'text': f"{text} {mcq_prompt}"
    }
    headers = {
        'Authorization': f'Bearer {EDENAI_API_KEY}',
        'Content-Type': 'application/json',
        "accept": "application/json",
    }
    response1 = requests.post(EDENAI_TEXT_URL, json=payload1, headers=headers)
    if response1.status_code != 200:
        app.logger.error(f"MCQ generation API failed with status code: {response1.status_code} and response: {response1.text}")
        return None
    
    mcqs = response1.json()
    mcq_content = mcqs.get("openai", {}).get("generated_text", "No MCQs generated.")

    # Generate summary
    summary_payload = {
        'providers': ['openai'],
        "response_as_dict": True,
        "attributes_as_list": False,
        "show_original_response": False,
        "temperature": 0.7,
        "max_tokens": 1500,
        'text': f"Summarize the following text also generate some explanations: {input_text}"
    }
    response2 = requests.post(EDENAI_TEXT_URL, json=summary_payload, headers=headers)
    if response2.status_code != 200:
        app.logger.error(f"Summary generation API failed with status code: {response2.status_code} and response: {response2.text}")
        return None
    
    summary = response2.json()
    summary_text = summary.get("openai", {}).get("generated_text", "No summary generated.")

    return {
        "summary": summary_text,
        "mcqs": mcq_content
    }

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files or request.files['video'].filename == '':
        return {"message": "No video file provided"}, 400
    
    video = request.files['video']
    
    try:
        # Save the video to a temporary file in a controlled temporary directory
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_file:
            video.save(temp_video_file.name)
            temp_video_path = temp_video_file.name
        
        try:
            # Extract audio and save as .wav in another temporary file
            temp_audio_path = temp_video_path.replace('.mp4', '.wav')
            with VideoFileClip(temp_video_path) as video_clip:
                audio_clip = video_clip.audio
                audio_clip.write_audiofile(temp_audio_path)
            
            # Transcribe the audio using SpeechRecognition library
            transcript = transcribe_audio(temp_audio_path)
            if transcript is None:
                return {"message": "Transcription failed"}, 500

            # Generate summary and MCQs
            processed_content = generate_summary_and_mcqs(transcript)
            if processed_content is None:
                return {"message": "MCQ or summary generation failed"}, 500

            return jsonify({
                "summary": processed_content["summary"],
                "mcqs": processed_content["mcqs"]
            })
        
        finally:
            # Clean up temporary files
            os.remove(temp_video_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    except Exception as e:
        app.logger.error(f"Exception occurred: {str(e)}")
        return {"message": "Internal server error"}, 500
@app.route('/')
def hello():
    return "Hello, World!"
if __name__ == '__main__':
    app.run(debug=True)

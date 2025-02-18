from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
import torch
from deepfake_detection import extract_frames, preprocess_frames, predict_deepfake, ensemble_model

app = Flask(__name__)
CORS(app, resources={r"/uploads": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'uploads'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads', methods=['POST'])
def upload_video():
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        logger.info(f"File saved: {filepath}")
    
        frames = extract_frames(filepath)
        logger.info(f"{len(frames)} frames extracted")
        fake_prob = predict_deepfake(frames, ensemble_model)
        result = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = round(fake_prob * 100 if result == "FAKE" else (100 - fake_prob * 100), 2)

        os.remove(filepath)
        logger.info(f"Detection done. Result: {result}, Confidence: {confidence}%")

        return jsonify({'filename': file.filename, 'result': result, 'confidence': f"{confidence}%"})
    return jsonify({'error': 'Upload failed'}), 400


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    app.run(host='0.0.0.0', debug=True)

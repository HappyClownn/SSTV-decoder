import os
import logging
from flask import Flask, render_template, request, jsonify, send_file
import uuid
import time
from werkzeug.utils import secure_filename
from sstv_processor import process_sstv_file, create_test_wav

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-development-key")

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
RESULT_FOLDER = 'temp_results'
ALLOWED_EXTENSIONS = {'wav'}

# Create temporary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and SSTV decoding"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename for the uploaded file
        unique_id = str(uuid.uuid4())
        base_filename = secure_filename(file.filename)
        wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{base_filename}")
        file.save(wav_path)
        
        try:
            # Start the SSTV decoding process
            logger.info(f"Processing SSTV file: {wav_path}")
            result = process_sstv_file(wav_path, RESULT_FOLDER, unique_id)
            
            if result['success']:
                # Check for alternative decoding method results
                # These would be named unique_id_method1.png, unique_id_method2.png, etc.
                alt_methods = []
                for method_num in range(1, 4):  # We have 3 methods
                    method_filename = f"{unique_id}_method{method_num}.png"
                    method_path = os.path.join(RESULT_FOLDER, method_filename)
                    
                    if os.path.exists(method_path):
                        alt_methods.append({
                            'number': method_num,
                            'name': f"Method {method_num}",
                            'image_path': f"/get_image/{method_filename}",
                            'download_path': f"/download/{method_filename}"
                        })
                
                # Return the result details
                return jsonify({
                    'success': True,
                    'image_path': f"/get_image/{result['image_filename']}",
                    'download_path': f"/download/{result['image_filename']}",
                    'sstv_mode': result['mode'],
                    'decoding_method': result.get('method', 'Standard'),
                    'processing_time': f"{result['duration']} seconds",
                    'alternative_methods': alt_methods
                })
            else:
                return jsonify({'error': f"Failed to decode SSTV signal: {result.get('error', 'Unknown error')}"}), 500
        except Exception as e:
            logger.error(f"Error during SSTV decoding: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
        finally:
            # Clean up the uploaded WAV file
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")
    
    return jsonify({'error': 'File type not allowed. Please upload a WAV file.'}), 400

@app.route('/get_image/<filename>')
def get_image(filename):
    """Serve the decoded image"""
    return send_file(os.path.join(RESULT_FOLDER, filename))

@app.route('/download/<filename>')
def download_image(filename):
    """Download the decoded image"""
    return send_file(os.path.join(RESULT_FOLDER, filename), 
                     as_attachment=True,
                     download_name=f"sstv_decoded_{filename}")
                     
@app.route('/generate_test')
def generate_test_file():
    """Generate a test SSTV WAV file"""
    try:
        unique_id = str(uuid.uuid4())
        test_filename = f"test_sstv_{unique_id}.wav"
        test_path = os.path.join(UPLOAD_FOLDER, test_filename)
        
        # Create the test WAV file
        create_test_wav(test_path, duration=5.0)
        
        # Process the test file
        logger.info(f"Processing generated test SSTV file: {test_path}")
        result = process_sstv_file(test_path, RESULT_FOLDER, unique_id)
        
        if result['success']:
            # Check for alternative decoding method results
            alt_methods = []
            for method_num in range(1, 4):  # We have 3 methods
                method_filename = f"{unique_id}_method{method_num}.png"
                method_path = os.path.join(RESULT_FOLDER, method_filename)
                
                if os.path.exists(method_path):
                    alt_methods.append({
                        'number': method_num,
                        'name': f"Method {method_num}",
                        'image_path': f"/get_image/{method_filename}",
                        'download_path': f"/download/{method_filename}"
                    })
            
            # Return the result details
            return jsonify({
                'success': True,
                'image_path': f"/get_image/{result['image_filename']}",
                'download_path': f"/download/{result['image_filename']}",
                'sstv_mode': result['mode'],
                'decoding_method': result.get('method', 'Standard'),
                'processing_time': f"{result['duration']} seconds",
                'test_wav_path': f"/download_wav/{test_filename}",
                'alternative_methods': alt_methods
            })
        else:
            return jsonify({'error': f"Failed to decode test SSTV signal: {result.get('error', 'Unknown error')}"}), 500
    
    except Exception as e:
        logger.error(f"Error generating test file: {str(e)}")
        return jsonify({'error': f'Error generating test file: {str(e)}'}), 500

@app.route('/download_wav/<filename>')
def download_wav(filename):
    """Download the generated WAV file"""
    return send_file(os.path.join(UPLOAD_FOLDER, filename), 
                     as_attachment=True,
                     download_name=filename)

# Clean up old temporary files
@app.before_request
def cleanup_old_files():
    try:
        current_time = time.time()
        # Clean up files older than 1 hour
        for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) and current_time - os.path.getmtime(file_path) > 3600:
                    os.remove(file_path)
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

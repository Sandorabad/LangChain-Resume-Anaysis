import os
import uuid
import shutil
from flask import Flask, render_template, request, send_file, redirect, flash, make_response
from werkzeug.utils import secure_filename
from main_back import run_matches

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXCEL_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB upload limit
app.secret_key = 'your_secret_key'  # REPLACE with a strong secret key in production

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clear_folder(folder_path):
    """Clears all files and subdirectories in the given folder."""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # remove directory
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Instead of deleting the entire folder, clear its contents.
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        clear_folder(app.config['UPLOAD_FOLDER'])
        
        if 'files' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        files = request.files.getlist('files')
        job_description = request.form.get('job_description')
        if not job_description:
            flash('Job description is required.')
            return redirect(request.url)
        
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                saved_files.append(unique_filename)
            else:
                flash('Only PDF files are allowed.')
                return redirect(request.url)
        
        try:
            run_matches(app.config['UPLOAD_FOLDER'], job_description)
            
            # Assumes the Excel file is generated as "tournament_results.xlsx" in the current directory.
            output_file = 'tournament_results.xlsx'
            os.makedirs(app.config['EXCEL_FOLDER'], exist_ok=True)
            output_path = os.path.join(app.config['EXCEL_FOLDER'], output_file)
            
            shutil.move(output_file, output_path)
            
            # Optionally, clear the uploads folder after processing
            clear_folder(app.config['UPLOAD_FOLDER'])
            
            # Create a response that sets a cookie to indicate download completion.
            response = make_response(send_file(output_path, as_attachment=True))
            response.set_cookie('fileDownload', 'true', max_age=60)
            return response
        except Exception as e:
            flash('Error processing resumes: ' + str(e))
            return redirect(request.url)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

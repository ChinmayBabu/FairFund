from flask import Flask, render_template, request, redirect, url_for
import os
from ibm_granite_utils import analyze_document

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.pdf'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            results = analyze_document(filepath)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

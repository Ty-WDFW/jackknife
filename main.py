from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd


ALLOWED_EXTENSIONS = set(['csv'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my_secret_key'


@app.route('/')
def main():
    if session['data']:
        session['data'] = None
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        #if file is good read it and convert it into json
        #to be stored in session
        DataFrame = pd.read_csv(file)
        session['data'] = DataFrame.to_json()
        return redirect(url_for('view'))

@app.route('/view', methods=['GET', 'POST'])
def view():
    if session['data'] is None:
        return redirect(url_for('main'))

    if request.method == 'GET':
        data = pd.read_json(session['data'])
        return render_template('view.html', columns=data.columns, data=data)

    if request.method == 'POST':
        selected_year = request.form['year']
        selected_runsize = request.form['runsize']
        selected_predictor = request.form['predictor']

        session['year_selected_column'] = selected_year
        session['runsize_selected_column'] = selected_runsize
        session['predictor_selected_column'] = selected_predictor

        return str(selected_year)

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(debug=True)
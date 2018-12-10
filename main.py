from flask import Flask, render_template, request, redirect, url_for, session
from stats.jackknife import JackKnife
import pandas as pd


ALLOWED_EXTENSIONS = {'csv', 'xlsx'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'some_crazy_secret_key'


@app.route('/')
def main():

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
        #read csv, excel
        if file.filename.rsplit('.', 1)[1].lower() == 'csv':
            DataFrame = pd.read_csv(file)
        elif file.filename.rsplit('.', 1)[1].lower() == 'xlsx':
            DataFrame = pd.read_excel(file)

        session['data'] = DataFrame.to_json()
        return redirect(url_for('view'))

@app.route('/view', methods=['GET', 'POST'])
def view():
    if session['data'] is None:
        return redirect(url_for('main'))

    data = session['data']
    data_df = pd.read_json(data)
    if request.method == 'GET':
        return render_template('view.html', columns=data_df.columns, data=data_df)

    if request.method == 'POST':
        selected_year = request.form['year']
        selected_runsize = request.form['runsize']
        selected_predictor = request.form['predictor']

        # session['year_selected_column'] = selected_year
        # session['runsize_selected_column'] = selected_runsize
        # session['predictor_selected_column'] = selected_predictor

        results = JackKnife(data, predictor_column=selected_predictor, result_column=selected_runsize,
                            year_column=selected_year)

        return render_template('view.html', results=results, columns=data_df.columns, data=data_df,
                               selected_year=selected_year, selected_runsize=selected_runsize,
                               selected_predictor=selected_predictor)


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(debug=True)

import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, url_for
from flask import Flask, abort, jsonify
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    return jsonify(error=str(e)), code

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

	model = pickle.load(open('lgbm.pkl', 'rb'))
	encoder = MinMaxScaler()
	dt1 = request.form['a']
	dt2 = float(request.form['b'])
	dt3 = float(request.form['c'])
	dt4 = int(request.form['d'])
	dt5 = int(request.form['e'])
	dt6 = int(request.form['f'])
	dt7 = int(request.form['g'])
	dt8 = int(request.form['h'])
	dt9 = int(request.form['i'])
	dt10 = int(request.form['j'])
	arr = np.array([[dt2, dt3, dt4, dt5, dt6,dt7,dt8,dt9,dt10]])
	encoded = encoder.fit_transform(arr)
	pred = model.predict(encoded)
	return render_template('predict.html', name=dt1, pred=pred)	

if __name__ == "__main__":
	app.run(debug=True)
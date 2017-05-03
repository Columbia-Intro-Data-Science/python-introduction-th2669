from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home2.html')

@app.route('/getdelay',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form
        origin = result['origin']
        dest = result['dest']
        unique_carrier = result['unique_carrier']
        day_of_week = result['day_of_week']
        dep_hour = result['dep_hour']
        arr_hour = result['arr_hour']

        pkl_file = open('flightdata', 'rb')
        index_dict = pickle.load(pkl_file)
        flightdata_v = np.zeros(len(index_dict))
        
        try:
            flightdata_v[index_dict['DAY_OF_WEEK_'+str(day_of_week)]] = 1
        except:
            pass
        try:
            flightdata_v[index_dict['UNIQUE_CARRIER_'+str(unique_carrier)]] = 1
        except:
            pass
        try:
            flightdata_v[index_dict['ORIGIN_'+str(origin)]] = 1
        except:
            pass
        try:
            flightdata_v[index_dict['DEST_'+str(dest)]] = 1
        except:
            pass
        try:
            flightdata_v[index_dict['DEP_HOUR_'+str(dep_hour)]] = 1
        except:
            pass
        try:
            flightdata_v[index_dict['ARR_HOUR_'+str(arr_hour)]] = 1
        except:
            pass

        pkl_file = open('logmodel_arr.pkl', 'rb')
        logmodel_arr = pickle.load(pkl_file)
        prediction_carr = logmodel_arr.predict(flightdata_v)

        pkl_file = open('logmodel_dep.pkl', 'rb')
        logmodel_dep = pickle.load(pkl_file)
        prediction_cdep = logmodel_dep.predict(flightdata_v)

        pkl_file = open('decisiontree_arr.pkl', 'rb')
        decisiontree_arr = pickle.load(pkl_file)
        prediction_rarr = decisiontree_arr.predict(flightdata_v)

        pkl_file = open('decisiontree_dep.pkl', 'rb')
        decisiontree_dep = pickle.load(pkl_file)
        prediction_rdep = decisiontree_dep.predict(flightdata_v)


        return render_template('result2.html',prediction_carr=prediction_carr
                              ,prediction_cdep=prediction_cdep,
                               prediction_rarr=prediction_rarr,
                               prediction_rdep = prediction_rdep)

    
if __name__ == '__main__':
	app.debug = True
	app.run(host='127.0.0.1', port=8080, debug=True)

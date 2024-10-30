# Importing required functions 
from flask import Flask, render_template , request , jsonify
import datetime
import csv
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
#addre = os.path.join('C:/bikku/Downloads/SLDC_Data', str(year), f"{month:02d}")
overal_dir = 'C:/bikku/Downloads/SLDC_Data/2024/04/02-04-2024.csv'
scrapd = pd.read_csv(overal_dir)
with open(overal_dir, 'r') as file:
    csv_reader = csv.reader(file)
    labelscsv = next(csv_reader)  # Read the header row
dwhole = list(scrapd)
adTs = list(scrapd["TIMESLOT"])
addh = list(scrapd["DELHI"])
adbrpl = list(scrapd["BRPL"])
dbypl = list(scrapd["BYPL"])
dndpl = list(scrapd["NDPL"])
dndmc = list(scrapd["NDMC"])
dmes = list(scrapd["MES"])
dothers = list(scrapd["Other"]) # Read the each rows
dts = []
ddh = []
dbrpl = []
y=0
g=1
z=""
dif=12
for x in adTs:
	z=x
	y=y+1
	if(y==g):
		dts.append(z)
		g=g+dif

for x in addh:
	z=x
	y=y+1
	if(y==g):
		ddh.append(z)
		g=g+dif

for x in adbrpl:
	z=x
	y=y+1
	if(y==g):
		dbrpl.append(z)
		g=g+dif

# Define Plot Data 
	labels = labelscsv

# Flask constructor 
app = Flask(__name__)

# Root endpoint 
@app.route('/')
def homepage():

	


	# Return the components to the HTML template 
	return render_template(
		template_name_or_list='index.html',
		dWhole = dwhole,
		dDh = ddh,
		dTs = dts,
		dBrpl = dbrpl,
		dBypl = dbypl,
		dNdpl= dndpl,
		dNdmc = dndmc,
		dMes = dmes,
		dOthers = dothers,
		labels = labels,
	)

@app.route("/update_date", methods=["POST"])
def update_date():
    selected_date = request.form.get("date")
    # Do something with the selected date (e.g., update a database record)
    # ...
    return jsonify({"message": "Date updated successfully!"})


# Main Driver Function 
if __name__ == '__main__':
	# Run the application on the local development server ##
	app.run(debug=True)


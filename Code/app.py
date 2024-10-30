# Importing required functions 
from flask import Flask, render_template
import csv
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

overal_dir = 'C:/bikku/Downloads/SLDC_Data/2024/04/02-04-2024.csv'
scrapd = pd.read_csv(overal_dir)
with open(overal_dir, 'r') as file:
    csv_reader = csv.reader(file)
    labelscsv = next(csv_reader)  # Read the header row
dwhole = list(scrapd)
dts = list(scrapd["TIMESLOT"])
ddh = list(scrapd["DELHI"])
dbrpl = list(scrapd["BRPL"])
dbypl = list(scrapd["BYPL"])
dndpl = list(scrapd["NDPL"])
dndmc = list(scrapd["NDMC"])
dmes = list(scrapd["MES"])
dothers = list(scrapd["Other"]) # Read the each rows
	



# Flask constructor 
app = Flask(__name__)

# Root endpoint 
@app.route('/')
def homepage():

	# Define Plot Data 
	labels = labelscsv
	

	# Return the components to the HTML template 
	return render_template(
		template_name_or_list='chartjs-example.html',
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


# Main Driver Function 
if __name__ == '__main__':
	# Run the application on the local development server ##
	app.run(debug=True)


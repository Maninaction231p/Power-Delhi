# Importing required functions 
from flask import Flask, render_template , request , jsonify
import csv
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime ,timedelta

todaydate = datetime.now() - timedelta(1)

todayyear = todaydate.year
todaymonth = todaydate.month
todayday = todaydate.day

def importdata(filedate,period):
	year = filedate[0:4]
	month = filedate[5:7]
	day = filedate[8:10]
	addre = f"C:/bikku/Downloads/SLDC_Data/{year}/{month}/{day}-{month}-{year}.csv"
	if os.path.exists(addre):
		return getcsvfile(addre,period)
	else:
		return scrapdate(year,month,day,period)

def getcsvfile(addre,period):
	with open(addre, 'r') as file:
		csv_reader = csv.reader(file)
		labelscsv = next(csv_reader)  # Read the header row
	scrapd = pd.read_csv(addre)
	dwhole = list(scrapd)
	adTs = list(scrapd["TIMESLOT"])
	addh = list(scrapd["DELHI"])
	adbrpl = list(scrapd["BRPL"])
	adbypl = list(scrapd["BYPL"])
	adndpl = list(scrapd["NDPL"])
	adndmc = list(scrapd["NDMC"])
	admes = list(scrapd["MES"])
	adothers = list(scrapd["Other"]) # Read the each rows
	dts = []
	ddh = []
	dbrpl = []
	dbypl = []
	dndpl = []
	dndmc = []
	dmes = []
	dothers = []
	y=0
	g=1
	z=""
	dif = int(period)
	print(dif)
	for x in adTs:
		z=x
		y=y+1
		if(y==g):
			dts.append(z)
			g=g+dif
	y=0
	g=1
	for x in addh:
		z=x
		y=y+1
		if(y==g):
			ddh.append(z)
			g=g+dif
	y=0
	g=1
	for x in adbrpl:
		z=x
		y=y+1
		if(y==g):
			dbrpl.append(z)
			g=g+dif
	# Define Plot Data 
		labels = labelscsv
	y=0
	g=1
	for x in adbypl:
		z=x
		y=y+1
		if(y==g):
			dbypl.append(z)
			g=g+dif
	y=0
	g=1
	for x in adndpl:
		z=x
		y=y+1
		if(y==g):
			dndpl.append(z)
			g=g+dif
	y=0
	g=1
	for x in adndmc:
		z=x
		y=y+1
		if(y==g):
			dndmc.append(z)
			g=g+dif
	# Define Plot Data 
		labels = labelscsv
	y=0
	g=1
	for x in admes:
		z=x
		y=y+1
		if(y==g):
			dmes.append(z)
			g=g+dif
	y=0
	g=1
	for x in adothers:
		z=x
		y=y+1
		if(y==g):
			dothers.append(z)
			g=g+dif
	# Define Plot Data 
	labels = labelscsv 
	return dwhole,dts,ddh,dbrpl,dbypl,dndpl,dndmc,dmes,dothers,labels,period	

def scrapdate(year,month,day,period):
	# URL of the website to scrape
    url = 'http://www.delhisldc.org/Loaddata.aspx?mode='

    target_dir = 'C:/bikku/Downloads/SLDC_Data'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print("Target directory created.")
		
    month_dir,date_str = os.path.join(target_dir, str(year), f"{month}"),f"{day}/{month}/{year}"

    if not os.path.exists(month_dir):
        os.makedirs(month_dir)
        print(f"Directory created for {month}/{year}")
	
    try:
        if (day == '6') and month == '1' and year == '2024':
            print("06/01/2024 data Not Available..")
        elif (int(day) > int(todayday) and int(month) >= int(todaymonth) and int(year) >= int(todayyear)) or (int(month) > int(todaymonth) and int(year) == int(todayyear)) or (int(year) > int(todayyear)):
            print("This is Actuall Data. Future Data Can't be found")
        else: 
            print(f"Scraping data for {date_str}")

             # Send an HTTP GET request to the URL with the date
            response = requests.get(url + date_str)

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'lxml')

            # Find the table with the specific ID
            table = soup.find('table', {'id': 'ContentPlaceHolder3_DGGridAv'})
			
            # Extract headers and data rows from the table
            headers = []
            rows = []
			
            for i, row in enumerate(table.find_all('tr')):
                if i == 0:
                    # Extract headers from the first row
                    headers = [el.text.strip() for el in row.find_all('td')];
                else:
                    # Extract data rows from subsequent rows
                    rows.append([el.text.strip() for el in row.find_all('td')])
			
             # Check if there's data in the table
            if len(rows) > 0:
                # Construct the CSV filename
                csv_filename = os.path.join(month_dir, f"{date_str.replace('/', '-')}.csv")
                # Remove the existing CSV file if it exists
                if os.path.exists(csv_filename):
                    os.remove(csv_filename)
                    print(f"Removed existing CSV file: {csv_filename}")
				
                # Write headers and data to the CSV file
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(rows)
					
                # Optionally clean and save the CSV using pandas (commented out)
                df = pd.read_csv(csv_filename)
                df_clean = df.dropna()
                df_clean['Other'] = round((df_clean['DELHI'] - (df_clean['BRPL']+df_clean['BYPL']+df_clean['NDPL']+df_clean['NDMC']+df_clean['MES'])),2)
                df_clean['Date'] = f"{date_str.replace('/', '-')}"
                df_clean.to_csv(csv_filename, index=False)
            return importdata(f"{year}-{month}-{day}",period)

    
    except Exception as e:
        print(f"Error occurred while scraping {date_str}: {str(e)}")

def rendering(selected_dateopt,perioddataopt,mode):
	dwhole1,dts,ddh1,dbrpl1,dbypl1,dndpl1,dndmc1,dmes1,dothers1,labels1,Perioddata1=importdata(selected_dateopt,perioddataopt)
	return render_template(
		modeopt = mode,
		template_name_or_list='index.html',
		p_data = Perioddata1,
		s_date = selected_dateopt,
		dWhole = dwhole1,
		dDh = ddh1,
		dTs = dts,
		dBrpl = dbrpl1,
		dBypl = dbypl1,
		dNdpl = dndpl1,
		dNdmc = dndmc1,
		dMes = dmes1,
		dOthers = dothers1,
		labels = labels1,
	)
	


selected_datedef = "2024-04-03"
selected_date4 = "2024-05-03"
perioddatadef = '24'
modedef = 0

# Flask constructor 
app = Flask(__name__) 
# Root endpoint 
@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		selected_date = request.form['selected_date']
		mode_data = request.form['mode']
		perioddata0 = request.form.get('pda')
		updated_content = f"You selected: {selected_date} {perioddata0} {mode_data}"
		print(updated_content)
		try:
			return rendering(selected_date,perioddata0,mode_data)
		except Exception as e:
			return rendering(selected_datedef,perioddatadef,modedef)
			print(f"Error occurred while reading {selected_date}: {str(e)}")
			
	else:
		print("Not Posted")

	return rendering(selected_datedef,perioddatadef,modedef)

@app.route('/update_data', methods=['POST'])
def update_data():
    selected_date = request.form['selected_date']
    # Process the selected date here
    return render_template('index.html', selected_date=selected_date)

# Main Driver Function 
if __name__ == '__main__':
	# Run the application on the local development server ##
	app.run(debug=True)


import csv
import os

import pandas as pd
import requests
from bs4 import BeautifulSoup

# URL of the website to scrape
url = 'http://www.delhisldc.org/Loaddata.aspx?mode='

# Define date ranges for scraping
day_range = list(range(1, 32))  # Days 1 to 31
month_range = {
 
   # 2017: [1,2,3,4,5,6,7,8,9,10,11,12],
   #2018: [1,2,3,4,5,6,7,8,9,10,11,12],
   # 2019: [1,2,3,4,5,6,7,8,9,10,11,12],
   # 2020: [1,2,3,4,5,6,7,8,9,10,11,12],
    #2021: [1,2,3,4,5,6,7,8,9,10,11,12],
   #2022: [1,2,3,4,5,6,7,8,9,10,11,12],
 #  2023: [1,2,3,4,5,6,7,8,9,10,11,12],
   2024: [4]
}  # Months to scrape for each year 2017,2018,2019,2020,2021,2022,2023,
year_range = [2024]  # Years to scrape

# Create the target directory if it doesn't exist
target_dir = 'C:/bikku/Downloads/SLDC_Data'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print("Target directory created.")

# Iterate through each year, month, and day
for year in year_range:
    for month in month_range[year]:
        # Create a directory for the specific month and year
        month_dir = os.path.join(target_dir, str(year), f"{month:02d}")
        if not os.path.exists(month_dir):
            os.makedirs(month_dir)
            print(f"Directory created for {month}/{year}")

        try:
            for day in day_range:
                if (day == 6) and month == 1 and year == 2024:
                    print("06/01/2024 data Not Available..")
                else:
                    # Construct the date string
                    date_str = f"{day:02d}/{month:02d}/{year}"
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
                            headers = [el.text.strip() for el in row.find_all('td')]
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
                        df_clean.to_csv(csv_filename, index=False)

        except Exception as e:
            print(f"Error occurred while scraping {date_str}: {str(e)}")


                       

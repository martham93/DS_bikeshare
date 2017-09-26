import requests
import pandas as pd
from time import sleep, strftime, gmtime
import json
import os

# define the city you would like to get information from here:
# for full list see http://api.citybik.es
API_URL = "https://gbfs.bcycle.com/bcycle_boulder/station_status.json"

#Settings:
SAMPLE_TIME = 1800                   # number of seconds between samples, currently 30min
CSV_FILE = "~/Desktop/bcycle_api_30min_3.csv"             # CSV file to save data in

def getAllStationDetails():
    print ("\n\nScraping at " + strftime("%Y%m%d%H%M%S", gmtime()))

    try:
        # this url has all the details
        decoder = json.JSONDecoder()
        station_json = requests.get(API_URL, proxies='')
        station_data = decoder.decode(station_json.text)
    except Exception as err:
        print ("---- FAIL ----")
        print(str(err))
        return None


    print (" --- SUCCESS --- ")
    return station_data["data"]["stations"]

def writeToCsv(data, filename=CSV_FILE, header=False):
    """
    Take the list of results and write as csv to filename.
    """
    data_frame = pd.DataFrame.from_records(data)
    data_frame['time'] = strftime("%Y%m%d%H%M%S", gmtime())
    data_frame.to_csv(filename, header=header, mode="a") #mode a means won't override, will just append #fgure out header = False, set header lables in this


if __name__ == "__main__":

    first = True
    while True:
        station_data = getAllStationDetails()
        if station_data:
            writeToCsv(station_data, filename=CSV_FILE, header=first) #?????
            first = False
        print ("Sleeping for 1800 seconds.")
        sleep(SAMPLE_TIME)

        #still need to manually kill

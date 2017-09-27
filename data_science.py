import pandas as pd
import numpy as np

bcycle = pd.read_csv('/Users/marthamorrissey/Desktop/sheet_checkouts_by_station_map_data.csv', encoding = 'utf-8')


bcycle.head()

bcycle['Entry Pass Type'].unique()


#Figure out over-time

#split big data frame into (2), 1 with the 60min time limit catagories and the other with the 30 min catagories then recombine
bcycle_60 = bcycle[(bcycle['Entry Pass Type'] == 'Annual') | (bcycle['Entry Pass Type']== 'Semester (150-day)')]

bcycle_60['Entry Pass Type'].unique()
bcycle_60.columns

bcycle_60['ot'] = np.where((bcycle_60['Trip Duration (Minutes)'] > 60), 1, 0)


bcycle_30 = bcycle[(bcycle['Entry Pass Type'] != 'Annual') & (bcycle['Entry Pass Type'] != 'Semester (150-day)')]
bcycle_30['Entry Pass Type'].unique()
bcycle_30['ot'] = np.where((bcycle_30['Trip Duration (Minutes)'] > 30), 1, 0)

bcycle_30.head()

frames = [bcycle_30, bcycle_60]

bcycle_new = pd.concat(frames)


bcycle_new.columns

bcycle_new['ot'].unique()

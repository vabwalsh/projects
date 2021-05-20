#!/usr/bin/env python

from endv_all_sites import create_data_dict
from process_flux_data import process_flux_data
import numpy as np
import pandas as pd

def fpar_stats(site_df): 
    
    jan_fpar = []
    feb_fpar = []
    mar_fpar = []
    apr_fpar = []
    may_fpar = []
    jun_fpar = []
    jul_fpar = []
    aug_fpar = []
    sep_fpar = []
    oct_fpar = []
    nov_fpar = []
    dec_fpar = []

    for row_num in range(0, len(site_df)):
        if row_num in range(0,32):
            jan_fpar.append(site_df['fPAR'].loc[row_num])
        elif row_num in range(32,60):
            feb_fpar.append(site_df['fPAR'].loc[row_num])
        elif row_num in range(60,91):
            mar_fpar.append(site_df['fPAR'].loc[row_num])
        elif row_num in range(91,121):
            apr_fpar.append(site_df['fPAR'].loc[row_num])
        elif row_num in range(121,152):
            may_fpar.append(site_df['fPAR'].loc[row_num])
        elif row_num in range(152,182):
            jun_fpar.append(site_df['fPAR'].loc[row_num])
        elif row_num in range(182,213):
            jul_fpar.append(site_df['fPAR'].loc[row_num])
        elif row_num in range(213,244):
            aug_fpar.append(site_df['fPAR'].loc[row_num])
        elif row_num in range(244,274):
            sep_fpar.append(site_df['fPAR'].loc[row_num])
        elif row_num in range(274,305):
            oct_fpar.append(site_df['fPAR'].loc[row_num])
        elif row_num in range(305,335):
            nov_fpar.append(site_df['fPAR'].loc[row_num])
        elif row_num in range(335,366):
            dec_fpar.append(site_df['fPAR'].loc[row_num])
        else :
            print('too long!')
    
    monthly_fpar = []
    for i in jan_fpar, feb_fpar, mar_fpar, apr_fpar, may_fpar, jun_fpar, \
    jul_fpar, aug_fpar, sep_fpar, oct_fpar, nov_fpar, dec_fpar:
        monthly_fpar.append(np.nanmean(i))
    avg_monthly_fpar = np.mean(monthly_fpar)

    yearly_fpar = np.nanmean(avg_monthly_fpar)
    max_fpar = np.max(monthly_fpar)
    min_fpar = np.min(monthly_fpar)
    
    return(yearly_fpar, max_fpar, min_fpar, monthly_fpar)

site_coords = {'vaira': [38.4133, -120.9508],
               'tonzi': [38.4309, -120.9660],
               'ME4': [44.4992, -121.6224],
               'ME5': [44.4372, -121.5668]}

data=np.load('fapar_vals_dict.npy', allow_pickle=True)
data = data[()]

site_wise_data = {}
for site in site_coords:
    site_wise_data[site] = {}
    for key in data:
        if site in str(key):
            site_wise_data[site][key] = np.mean(data[key])

vaira_df = pd.DataFrame({'keys' : site_wise_data['vaira'].keys() , 'fPAR' : site_wise_data['vaira'].values()})
tonzi_df = pd.DataFrame({'keys' : site_wise_data['tonzi'].keys() , 'fPAR' : site_wise_data['tonzi'].values()})
ME4_df = pd.DataFrame({'keys' : site_wise_data['ME4'].keys() , 'fPAR' : site_wise_data['ME4'].values()})
ME5_df = pd.DataFrame({'keys' : site_wise_data['ME5'].keys() , 'fPAR' : site_wise_data['ME5'].values()})

if __name__ == "__main__":
    #return fpar stats
    #yearly data (binned by month), yearly max, yearly min
    print(fpar_stats(vaira_df), fpar_stats(tonzi_df), fpar_stats(ME4_df), fpar_stats(ME5_df))
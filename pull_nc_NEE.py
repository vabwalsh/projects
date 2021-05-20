#!/usr/bin/env python
import netCDF4 as nc
import pandas as pd
import xarray as xr
import numpy as np

def pull_rs_NEE(filepath='TBM_LPJmL_Standardized/TBM_LPJmL_NEE_1d.nc', output_filename='NEE_vals_dict'):
    
    site_coords = {'vaira': [38.4133, -120.9508],
                   'tonzi': [38.4309, -120.9660],
                   'ME4': [44.4992, -121.6224],
                   'ME5': [4.4372, -121.5668]}
    
    A = xr.open_dataset('TBM_LPJmL_Standardized/TBM_LPJmL_NEE_1d.nc', decode_times=False)
    B = A.to_dataframe()
    C = B.reset_index()
    D = C.drop(columns = ['nv','crs','lon_bnds','lat_bnds','time_bnds'])
    E = D[(D['NEE'].isna() == False)]
    F = E[(E['time'] >= 145) & (E['time'] <= 156)]
    
    def filter_NEE_coords(site):
        G = F[(F['lat'] >= site[0] - 0.5) 
              & (F['lat'] <= (site[0] + 0.5))
              & (F['lon'] >= (site[1] - 0.5))
              & (F['lon'] <= (site[1] + 0.5))]
        return(G)

    site_NEE_dict = {}
    for site in site_coords:
        site_NEE = filter_NEE_coords(site_coords[site])
        #T = print(site_NEE[['NEE','time']])#[(site_NEE['NEE'].notna() == True)])
        site_NEE_dict[(site)] = np.array(site_NEE['NEE'][(site_NEE['NEE'].notna() == True)])
        #site_NEE_dict[(T)]
    
    np.save(output_filename+'.npy', site_NEE_dict)
    
##############
if __name__ == "__main__":
    pull_rs_NEE(filepath='TBM_LPJmL_Standardized/TBM_LPJmL_NEE_1d.nc', output_filename='NEE_vals_dict')
    data=np.load('NEE_vals_dict.npy', allow_pickle = True)
    print(data)


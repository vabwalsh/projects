#!/usr/bin/env python

import netCDF4 as nc
import glob, os
import pandas as pd
import xarray as xr
import numpy as np


def create_data_dict(filepath="all_2002_netCDF4_data/*.nc", days_to_load=365, output_filename='fapar_vals_dict'):
    # Order the nc files by date, ascending
    file_names = pd.Series(glob.glob(filepath))
    print("file names:", file_names)

    ordered_dates = sorted(file_names.str.slice(start = 54, stop = 62))

    ordered_files = []

    print("looping ordered dates")
    for day in ordered_dates:
        day_containing_file = file_names[file_names.str.contains(day)].values
        ordered_files.append(day_containing_file)

    ordered_ncs = []
    print("looping ordered files")
    for name in ordered_files:
        ordered_ncs.append(name[0])

    def filter_coords(site):
        site_fpar = fpar_coords[(fpar_coords['latitude'] >= site[0] - 0.5)
                                 & (fpar_coords['latitude'] <= (site[0] + 0.5))
                                 & (fpar_coords['longitude'] >= (site[1] - 0.5))
                                 & (fpar_coords['longitude'] <= (site[1] + 0.5))]
        return(site_fpar)

    #return fpar within my coordinate subset
    #agricultural sites
    site_coords = {'vaira': [38.4133, -120.9508],
                   'tonzi': [38.4309, -120.9660],
                   'ME4': [44.4992, -121.6224],
                   'ME5': [4.4372, -121.5668]}

    print("looping ordered ncs")

    fapar_vals_dict = {}
    for i, file in enumerate(ordered_ncs):

        #use xr to load nc file and convert to DF
        netcdf = xr.open_dataset(file)
        nc_df = netcdf.to_dataframe()

        #adjust DF indexing to make it readable
        A = nc_df.reset_index('time', drop = True)
        B = A.reset_index('nv', drop = True)
        C = B.reset_index('ncrs', drop = True)

        #preserve select only column with fpar values
        #convert lat and lon to columns instead of indices
        fpar = C[['FAPAR']]
        fpar_coords = fpar.reset_index(level = ['latitude', 'longitude'])

        for site in site_coords:
            site_fpar = filter_coords(site_coords[site])
            fapar_vals_dict[(i, site, file)] = np.array(site_fpar['FAPAR'][(site_fpar['FAPAR'].notna() == True)
                                                                           & (site_fpar['FAPAR'] > 0)])
            #print(fapar_vals_dict[(i, site, file)])

        if i>days_to_load:
            break

    np.save(output_filename+'.npy', fapar_vals_dict)

##############
if __name__ == "__main__":
    create_data_dict(filepath="all_2002_netCDF4_data/*.nc", days_to_load=365, output_filename='fapar_vals_dict')
    data=np.load('fapar_vals_dict.npy', allow_pickle=True)
    print(data)
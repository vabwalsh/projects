#!/usr/bin/env python
# coding: utf-8
import pandas as pd


def process_flux_data():
    #Agricultural Sites
    vaira_ranch = pd.read_csv('selected_flux_data/vaira_ranch/FLX_US-Var_FLUXNET2015_FULLSET_MM_2000-2014_1-4.csv')
    tonzi_ranch = pd.read_csv('selected_flux_data/tonzi_ranch/FLX_US-Ton_FLUXNET2015_FULLSET_MM_2001-2014_1-4.csv')

    #Forest Sites
    young_pine = pd.read_csv('selected_flux_data/metolius-first--young_aged_pine/FLX_US-Me5_FLUXNET2015_FULLSET_MM_2000-2002_1-4.csv')
    aged_pine = pd.read_csv('selected_flux_data/metolius-old--aged_ponderosa_pine/FLX_US-Me4_FLUXNET2015_FULLSET_MM_1996-2000_1-4.csv')

    sites = [vaira_ranch, tonzi_ranch, young_pine, aged_pine]

    dfs = []
    insitu = []
    for loc in sites:
        filled = (loc[(loc['TIMESTAMP'] > 200112) & (loc['TIMESTAMP'] < 200301)])

        # Return predicted vairables:
            ### LE and H corrected by energy balance closure factor
            ### In-Situ precip and temperature
            ### Reco (night-time and day-time): 
            ### 50th percentile u* threshhold and  mean for CUT and VUT methods (8 vars total)
        selected = filled[['TIMESTAMP','LE_CORR','H_CORR','RECO_NT_CUT_USTAR50','RECO_DT_CUT_USTAR50',
                           'NEE_CUT_USTAR50','GPP_NT_CUT_USTAR50','GPP_DT_CUT_USTAR50']]
        insite = filled[['TIMESTAMP','TA_F_MDS','P_ERA']]

        df = selected.replace(to_replace = -9999, value = 0)
        at_site = insite.replace(to_replace = -9999, value = 0)

        dfs.append(df)
        insitu.append(at_site)

    vaira = dfs[0]
    tonzi = dfs[1]
    ME5 = dfs[2]
    ME4 = dfs[3]
    
    vaira_i = insitu[0]
    tonzi_i = insitu[1]
    ME5_i = insitu[2]
    ME4_i = insitu[3]
    
    return(dfs, insitu)

if __name__ == "__main__":
    process_flux_data()
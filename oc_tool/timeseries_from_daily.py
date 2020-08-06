
import os, sys, glob

import numpy as np
import pandas as pd
import xarray as xr
import dask.multiprocessing
# dask.set_options(get=dask.multiprocessing.get);  # Enable multicore parallelism


import cmocean as cm
import cartopy as cpy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import regionmask

import satpy

opj = os.path.join

# from oc_tool import utils as u
projet = '/local/AIX/tristan.harmel/project/ardyna/'
satdir = opj(projet, 'satellite')
odir=opj(projet,'satellite/timeseries')

sats = ['modisa', 'modist', 'viirs']
sat = sats[1]
dirL3 = 'L2daily'
ofig = opj(projet, 'fig',dirL3)


# load image data
# ds = xr.open_mfdataset(files, concat_dim='time', preprocess=u.get_time, mask_and_scale=True)
# ds = ds.dropna('time', how='all')
crs = ccrs.NearsidePerspective(100, 71)
land_feat = cpy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face', facecolor=cpy.feature.COLORS['land'])
extent = [90, 180, 71, 78]
# file = files[14]

params = ['chl_oci', 'nflh', 'aot_555', 'angstrom']
params = ['chlor_a_p50']#, 'nflh_mean', 'aot_869_mean']


# ----------------------------------
# format data into timeseries
# ----------------------------------
lon_name = 'lon'
lat_name = 'lat'
lat_min1, lat_max1, lon_min1, lon_max1 = 79, 80.4, 112, 140
lat_min2, lat_max2, lon_min2, lon_max2 = 80.45, 87, 128, 155

roi1 = [[lon_min1, lat_min1], [lon_max1, lat_min1], [lon_max1, lat_max1], [lon_min1, lat_max1]]
roi2 = [[lon_min2, lat_min2], [lon_max2, lat_min2], [lon_max2, lat_max2], [lon_min2, lat_max2]]
roi = [roi1, roi2]
id = [0, 1]
names = ['roi_laptev', 'roi_laptev_north']
abbrevs = ['roi1', 'roi2']
mask = regionmask.Regions(roi, names=names, abbrevs=abbrevs, name='roi')


def ds_stats(ds, lat_name='lat', lon_name='lon'):
    ts_ = ds.mean(dim=(lat_name, lon_name))
    ts_25 = ds.quantile(0.25, dim=(lat_name, lon_name))
    ts_50 = ds.quantile(0.50, dim=(lat_name, lon_name))
    ts_75 = ds.quantile(0.75, dim=(lat_name, lon_name))
    N = ds.count(dim=(lat_name, lon_name))

    return {'ave': ts_, 'q25': ts_25, 'q50': ts_50, 'q75': ts_75, 'N': N}

def stats(ds, lat_name='lat', lon_name='lon',):
    ave = ds.mean(dim=(lat_name, lon_name)).values
    q25 = ds.quantile(0.25, dim=(lat_name, lon_name)).values
    q50 = ds.quantile(0.50, dim=(lat_name, lon_name)).values
    q75 = ds.quantile(0.75, dim=(lat_name, lon_name)).values
    N = ds.count(dim=(lat_name, lon_name)).values

    return [ave, q25, q50, q75, N]

def multiparams_stats(ds, params):
    res = []
    for param in params:
        res.append(stats(ds[param]))
    return np.concatenate(res)

def df_create(params, stat_params):
    return pd.DataFrame(columns=np.concatenate([['sat','ID', 'date'], [p + '_' + s for p in params for s in stat_params]]))


stat_params = ['ave', 'q25', 'q50', 'q75', 'N']
for year in range(2002,2014):
    print(year)
    for sat in sats:

        files = glob.glob(opj(satdir, sat, dirL3,  str(year), '[ATV]*.nc'))
        if not files:
            continue
        files.sort()
        dfroi1, dfroi2 = df_create(params, stat_params), df_create(params, stat_params)
        bins = ["%.1f" % x for x in np.histogram(0,range=[0,30],bins=100)[1][1:]]
        histo_roi1 = pd.DataFrame(columns=np.concatenate([['sat','ID', 'date'],bins] ))
        histo_roi2 = pd.DataFrame(columns=np.concatenate([['sat','ID', 'date'],bins] ))

        i = 0
        for file in files:
            date = os.path.basename(file).split('.')[0][1:]
            #print(pd.to_datetime(date, format='%Y%j'))
            # file='/DATA/projet/ardyna/satellite/level-3.nc'
            #try:
            print(date)
            #---------------
            # read data file
            ds = xr.open_dataset(file, mask_and_scale=True)  # ,group='geophysical_data')

            #---------------
            # mask array for ROIs
            mask_ = mask.mask(ds)
            _roi1 = ds[params].where(mask_ == 0)
            _roi2 = ds[params].where(mask_ == 1)

            histo_roi1.loc[i] = np.concatenate([[sat,'roi1', pd.to_datetime(date, format='%Y%j')], np.histogram(_roi1[params[0]],range=[0,30],bins=100)[0]])
            histo_roi2.loc[i] = np.concatenate([[sat,'roi2', pd.to_datetime(date, format='%Y%j')], np.histogram(_roi2[params[0]],range=[0,30],bins=100)[0]])


            # #---------------
            # # compute stats for each ROI
            stat_roi1 = multiparams_stats(_roi1,params)
            stat_roi2 = multiparams_stats(_roi2,params)

            dfroi1.loc[i] = np.concatenate([[sat,'roi1', pd.to_datetime(date, format='%Y%j')], stat_roi1])
            dfroi2.loc[i] = np.concatenate([[sat,'roi2', pd.to_datetime(date, format='%Y%j')], stat_roi2])

            i += 1
            # except:
            #     continue

            # coords = xr.open_dataset(file,group='navigation_data')
        dfroi1.to_csv(opj(odir,'timeseries_'+sat+'_roi1_'+str(year)+'.csv'),index=False)
        dfroi2.to_csv(opj(odir,'timeseries_'+sat+'_roi2_'+str(year)+'.csv'),index=False)
        histo_roi1.to_csv(opj(odir,'histo_'+sat+'_roi1_'+str(year)+'.csv'),index=False)
        histo_roi2.to_csv(opj(odir,'histo_'+sat+'_roi2_'+str(year)+'.csv'),index=False)

##---------------------------------------
# END


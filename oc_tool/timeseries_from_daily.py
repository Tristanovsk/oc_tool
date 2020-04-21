
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
satdir = opj(projet, 'satellite/modisa/L2daily')
odir=opj(projet,'satellite/modisa/histo')

ofig = opj(projet, 'fig')

# load image data
# ds = xr.open_mfdataset(files, concat_dim='time', preprocess=u.get_time, mask_and_scale=True)
# ds = ds.dropna('time', how='all')
crs = ccrs.NearsidePerspective(100, 71)
land_feat = cpy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face', facecolor=cpy.feature.COLORS['land'])
extent = [90, 180, 71, 78]
# file = files[14]

params = ['chl_oci', 'nflh', 'aot_555', 'angstrom']
params = ['chlor_a_mean']#, 'nflh_mean', 'aot_869_mean']


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
    return pd.DataFrame(columns=np.concatenate([['ID', 'date'], [p + '_' + s for p in params for s in stat_params]]))


stat_params = ['ave', 'q25', 'q50', 'q75', 'N']
for year in range(2002,2020):
    print(year)
    files = glob.glob(opj(satdir, str(year), 'A*.nc'))
    files.sort()
    dfroi1, dfroi2 = df_create(params, stat_params), df_create(params, stat_params)
    bins = ["%.1f" % x for x in np.histogram(0,range=[0,30],bins=100)[1][1:]]
    histo_roi1 = pd.DataFrame(columns=np.concatenate([['ID', 'date'],bins] ))
    histo_roi2 = pd.DataFrame(columns=np.concatenate([['ID', 'date'],bins] ))

    i = 0
    for file in files:
        date = os.path.basename(file).split('.')[0].replace('A', '')
        print(pd.to_datetime(date, format='%Y%j'))
        # file='/DATA/projet/ardyna/satellite/level-3.nc'
        try:
            print(date)
            #---------------
            # read data file
            ds = xr.open_dataset(file, mask_and_scale=True)  # ,group='geophysical_data')

            #---------------
            # mask array for ROIs
            mask_ = mask.mask(ds)
            _roi1 = ds[params].where(mask_ == 0)
            _roi2 = ds[params].where(mask_ == 1)

            histo_roi1.loc[i] = np.concatenate([['roi1', pd.to_datetime(date, format='%Y%j')], np.histogram(_roi1.chlor_a_mean,range=[0,30],bins=100)[0]])
            histo_roi2.loc[i] = np.concatenate([['roi2', pd.to_datetime(date, format='%Y%j')], np.histogram(_roi2.chlor_a_mean,range=[0,30],bins=100)[0]])


            # #---------------
            # # compute stats for each ROI
            # stat_roi1 = multiparams_stats(_roi1,params)
            # stat_roi2 = multiparams_stats(_roi2,params)
            #
            # dfroi1.loc[i] = np.concatenate([['roi1', pd.to_datetime(date, format='%Y%j')], stat_roi1])
            # dfroi2.loc[i] = np.concatenate([['roi2', pd.to_datetime(date, format='%Y%j')], stat_roi2])

            i += 1
        except:
            continue

        # coords = xr.open_dataset(file,group='navigation_data')

    histo_roi1.to_csv(opj(odir,'histo_roi1_'+str(year)+'.csv'),index=False)
    histo_roi2.to_csv(opj(odir,'histo_roi2_'+str(year)+'.csv'),index=False)

##---------------------------------------
# END



import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as po

df = pd.concat([dfroi1,dfroi2])
fig = px.scatter(df, x='date', y='chlor_a_mean_ave',hover_name='chlor_a_mean_N',color='ID',opacity=0.5)#,size=

po.plot(fig)



# Chl_oci param
param = params[0]
ds = dcube[param]
ds = ds.where(ds > 0)
chl_mean, chl_sum, chl_N = ds.mean(dim=('time')), ds.sum(dim=('time')), ds.count(dim='time')

chl_roi1 = ds_stat_mask(ds, 0)
chl_roi2 = ds_stat_mask(ds, 1)

# AOD param
param = params[2]
ds = dcube[param]
ds = ds.where(ds > 0)
aod_mean, aod_sum = ds.mean(dim=('time')), ds.sum(dim=('time'))
aod_roi1 = ds_stat_mask(ds, 0)
aod_roi2 = ds_stat_mask(ds, 1)

# ----------------------------------
#  plot mean and time series*
# ----------------------------------
var = 'q50'
var = 'ave_rel'
cmap = cm.tools.crop_by_percent(cm.cm.delta, 30, which='both')
cice = cm.tools.crop_by_percent(cm.cm.ice, 25, which='min')
caero = cm.tools.crop_by_percent(cmap, 20, which='max')

plt.figure(figsize=(15, 10))
# create subplot grid
G = gridspec.GridSpec(3, 8, left=0.01, wspace=0.25, hspace=0.25)

# ----- aod
ax = plt.subplot(G[0, :3], projection=crs)
ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.add_feature(land_feat)
ax.grid()
ax.gridlines()
ax.coastlines('50m', linewidth=0.5)

mask.plot(ax=ax, regions=[0, 1], add_ocean=False, coastlines=False, label='abbrev', )
p = aod_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=caero, cbar_kwargs=dict(pad=.01, aspect=20, shrink=0.8))
p.set_clim(0, 0.2)

ax = plt.subplot(G[0, 3:])
aod_roi1[var].plot.line(marker='o', label='roi1')
plt.fill_between(aod_roi1[var].time.values, aod_roi1['q25'].values, aod_roi1['q75'].values, alpha=.4)

aod_roi2[var].plot.line(marker='o', label='roi2')
plt.fill_between(aod_roi2[var].time.values, aod_roi2['q25'].values, aod_roi2['q75'].values, alpha=.4)

plt.legend(ncol=2)
# plt.ylim([0,1])

# ----- chl
ax = plt.subplot(G[1, :3], projection=crs)
ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.add_feature(land_feat)
ax.grid()
ax.gridlines()
ax.coastlines('50m', linewidth=0.5)

mask.plot(ax=ax, regions=[0, 1], add_ocean=False, coastlines=False, label='abbrev', )
p = chl_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=caero, cbar_kwargs=dict(pad=.01, aspect=20, shrink=0.8))
p.set_clim(0.1, 20)

ax = plt.subplot(G[1, 3:])
chl_roi1[var].plot.line(marker='o', label='roi1')  # ,yscale='log')
plt.fill_between(chl_roi1[var].time.values, chl_roi1['q25'].values, chl_roi1['q75'].values, alpha=.4)

chl_roi2[var].plot.line(marker='o', label='roi2')
plt.fill_between(chl_roi2[var].time.values, chl_roi2['q25'].values, chl_roi2['q75'].values, alpha=.4)

plt.legend(ncol=2)

# number of pixels

ax = plt.subplot(G[2, :3], projection=crs)
ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.add_feature(land_feat)
ax.grid()
ax.gridlines()
ax.coastlines('50m', linewidth=0.5)

mask.plot(ax=ax, regions=[0, 1], add_ocean=False, coastlines=False, label='abbrev', )
p = chl_N.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=caero, cbar_kwargs=dict(pad=.01, aspect=20, shrink=0.8))

ax = plt.subplot(G[2, 3:])
chl_roi1['N'].plot.line(marker='o', label='roi1')
chl_roi2['N'].plot.line(marker='o', label='roi2')
plt.scatter(x=chl_roi2['N'].time.values, y=chl_roi2['N'].values, s=(chl_roi2['N'] / chl_roi2['N'].max() * 40).values)
plt.legend(ncol=2)
plt.savefig(os.path.join(ofig, 'aot_chl_from_modis_seadas_2014.png'), dpi=300)

import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.express as px
import plotly.offline as po

df = chl_roi2[var].to_dataframe()
data = [go.Scatter(x=df.index, y=df['chl_oci'], mode='markers',
                   marker=dict(size=chl_roi2['N'] / chl_roi2['N'].max() * 40, sizemin=4, line_width=2)
                   )]

po.plot(data, filename='time-series-simple')

#################
## END

# date = ds.start_date.split()[0]
date = os.path.basename(file)[1:8]
date = pd.datetime.strptime(date, '%Y%j')
date = date.date().__str__()
# TODO assign coordinates to ds to plot with georeference
# ds.assign_coords(coords.latitude)
# ds.aot_869.plot()

try:
    plt.figure(figsize=(15, 10))
    param = params[0]
    cmap = cm.tools.crop_by_percent(cm.cm.delta, 30, which='both')
    caero = cm.tools.crop_by_percent(cmap, 20, which='max')
    ax = plt.subplot(2, 2, 1, projection=crs)
    ax
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(land_feat)
    ax.grid()
    ax.gridlines()
    ax.coastlines('50m', linewidth=0.5)
    p = ds[param].where(ds[param] != 0).plot(ax=ax, transform=ccrs.PlateCarree(), norm=mpl.colors.LogNorm(),
                                             cmap=cmap, cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
    p.set_clim(0.1, 30)
    plt.title(date)
    #
    param = params[1]
    cmap = cm.cm.curl  # cm.tools.crop_by_percent(cm.cm.curl,25,which='min')
    ax = plt.subplot(2, 2, 2, projection=crs)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(land_feat)
    ax.coastlines('50m', linewidth=0.5)
    ax.grid()
    ax.gridlines()
    p = ds[param].where(ds[param] != 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                                             cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
    p.set_clim(-0.35, 0.35)
    plt.title(date)
    param = params[2]
    ax = plt.subplot(2, 2, 3, projection=crs)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(land_feat)
    ax.coastlines('50m', linewidth=0.5)
    ax.grid()
    ax.gridlines()
    p = ds[param].where(ds[param] != 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=caero,
                                             cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
    p.set_clim(0, 0.2)
    plt.title(date)
    param = params[3]
    ax = plt.subplot(2, 2, 4, projection=crs)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(land_feat)
    ax.coastlines('50m', linewidth=0.5)
    ax.grid()
    ax.gridlines()
    p = ds[param].where(ds[param] != 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=caero,
                                             cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
    p.set_clim(0, 2)
    plt.title(date)

    plt.savefig(os.path.join(ofig, 'oc_chlor_a_nflh_' + os.path.basename(file).split('.')[0] + '.png'), dpi=400)

    plt.close()
except:
    print('no data available')
ds.close()

file = '/DATA/projet/ardyna/satellite/level-3.nc'
ds = xr.open_dataset(file, mask_and_scale=True)
date = 'Aug-2014'
plt.figure(figsize=(15, 10))

cmap = cm.tools.crop_by_percent(cm.cm.delta, 30, which='both')
caero = cm.tools.crop_by_percent(cmap, 20, which='max')
ax = plt.subplot(2, 2, 1, projection=crs)
ax
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(land_feat)
ax.grid()
ax.coastlines('50m', linewidth=0.5)
ax.gridlines()
p = ds.chlor_a_mean.where(ds.chlor_a_mean != 0).plot(ax=ax, transform=ccrs.PlateCarree(), norm=mpl.colors.LogNorm(),
                                                     cmap=cmap, cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
p.set_clim(0.1, 30)
plt.title(date)
#
cmap = cm.cm.curl  # cm.tools.crop_by_percent(cm.cm.curl,25,which='min')
ax = plt.subplot(2, 2, 2, projection=crs)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(land_feat)
ax.coastlines('50m', linewidth=0.5)
ax.grid()
ax.gridlines()
p = ds.nflh_mean.where(ds.nflh_mean != 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                                               cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
p.set_clim(-0.35, 0.35)
plt.title(date)

ax = plt.subplot(2, 2, 3, projection=crs)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(land_feat)
ax.coastlines('50m', linewidth=0.5)
ax.grid()
ax.gridlines()
p = ds.aot_869_mean.where(ds.aot_869_mean != 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=caero,
                                                     cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
p.set_clim(0, 0.2)
plt.title(date)

ax = plt.subplot(2, 2, 4, projection=crs)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(land_feat)
ax.coastlines('50m', linewidth=0.5)
ax.grid()
ax.gridlines()
p = ds.angstrom_mean.where(ds.angstrom_mean != 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=caero,
                                                       cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
p.set_clim(0, 2)
plt.title(date)

plt.savefig(os.path.join(ofig, 'oc_chlor_a_nflh_' + os.path.basename(file).split('.')[0] + '.png'), dpi=400)

plt.close()

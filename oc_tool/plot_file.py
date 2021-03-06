import os, sys

import numpy as np
import pandas as pd
import xarray as xr
import glob
import datetime

import cmocean as cm
import cartopy as cpy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.ioff()
import satpy

# from oc_tool import utils as u


################
# begin

opj = os.path.join
sst=True
file = sys.argv[1]
odirfig = sys.argv[2]
param = sys.argv[3]
#file='/home/harmel/satellite/modis/sst/L3daily/2014/AQUA_MODIS.20140809.from_L2_SST.nc'

basename = os.path.basename(file)

figfile = opj(odirfig, basename.replace('.nc', '.png'))

crs = ccrs.NearsidePerspective(100, 71)
land_feat = cpy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face', facecolor=cpy.feature.COLORS['land'])
extent = [90, 180, 71, 78]
# file = files[14]
crs = ccrs.PlateCarree()
crs1 = ccrs.NorthPolarStereo()
# crs1 = ccrs.NearsidePerspective(100, 71)
params = ['chlor_a_mean', 'chlor_a_sigma', 'num_obs']
params = ['chlor_a_p50_mean', 'chlor_a_p50_sigma', 'num_obs_mean']
if sst:
    params = ['sst_mean', 'chlor_a_p50_sigma', 'num_obs_mean']

if not os.path.exists(odirfig):
    os.makedirs(odirfig)

# load image data
ds = xr.open_dataset(file, mask_and_scale=True)  # , engine='h5netcdf')  # ,group='geophysical_data')
ds = ds.dropna('lon', how='all')

#---------
# get dayof year of start and end date
doy_start = pd.to_datetime(ds.start_date).dayofyear
doy_end = pd.to_datetime(ds.stop_date).dayofyear
print('start,stop day of year:',doy_start,doy_end)
plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0.05, bottom=0.04, right=0.95, top=0.94,
                    wspace=None, hspace=0.25)
#
#param = params[0]
  # cm.cm.haline #tools.crop_by_percent(cm.cm.delta, 30, which='both')
#caero = cm.tools.crop_by_percent(cmap, 20, which='max')
ax = plt.subplot(projection=crs1)
ax
ax.set_extent(extent, crs=crs)
ax.add_feature(land_feat)

gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--', x_inline=False, y_inline=False)
gl.xlabels_top = False
gl.ylabels_right = False
ax.coastlines('50m', linewidth=0.5)
if sst:
    # ---------------------------------------------------
    # generate SST map
    # ---------------------------------------------------
    cmap = plt.cm.nipy_spectral
    # cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["darkblue","dodgerblue","green","lime","darkorange",'yellow'])
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["slateblue","darkblue","dodgerblue","greenyellow","yellowgreen","darkorange",'yellow'])
    #cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["darkblue","dodgerblue","slateblue","khaki","goldenrod","darkorange",'yellow'])
    p = ds[param].where(ds[param] > 0).plot(ax=ax, transform=crs,
                                            cmap=cmap, cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.9))
    p.set_clim(-1, 6)

    # ---------------------------------------------------
    # generate ice concentration ERA data
    # ---------------------------------------------------
    file_ice = opj('~/Dropbox/work/projet/ardyna/cams/cams', 'era5_ice_artic_jul_aug_2014.nc')
    ds_ice = xr.open_dataset(file_ice)
    # subset
    ds_ice =ds_ice.sel(latitude=slice(90,70))
    # daily average
    ds_ice = ds_ice.groupby('time.dayofyear').mean()
    # get dates
    ds_ice = ds_ice.sel(dayofyear=slice(doy_start,doy_end)).mean(dim='dayofyear')
    # resample
    new_lon = np.linspace(ds_ice.longitude[0], ds_ice.longitude[-1], ds_ice.dims["longitude"] * 4)
    new_lat = np.linspace(ds_ice.latitude[0], ds_ice.latitude[-1], ds_ice.dims["latitude"] * 4)
    ds_ice = ds_ice.interp(latitude=new_lat, longitude=new_lon)
    # remove data if no ice
    ds_ice = ds_ice.where(ds_ice.siconc>0.2)
    ds_ice.siconc.plot(ax=ax, transform=crs,cmap=plt.cm.gist_gray, cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.9))#,zorder=0)
else:

    cmap = plt.cm.nipy_spectral
    p = ds[param].where(ds[param] > 0).plot(ax=ax, transform=crs, norm=mpl.colors.LogNorm(),
                                            cmap=cmap, cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.9))
    p.set_clim(0.1, 10)
year,doy=basename.split('_')[5:7]
doy = doy.split('to')[0]
date = datetime.datetime.strptime(year+doy, '%Y%j')
date = date.date().__str__()
plt.title(date)
ds.close()
# plt.suptitle(date)

plt.savefig(figfile, dpi=300)

plt.close()
# except:
#     print('no data available or issue for ',file)

#
#
#
# file = '/DATA/projet/ardyna/satellite/level-3.nc'
# ds = xr.open_dataset(file, mask_and_scale=True)
# date = 'Aug-2014'
# plt.figure(figsize=(15, 10))
#
# cmap = cm.tools.crop_by_percent(cm.cm.delta, 30, which='both')
# caero = cm.tools.crop_by_percent(cmap, 20, which='max')
# ax = plt.subplot(2, 2, 1, projection=crs)
# ax
# ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.add_feature(land_feat)
# ax.grid()
# ax.coastlines('50m', linewidth=0.5)
# ax.gridlines()
# p = ds.chlor_a_mean.where(ds.chlor_a_mean != 0).plot(ax=ax, transform=ccrs.PlateCarree(), norm=mpl.colors.LogNorm(),
#                                                      cmap=cmap, cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
# p.set_clim(0.1, 30)
# plt.title(date)
# #
# cmap = cm.cm.curl  # cm.tools.crop_by_percent(cm.cm.curl,25,which='min')
# ax = plt.subplot(2, 2, 2, projection=crs)
# ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.add_feature(land_feat)
# ax.coastlines('50m', linewidth=0.5)
# ax.grid()
# ax.gridlines()
# p = ds.nflh_mean.where(ds.nflh_mean != 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
#                                                cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
# p.set_clim(-0.35, 0.35)
# plt.title(date)
#
# ax = plt.subplot(2, 2, 3, projection=crs)
# ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.add_feature(land_feat)
# ax.coastlines('50m', linewidth=0.5)
# ax.grid()
# ax.gridlines()
# p = ds.aot_869_mean.where(ds.aot_869_mean != 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=caero,
#                                                      cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
# p.set_clim(0, 0.2)
# plt.title(date)
#
# ax = plt.subplot(2, 2, 4, projection=crs)
# ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.add_feature(land_feat)
# ax.coastlines('50m', linewidth=0.5)
# ax.grid()
# ax.gridlines()
# p = ds.angstrom_mean.where(ds.angstrom_mean != 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=caero,
#                                                        cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
# p.set_clim(0, 2)
# plt.title(date)
#
# plt.savefig(os.path.join(ofig, 'oc_chlor_a_nflh_' + os.path.basename(file).split('.')[0] + '.png'), dpi=400)
#
# plt.close()

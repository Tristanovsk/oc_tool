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

dirL3 = "L3_summer"
# from oc_tool import utils as u
projet = '/local/AIX/tristan.harmel/project/ardyna/'

odirfig = opj(projet, 'fig/yearly')

# load image data
# ds = xr.open_mfdataset(files, concat_dim='time', preprocess=u.get_time, mask_and_scale=True)
# ds = ds.dropna('time', how='all')
crs = ccrs.NearsidePerspective(100, 71)
land_feat = cpy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face', facecolor=cpy.feature.COLORS['land'])
extent = [90, 180, 71, 78]
# file = files[14]
crs = ccrs.PlateCarree()
crs1 = ccrs.NorthPolarStereo()
# crs1 = ccrs.NearsidePerspective(100, 71)
params = ['chlor_a_mean', 'chlor_a_sigma', 'num_obs']
params = ['chlor_a_p50_mean_mean', 'chlor_a_p50_sigma', 'num_obs_mean']

if not os.path.exists(odirfig):
        os.makedirs(odirfig)
for year in range(2016, 2020):

    date = str(year)
    file_ = 'oc*' + date + '*.nc'
    file = glob.glob(opj(projet, 'satellite', dirL3, date, file_))[0]

    figfile = opj(odirfig, 'oc_' + dirL3 + '_' + date + '_chl.png')
    if os.path.exists(figfile):
        print(figfile)
        continue

    ds = xr.open_dataset(file, mask_and_scale=True)  # , engine='h5netcdf')  # ,group='geophysical_data')
    ds = ds.dropna('lon', how='all')

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.05, bottom=0.04, right=0.95, top=0.94,
                        wspace=None, hspace=0.25)
    #
    param = params[0]
    if ds[param].shape[1] == 0:
        continue
    cmap = plt.cm.nipy_spectral  # cm.cm.haline #tools.crop_by_percent(cm.cm.delta, 30, which='both')
    caero = cm.tools.crop_by_percent(cmap, 20, which='max')
    ax = plt.subplot(projection=crs1)
    ax
    ax.set_extent(extent, crs=crs)
    ax.add_feature(land_feat)

    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--', x_inline=False, y_inline=False)
    gl.xlabels_top = False
    gl.ylabels_right = False
    ax.coastlines('50m', linewidth=0.5)
    p = ds[param].where(ds[param] > 0).plot(ax=ax, transform=crs, norm=mpl.colors.LogNorm(),
                                            cmap=cmap, cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.9))
    p.set_clim(0.1, 10)

    plt.title(date)
    ds.close()
    #plt.suptitle(date)

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

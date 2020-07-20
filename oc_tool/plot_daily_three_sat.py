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

# from oc_tool import utils as u
projet = '/local/AIX/tristan.harmel/project/ardyna/'
sats=['modisa','modist','viirs']
sat = sats[1]
satdir = opj(projet, 'satellite',sat,'L2daily')
odir = opj(projet, 'satellite',sat,'histo')

ofig = opj(projet, 'fig/multisat')

# satdir = os.path.abspath('/home/harmel/satellite/modis/L2mosaic')
# ofig = os.path.abspath('/home/harmel/satellite/modis/fig')
year = 2014
files = glob.glob(opj(satdir, str(year), '[ATV]*.nc'))
files.sort()
# load image data
# ds = xr.open_mfdataset(files, concat_dim='time', preprocess=u.get_time, mask_and_scale=True)
# ds = ds.dropna('time', how='all')
crs = ccrs.NearsidePerspective(100, 71)
land_feat = cpy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face', facecolor=cpy.feature.COLORS['land'])
extent = [90, 180, 71, 78]
# file = files[14]
crs= ccrs.PlateCarree()
crs1 = ccrs.NorthPolarStereo()
#crs1 = ccrs.NearsidePerspective(100, 71)
params = ['chlor_a_p50', 'chlor_a_sigma', 'num_passes']
for year in range(2002,2020):
    odirfig = opj(ofig,str(year))
    if not os.path.exists(odirfig):
        os.makedirs(odirfig)
    for doy in range(181,276):

        date = str(year)+str(doy)
        file = '[ATV]'+date+'*.nc'
        file_a = glob.glob(opj(projet, 'satellite',sats[0],'L2daily',str(year),file))
        file_t = glob.glob(opj(projet, 'satellite',sats[1],'L2daily',str(year),file))
        file_v = glob.glob(opj(projet, 'satellite',sats[2],'L2daily',str(year),file))
        print(file_a,file_t,file_v)

        date = datetime.datetime.strptime(date, '%Y%j')
        date = date.date().__str__()

        figfile = opj(odirfig, 'oc_daily_' + date + '_chl.png')
        if os.path.exists(figfile):
            print(figfile)
            continue
        plt.figure(figsize=(15, 10))
        for i, file in enumerate([file_a,file_t,file_v]):
            if len(file)==0:
                print(file)
                continue
            ds = xr.open_dataset(file[0], mask_and_scale=True)  # , engine='h5netcdf')  # ,group='geophysical_data')
            ds = ds.dropna('lon',how='all')


            param = params[0]
            if ds[param].shape[1]==0:
                continue
            cmap = plt.cm.nipy_spectral  # cm.cm.haline #tools.crop_by_percent(cm.cm.delta, 30, which='both')
            caero = cm.tools.crop_by_percent(cmap, 20, which='max')
            ax = plt.subplot(3,3, i*3+1, projection=crs1)
            ax
            ax.set_extent(extent, crs=crs)
            ax.add_feature(land_feat)
            ax.grid()
            gl=ax.gridlines(draw_labels=True)
            gl.xlabels_top = False
            gl.ylabels_right = False
            ax.coastlines('50m', linewidth=0.5)
            p = ds[param].where(ds[param] > 0 ).plot(ax=ax, transform=crs, norm=mpl.colors.LogNorm(),
                                                     cmap=cmap, cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
            p.set_clim(0.1, 10)
            plt.title(sats[i])
            #
            param = params[1]

            ax = plt.subplot(3,3, i*3+2, projection=crs1)
            ax.set_extent(extent, crs=crs)
            ax.add_feature(land_feat)
            ax.coastlines('50m', linewidth=0.5)
            ax.grid()
            ax.gridlines()
            p = ds[param].where(ds[param] > 0).plot(ax=ax, transform=crs, cmap=cmap,
                                                     cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
            p.set_clim(0, 1)
            plt.title(date)

            param = params[2]
            cmap = cm.cm.curl
            ax = plt.subplot(3,3, i*3+3, projection=crs1)
            ax.set_extent(extent, crs=crs)
            ax.add_feature(land_feat)
            ax.coastlines('50m', linewidth=0.5)
            ax.grid()
            ax.gridlines()
            p = ds[param].where(ds[param] > 0).plot(ax=ax, transform=crs, cmap=caero,
                                                     cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))

            p.set_clim(0, 6)
            plt.title(sats[i])
            ds.close()
        plt.suptitle(date)

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

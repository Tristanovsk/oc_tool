
import os, sys

import numpy as np
import pandas as pd
import xarray as xr
import glob

import cmocean as cm
import cartopy as cpy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import satpy

#from oc_tool import utils as u

satdir = os.path.abspath('/home/harmel/modis/mosaic')
ofig = os.path.abspath('/DATA/projet/ardyna/satellite/fig/MYD04')
files = glob.glob(satdir+'/MYD04*.nc')
files.sort()
# load image data
# ds = xr.open_mfdataset(files, concat_dim='time', preprocess=u.get_time, mask_and_scale=True)
# ds = ds.dropna('time', how='all')
crs = ccrs.NearsidePerspective(100, 71)
land_feat = cpy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face', facecolor=cpy.feature.COLORS['land'])
extent=[90,180, 71, 78]
#file = files[14]

params=["Image_Optical_Depth_Land_And_Ocean","Angstrom_Exponent_1_Ocean"]
for file in files:
    #file='/DATA/projet/ardyna/satellite/level-3.nc'
    ds = xr.open_dataset(file,mask_and_scale=True)#,group='geophysical_data')
    for param in params:
        ds[param]=ds[param]*1e-6
    #coords = xr.open_dataset(file,group='navigation_data')

    # date = ds.start_date.split()[0]
    date = os.path.basename(file)[6:13]
    date = pd.datetime.strptime(date, '%Y%j')
    date = date.date().__str__()
    # TODO assign coordinates to ds to plot with georeference
    # ds.assign_coords(coords.latitude)
    # ds.aot_869.plot()

    try:
        plt.figure(figsize=(15,5))
        param=params[0]
        cmap = cm.tools.crop_by_percent(cm.cm.delta, 30, which='both')
        caero = cm.tools.crop_by_percent(cmap,20,which='max')
        ax =plt.subplot(1,2,1,projection=crs)
        ax
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        #ax.add_feature(land_feat)
        ax.grid()
        ax.gridlines()
        ax.coastlines( '50m',linewidth=0.5)
        p=ds[param].where(ds[param]!=0).plot(ax=ax, transform=ccrs.PlateCarree(),
                          cmap=cmap, cbar_kwargs=dict( pad=.1, aspect=20, shrink=0.6))
        p.set_clim(0,0.4)
        plt.title(date)
        #
        param=params[1]
        cmap = cm.cm.curl #cm.tools.crop_by_percent(cm.cm.curl,25,which='min')
        ax = plt.subplot(1,2,2,projection=crs)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        #ax.add_feature(land_feat)
        ax.coastlines('50m',linewidth=0.5)
        ax.grid()
        ax.gridlines()
        p=ds[param].where(ds[param]!=0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, cbar_kwargs=dict( pad=.1, aspect=20, shrink=0.6))
        p.set_clim(-0.35,2.35)
        plt.title(date)


        plt.savefig(os.path.join(ofig,'oc_chlor_a_nflh_'+os.path.basename(file).split('.')[0]+'.png'),dpi=400)

        plt.close()
    except:
        print('no data available')
    ds.close()


file='/DATA/projet/ardyna/satellite/level-3.nc'
ds = xr.open_dataset(file,mask_and_scale=True)
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
p = ds.nflh_mean.where(ds.nflh_mean != 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                                     cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
p.set_clim(-0.35, 0.35)
plt.title(date)

ax = plt.subplot(2, 2, 3, projection=crs)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(land_feat)
ax.coastlines('50m', linewidth=0.5)
ax.grid()
p = ds.aot_869_mean.where(ds.aot_869_mean != 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=caero,
                                           cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
p.set_clim(0, 0.2)
plt.title(date)

ax = plt.subplot(2, 2, 4, projection=crs)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(land_feat)
ax.coastlines('50m', linewidth=0.5)
ax.grid()
p = ds.angstrom_mean.where(ds.angstrom_mean != 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=caero,
                                             cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
p.set_clim(0, 2)
plt.title(date)

plt.savefig(os.path.join(ofig, 'oc_chlor_a_nflh_' + os.path.basename(file).split('.')[0] + '.png'), dpi=400)

plt.close()
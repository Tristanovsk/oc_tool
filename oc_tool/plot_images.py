
import os, sys, glob

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import dask.multiprocessing
# dask.set_options(get=dask.multiprocessing.get);  # Enable multicore parallelism


import cmocean as cm
import cartopy as cpy
import cartopy.crs as ccrs
import matplotlib.path as mpath

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.ioff()
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
params = ['chlor_a_mean','chlor_a_p50']#, 'nflh_mean', 'aot_869_mean']
years = range(2014,2015) #range(2002,2020)

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
spatial_ref='PROJCS["Polar_Stereographic / World Geodetic System 1984", \
  GEOGCS["World Geodetic System 1984", \
    DATUM["World Geodetic System 1984", \
      SPHEROID["WGS 84", 6378137.0, 298.257223563, AUTHORITY["EPSG","7030"]], \
      AUTHORITY["EPSG","6326"]], \
    PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]], \
    UNIT["degree", 0.017453292519943295], \
    AXIS["Geodetic longitude", EAST], \
    AXIS["Geodetic latitude", NORTH]], \
  PROJECTION["Polar_Stereographic"], \
  PARAMETER["central_meridian", 0.0], \
  PARAMETER["latitude_of_origin", 90.0], \
  PARAMETER["scale_factor", 1.0], \
  PARAMETER["false_easting", 0.0], \
  PARAMETER["false_northing", 0.0], \
  UNIT["m", 1.0], \
  AXIS["Easting", EAST], \
  AXIS["Northing", NORTH]]'
for year in years:
    print(year)
    files = glob.glob(opj(satdir, str(year), 'A2014226*.nc'))
    files.sort()



    i = 0
    for file in files:
        date = os.path.basename(file).split('.')[0].replace('A', '')
        print(pd.to_datetime(date, format='%Y%j'))
        # file='/DATA/projet/ardyna/satellite/level-3.nc'
        try:
            print(date)
            #---------------
            # read data file
            ds = xr.open_dataset(files[0], mask_and_scale=True)  # ,group='geophysical_data')
            ds.rio.crs
            #ds=ds.where(ds[params[0]] > 0) # , drop=True)

        except:
            continue
    # coords = xr.open_dataset(file,group='navigation_data')

        # date = ds.start_date.split()[0]
        date = os.path.basename(file)[1:8]
        date = pd.datetime.strptime(date, '%Y%j')
        date = date.date().__str__()
        # TODO assign coordinates to ds to plot with georeference
        # ds.assign_coords(coords.latitude)
        # ds.aot_869.plot()
        crs = ccrs.PlateCarree()#ccrs.NorthPolarStereo("EPSG6326")
        crs = ccrs.NorthPolarStereo()
        plt.figure(figsize=(15, 10))
        param = params[0]
        cmap = cm.tools.crop_by_percent(cm.cm.delta, 30, which='both')
        caero = cm.tools.crop_by_percent(cmap, 20, which='max')
        ax = plt.subplot(1,1, 1, projection=crs) #=ccrs.PlateCarree())#.NorthPolarStereo())
        da = ds[param].plot(ax=ax,transform=crs)
        #ax.set_extent([-180, 180, 90, 70], ccrs.PlateCarree())
        ax.add_feature(land_feat)
        ax.grid()
        ax.gridlines()
        ax.coastlines('50m', linewidth=0.5)
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)

        ax.set_boundary(circle, transform=ax.transAxes)
        #.dropna(dim=('lon'),how='all').dropna(dim=('lat'),how='all')

        p = ds[param].plot(ax=ax, transform=ccrs.NorthPolarStereo(), norm=mpl.colors.LogNorm(),
                                                 cmap=cmap, cbar_kwargs=dict(pad=.1, aspect=20, shrink=0.6))
        p.set_clim(0.1, 30)
        plt.title(date)
        plt.savefig(os.path.join(ofig, 'oc_chlor_a_daily_' + os.path.basename(file).split('.')[0] + '.png'), dpi=400)
        plt.close()

        try:

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
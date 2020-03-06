
import os, sys

import numpy as np
import pandas as pd
import xarray as xr
import glob
import regionmask
import cmocean as cm
import cartopy as cpy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import satpy

#from oc_tool import utils as u

#------------------------------------------------------------------------
# function to format th time variable for concatenation of the nc images
def get_time(ds, key='start_date'):
    if key in ds.attrs.keys():
        grid_time = pd.to_datetime(ds.attrs[key])
        return ds.assign(time=grid_time)
    raise ValueError("Time attribute missing: {0}".format(key))

#------------------------------------------------------
# set the directories for date etc
satdir = os.path.abspath('/DATA/projet/morin/satellite')
ofig = os.path.abspath('/DATA/projet/ardyna/satellite/fig')
files = glob.glob(satdir+'/*S2*.nc')
files.sort()

#------------------------------------------------------
# load image data series

_f=[]
for f in files:
    d = xr.open_mfdataset(f)
    print(f,'xdim: ',d.dims.get('x'))
    _f.append(f)

print('number of images: ',len(_f))
ds = xr.open_mfdataset(_f, concat_dim='time', preprocess=get_time, mask_and_scale=True, engine='netcdf4')

ds = ds.reindex(time=sorted(ds.time.values))
ds = ds.dropna('time', how='all')

np.nanmedian(ds.spm.values)
#------------------------------------------------------
# plotting part (map)
crs = ccrs.PlateCarree()
land_feat = cpy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face', facecolor=cpy.feature.COLORS['land'])

extent=[90,180, 71, 78]

ax =plt.subplot(1,1,1,projection=crs)

#ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.add_feature(land_feat)
# ax.grid()
# ax.gridlines()
# ax.coastlines( '50m',linewidth=0.5)

var='spm'
p=ds[var].plot(ax=ax)#,transform=ccrs.PlateCarree())
p.set_clim(0,35)
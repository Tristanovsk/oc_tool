import numpy as np
import pandas as pd

from rasterio import features
from affine import Affine
import xarray as xr

import matplotlib.pyplot as plt


def get_time(ds,key='start_date'):

    if key in ds.attrs.keys():
        grid_time = pd.to_datetime(ds.attrs[key])
        return ds.assign(time=grid_time)
    raise ValueError("Time attribute missing: {0}".format(key))

def wktbox(center_lon, center_lat, width=100, height=100, ellps='WGS84'):
        '''

        :param center_lon: decimal longitude
        :param center_lat: decimal latitude
        :param width: width of the box in m
        :param height: height of the box in m
        :return: wkt of the box centered on provided coordinates
        '''
        from math import sqrt, atan, pi
        import pyproj
        geod = pyproj.Geod(ellps=ellps)

        rect_diag = sqrt(width ** 2 + height ** 2)

        azimuth1 = atan(width / height)
        azimuth2 = atan(-width / height)
        azimuth3 = atan(width / height) + pi  # first point + 180 degrees
        azimuth4 = atan(-width / height) + pi  # second point + 180 degrees

        pt1_lon, pt1_lat, _ = geod.fwd(center_lon, center_lat, azimuth1 * 180 / pi, rect_diag)
        pt2_lon, pt2_lat, _ = geod.fwd(center_lon, center_lat, azimuth2 * 180 / pi, rect_diag)
        pt3_lon, pt3_lat, _ = geod.fwd(center_lon, center_lat, azimuth3 * 180 / pi, rect_diag)
        pt4_lon, pt4_lat, _ = geod.fwd(center_lon, center_lat, azimuth4 * 180 / pi, rect_diag)

        wkt_point = 'POINT (%.6f %.6f)' % (center_lon, center_lat)
        wkt_poly = 'POLYGON (( %.6f %.6f, %.6f %.6f, %.6f %.6f, %.6f %.6f, %.6f %.6f ))' % (
            pt1_lon, pt1_lat, pt2_lon, pt2_lat, pt3_lon, pt3_lat, pt4_lon, pt4_lat, pt1_lon, pt1_lat)
        return wkt_poly

def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, latitude='lat', longitude='lon',
              fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.

    usage:
    -----
    1. read shapefile to geopandas.GeoDataFrame
          `states = gpd.read_file(shp_dir)`
    2. encode the different shapefiles that capture those lat-lons as different
        numbers i.e. 0.0, 1.0 ... and otherwise np.nan
          `shapes = (zip(states.geometry, range(len(states))))`
    3. Assign this to a new coord in your original xarray.DataArray
          `ds['states'] = rasterize(shapes, ds.coords, longitude='X', latitude='Y')`

    arguments:
    ---------
    : **kwargs (dict): passed to `rasterio.rasterize` function

    attrs:
    -----
    :transform (affine.Affine): how to translate from latlon to ...?
    :raster (numpy.ndarray): use rasterio.features.rasterize fill the values
      outside the .shp file with np.nan
    :spatial_coords (dict): dictionary of {"X":xr.DataArray, "Y":xr.DataArray()}
      with "X", "Y" as keys, and xr.DataArray as values

    returns:
    -------
    :(xr.DataArray): DataArray with `values` of nan for points outside shapefile
      and coords `Y` = latitude, 'X' = longitude.


    """
    #transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, #transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))

class data:
    def __init__(self):
        pass

    def read_aeronet_ocv3(self, file, skiprows=8):
        ''' Read and format in pandas data.frame the standard AERONET-OC data '''
        self.file = file
        dateparse = lambda x: pd.datetime.strptime(x, "%d:%m:%Y %H:%M:%S")
        ifile=self.file

        h1 = pd.read_csv(ifile, skiprows=skiprows - 1, nrows=1).columns[3:]
        h1 = np.insert(h1,0,'site')
        data_type = h1.str.replace('\[.*\]', '')
        data_type = data_type.str.replace('Exact_Wave.*', 'wavelength')
        #convert into float to order the dataframe with increasing wavelength
        h2 = h1.str.replace('.*\[', '')
        h2 = h2.str.replace('nm\].*', '')
        h2 = h2.str.replace('Exact_Wavelengths\(um\)_','')
        h2 = pd.to_numeric(h2, errors='coerce') #h2.str.extract('(\d+)').astype('float')
        h2 = h2.fillna('').T
        df = pd.read_csv(ifile, skiprows=skiprows, na_values=['N/A', -999.0,-9.999999 ], parse_dates={'date': [1, 2]},
                         date_parser=dateparse, index_col=False)

        # df['site'] = site
        # df.set_index(['site', 'date'],inplace=True)
        df.set_index('date', inplace=True)

        tuples = list(zip(h1, data_type, h2))
        df.columns = pd.MultiIndex.from_tuples(tuples, names=['l0', 'l1', 'l2'])
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.columns = pd.MultiIndex.from_tuples([(x[0], x[1], x[2]) for x in df.columns])
        df.sort_index(axis=1, level=2, inplace=True)
        return df

class plot:
    def __init__(self):
        pass

    def _plot_image(self, data, factor=2.5, vmax=1, title=None, cmap=None, filename=None):
            rows = data.shape[0] // 5 + (1 if data.shape[0] % 5 else 0)
            aspect_ratio = (1.0 * data.shape[1]) / data.shape[2]
            fig, axs = plt.subplots(nrows=rows, ncols=5, figsize=(15, 3 * rows * aspect_ratio))
            for index, ax in enumerate(axs.flatten()):
                if index < data.shape[0] and index < len(data.time):
                    time = pd.to_datetime(data.time[index].values)
                    caption = str(index) + ': ' + time.strftime('%Y-%m-%d')
                    # if self.cloud_coverage is not None:
                    #     caption = caption + '(' + "{0:2.0f}".format(self.cloud_coverage[index] * 100.0) + '%)'

                    ax.set_axis_off()
                    im = ax.imshow(data[index] * factor if data[index].shape[-1] == 3 or data[index].shape[-1] == 4 else
                              data[index] * factor, cmap=cmap, vmin=0.0, vmax=vmax, interpolation='nearest')
                    ax.text(0, -2, caption, fontsize=12)#, color='r') if self.mask[index] else 'g')
                else:
                    ax.set_axis_off()
            fig.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.85,
                     wspace=0.02, hspace=0.2)
            cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)

            fig.suptitle(title, fontsize=18)


            if filename:
                plt.savefig( filename)#, bbox_inches='tight')
                plt.close()
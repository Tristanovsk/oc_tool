import numpy as np
import pandas as pd
import xarray as xr
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

file = '~/Dropbox/work/projet/ardyna/satellite/timeseries/g4.areaAvgTimeSeries.MODISA_L3m_CHL_8d_4km_2018_chlor_a.20020703-20191231.113E_79N_159E_89N.csv'
df = pd.read_csv(file,skiprows=8,na_values=-32767,parse_dates=True,index_col=0)
df.columns=['chl_oci']

df.plot(marker='o', markersize=8, linestyle='-',alpha=0.5, )

df['year']=df.index.year
df['month']=df.index.month
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
pal = sns.cubehelix_palette(18, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="year", hue="year", aspect=15, height=.5, palette="rocket")

# Draw the densities in a few steps
g.map(sns.kdeplot, "chl_oci", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "chl_oci", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)
# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "year")
g.set_xlabels(r'Chl-a from OCI-algo $(mg\ m^{-3})$',fontweight="bold",)
# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)
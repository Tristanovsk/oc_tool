import os
import numpy as np
import pandas as pd
import xarray as xr
import glob
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

opj=os.path.join
projet = '/local/AIX/tristan.harmel/project/ardyna/'
odir=opj(projet,'satellite/modisa/histo')
years = range(2002,2020)
N=len(years)
plt.ioff()
mpl.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(nrows=N, ncols=1, figsize=(15, N*4))
fig.subplots_adjust(left=0.1, right=0.9, hspace=.5, wspace=0.29)
axs=axs.ravel()
dftot=[]
for year in years:
    file = opj(odir,'histo_roi2_'+str(year)+'.csv')
    dftot.append(pd.read_csv(file,index_col=1,parse_dates=True))
df = pd.concat(dftot, axis=0)

df['year']=df.index.year
df['month']=df.index.month
df['doy']=df.index.dayofyear

df=df.set_index(['ID','year','month','doy'])
df=df.drop(index=6,level=2).drop(index=10,level=2)
df.columns=df.columns.astype('float')
df.columns=df.columns.set_names('chl_oci')

for g,d in df.groupby('month').sum().iterrows():
    print(g)


        axs[i].plot(d.index.values,d,label=g)
    axs[i].set_title(str(year))

    plt.legend()
    axs[i].set_xlim([0,10])
    i+=1
plt.tight_layout()
plt.savefig('chlor_a_timeseries.pdf')

df.plot(marker='o', markersize=8, linestyle='-',alpha=0.5, )

from scipy.stats import gaussian_kde
def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)



sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
pal = sns.cubehelix_palette(18, rot=-.25, light=.7)

dff=df.xs(8,level='month').groupby('year').sum().stack().reset_index()

g = sns.FacetGrid(dff, row="year", hue="year", aspect=15, height=.5, palette="rocket",xlim=(0,5))
g.map(plt.bar, 'chl_oci',0, width=0.15,clip_on=False,  alpha=1, lw=1.5)
g.map(plt.plot, 'chl_oci',0, clip_on=False,  color="w", lw=2)

#, bw=.2)
# Draw the densities in a few steps
# g.map(sns.kdeplot, "chl_oci", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
# g.map(sns.kdeplot, "chl_oci", clip_on=False, color="w", lw=2, bw=.2)

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

g.savefig(fname='chlor_a_timeseries_histogram.png',dpi=300)

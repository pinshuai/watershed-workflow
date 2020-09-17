# for users
see official [guide](https://ecoon.github.io/watershed-workflow/build/html/install.html) for installation.

- setup Jupyter virtual enviroment
```
conda create -n ats_meshing -c conda-forge -c defaults python=3 ipython numpy matplotlib scipy meshpy fiona rasterio shapely cartopy descartes ipykernel requests sortedcontainers attrs
```
- Install workflow package
```
cd /path/to/repository
export PYTHONPATH=`pwd`:`pwd`/workflow_tpls:${PYTHONPATH}
```

# for developers
```
conda create -n watershed_workflow_dev -c conda-forge -c defaults python=3 ipython numpy matplotlib scipy meshpy fiona rasterio shapely cartopy descartes pysheds jupyterlab ipykernel requests sortedcontainers attrs pytest sphinx nbsphinx sphinx_rtd_theme
```




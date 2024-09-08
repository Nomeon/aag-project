.. aag-project documentation master file, created by
   sphinx-quickstart on Tue Jul  2 13:49:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AAG-project's Documentation
=======================================

Usage:
------

Add the initial data to a "data" folder inside the pipeline folder.
This is the initial data provided by AAG that has to be cleaned. Using
the provided conda environment, run the pipeline to create a final parquet
file that is used for the next part, clustering.



.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   main
   preprocessing
   customerClustering
   customerPredictions
   itemClustering
   itemPredictions
   helpers  


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

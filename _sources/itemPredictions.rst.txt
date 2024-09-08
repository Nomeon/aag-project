Item predictions
====================

This part of the pipeline is responsible for predicting the future purchases of
the items based on their purchase history. The predictions are done using
the RandomForestRegressor algorithm from the scikit-learn library. The model is
trained on the historical data and then used to predict the future purchases.

.. automodule:: itemPredictions
  :members:
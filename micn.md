Decomposition: The time series is split into seasonal and trend components.
Prediction:
    The seasonal part is predicted using a Seasonal Prediction block, which uses local-global analysis.
    The trend part is predicted using simple regression.
Local-Global Module for the seasonal part:
    Downsampling convolution extracts short-term, local features.
    Isometric convolution captures long-term, global correlations across the time series.
    This approach allows the model to account for both fine-grained and overarching patterns in the seasonal data, while a simpler regression model handles the general trend.







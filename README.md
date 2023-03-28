Crypto Price Prediction by Social Media
==============================

"Crypto Price Prediction by Social Media" is a repository that uses social media data to predict crypto prices. It includes algorithms, models, and techniques for data extraction, cleaning, analysis, and prediction. The repository aims to provide a comprehensive resource for those interested in using social media for crypto price prediction.

```
Now It predicts a drop price movement in next day.
Price prediction will be in future versions.
```

Installation
------------

```
pip install .
```


Usage
==============================

Make DataSet
------------
```
python -m src.data.make_dataset -query cz_binance -target_fraq=0.02
```

```
Usage: python -m src.data.make_dataset [OPTIONS]

  Compute dataset for model.

  Runs data processing scripts to download data from social media and trading
  data to dataset for model training.

Options:
  --social_source TEXT  Social media source (only one)  [default: twitter]
  -query TEXT           Social media query (only one @person)
  -ticker TEXT          Coin for trading data  [default: BTC-USD]
  -target_fraq FLOAT    Define drop for relative price diff  [default: -0.05]
  --help                Show this message and exit.
```

Train model
------------
```
python -m src.models.train_model
```

Model Prediction
------------
```
>>> python -m src.models.predict_model
  Enter text
>>> bitcoin is bad coin
  Prediction of Movement is 0.715 and Movement Verdict is True
```

***
***
***
***
***

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

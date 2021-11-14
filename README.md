# Team Knowledge++

CS5228 Final Project Repository for team - Knowledge++

This repository contains all the artifacts required for executing all 3 tasks. The final report is available in the top-dir named `final_report_knowledge++.pdf`. Read below on instructions for finding artifacts related to each task.

All the notebooks have been committed with outputs so that one can view the results without having to run them again.

## EDA & Preprocessing

`notebooks/eda.ipynb` contains the results of EDA. This was done using data files in `data/raw/`. The observations made during EDA were used in preprocessing raw data. This is done using a pipeline which can be found at `notebooks/data_cleaning_pipeline.ipynb`. You'll notice that this notebook uses custom transformers to clean and process the data - those transformer definitions can be found at `src/transformers.py`. The resulting processed files are store in `data/processed/`.

Note: `notebooks/sgcarmart_scraper.ipynb` contains code used to scrape sgcarmart for true COE expiry data for each entry in the dataset. This inturn is used to clean and pre-process features such as dereg_value, depreciation, etc,. The output of this scraping is stored in `data/raw/test_coe_text.csv` and `data/raw/train_coe_text.csv`.

## Task 1: Prediction of car resale prices

`notebooks/task1.ipynb` contains implementations of all models considered, including hyperparameter tuning and the final regressor used.

## Task 2: Car recommendation

`notebooks/task2.ipynb` provides the interface for running our implementation of recommendation system. To run this notebook you'll need to install the below packages:

- Beautiful Soup (`pip install beautifulsoup4`) - detailed installation instructrion [here](https://pypi.org/project/beautifulsoup4/)
- Requests (`pip install requests`) - detailed installation instructrion [here](https://docs.python-requests.org/en/latest/user/install/#install)
- Jupyter Widgets (`pip install ipywidgets`) - detailed installation instructrion [here](https://ipywidgets.readthedocs.io/en/latest/user_install.html)

Note that the above is in addition to the usual suspects of NumPy, pandas, and scikit-learn. Once you have the dependencies in place, you can go ahead and select `Cell -> Run All` from the jupyter notebook menubar. Select your preferences and click "Find Cars" - you should see results similar to the below screenshot (you can find the same at `data/other/ss_for_readme.png`):

![sample_screenshot](https://github.com/KrishnaKumarHariprasannan/cs5228-kpp/blob/main/data/other/ss_for_readme.png)

Please reach out to one of the contributors listed below if you're unable to run this notebook.

## Task 3: Open task

`notebooks/task3.ipynb` contains implementation along with descriptions of the insights discovered as part of this task.

## Contributors

- [Akshat Jhalani](akshat.jhalani@u.nus.edu)
- [Krishna Kumar Hariprasannan](krishnakh@u.nus.edu)
- [Rishabh Sheoran](rishabh.sheoran@u.nus.edu)
- [Ryo Fukuda](e0744014@u.nus.edu)

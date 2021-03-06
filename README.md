# Classification based on invisible features and thereby finding the effect of tuberculosis vaccine on COVID-19

Environment installation

Use the following command to downlaod the required packages:
```bash
# Install python requirements
pip install -r requirements.txt
```
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

In case of anaconda, use the following command:

* Create a conda environment: `conda env create -f environment.yml`
* Activate the created environment `conda activate [env_name]`

## Usage
### Available Datasets
* After installing the required packages, download the RKI_covid19.csv dataset [here](https://www.arcgis.com/home/item.html?id=f10774f1c63e40168479a1feb6c7ca74)
* Agegroup.csv
* Area.csv
* Landkreis.csv
* covid.json
* income.csv
* incomelandkreis.csv

### Generated Datasets for NN
* AgegroupCovidDataSet.csv 
* CovidDataSet.csv
* DaySeriesActiveCovidDataSet.csv 
* DaySeriesActiveCovidDataSetWithPast.csv
* DaySeriesCovidDataSet.csv
* DaySeriesCovidDataSetAgegroup.csv
* FirstdayCovidDataSet.csv 

To generate the datasets run covid_\*.py files available in Code/covid_\*.py

Once the Dataset is ready(The avialble datset must be updated as the RKI_covid19.csv contains updated information), run the network_\*.py files to train the model with data set. The results will be avialble as plots of different districts.

# eKPI is a Python package based on XGBoost to perform precise prediction of kinase-phosphosite interconnections
# Overview
Phosphorylation in eukaryotic cells plays a key role in regulating cell signaling and disease progression. Despite the ability to detect thousands of phosphosites in a single experiment using high-throughput technologies, the kinases responsible for regulating these sites are largely unidentified. To solve this, we collected the quantitative data at the transcriptional, protein, and phosphorylation levels of 10,159 samples from 23 tumor datasets and 15 adjacent normal tissue datasets. Building on the KPS correlations of different datasets as predictive features, we have developed an innovative approach that employed an oversampling method combined with and XGBoost algorithm (SMOTE-XGBoost) to predict potential kinase-specific phosphorylation sites in proteins.
# Install and use
eKPI could be installed from GitHub. [conda](https://anaconda.org/anaconda/conda) is required to easily install the package. A webserver version of this model could be accessed from https://ekpi.omicsbio.info/.
```
git clone https://github.com/lzxlab/eKPI
cd eKPI
conda env create -f environment.yaml
```
# Python Dependencies
```
dask=2025.1.0
pandas=2.2.3
numpy=2.2.2
scikit-learn=1.6.1
imblearn=0.0
xgboost=2.1.3
joblib=1.4.2
```
# How to run
The eKPI is a easy-to-use command-line package.The model could be run by the following command:
```
cd path_to_eKPI
python predict.py kinase_family input_file output_path
```
Three parameters are needed: `kinase_family` is the kinase family that selected for prediction; `input_file` is a common file with correlation coefficients in each dataset; `output_path` is the path to work and write results.

# Example of input file and output result
The `example` fold containing the examples of input file and output results: the `AGC_example.csv` is an example of input file; the `result.csv` is the output results containing the predict label and probability that represents the confidence of the model prediction; the `runInfo.txt` file records the running information.
# Example running code
```
cd path_to_eKPI
python predict.py AGC ./example/AGC_example.csv ./example
```


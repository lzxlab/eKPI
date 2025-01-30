# eKPI is a Python package based on XGBoost to perform precise prediction of kinase-phosphosite interconnections
# Overview
Phosphorylation in eukaryotic cells plays a key role in regulating cell signaling and disease progression. Despite the ability to detect thousands of phosphosites in a single experiment using high-throughput technologies, the kinases responsible for regulating these sites are largely unidentified. To solve this, we collected the quantitative data at the transcriptional, protein, and phosphorylation levels of 10,159 samples from 23 tumor datasets and 15 adjacent normal tissue datasets. Building on the KPS correlations of different datasets as predictive features, we have developed an innovative approach that employed an oversampling method combined with and XGBoost algorithm (SMOTE-XGBoost) to predict potential kinase-specific phosphorylation sites in proteins.
# Install and use
eKPI could be installed from GitHub. [conda](https://anaconda.org/anaconda/conda) is required to easily install the package. A webserver version of this model could be accessed from https://ekpi.omicsbio.info/.
```
git clone https://github.com/lzxlab/eKPI
cd eKPI
pip install -r requirements.txt
```
# System requirements
**Hardware requirements:** `eKPI` package requires a standard computer with enough RAM and with/without GPU.<br>
**Software requirements:** This package is supported for macOS and Linux. The package has been tested on the following systems: CentOS (el8) and macOS (10.14.1).<br>
**Other requirements:** [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is also needed if you want to run the model based on NVIDIA GPU.
# Python Dependencies
```
pandas=2.2.2
scikit-learn=1.1.2
imblearn=0.11.0
xgboost=2.0.0
joblib=1.2.0
matplotlib=3.6.0
seaborn=0.12.0
```
# How to run
The eKPI is a easy-to-use command-line package.The model could be run by the following command:
```
cd path_to_eKPI
python predict.py input_file output_path
```
Two parameters are needed: `input_file` is a common file with correlation coefficients in each dataset; `output_path` is the path to work and write and results. The `inputFile.csv` in the fold is an example.

# Example of input file and output result
The `example_output` fold containing the examples of input file and output results: the `inputFile.csv` is an example of fasta file pf input; the `merged.data.txt` is the merged output results; the `runInfo.txt` file records the running information.
# Example running code
```
cd path_to_eKPI
python predict.py example_output/inputFile.fasta example_output/
```


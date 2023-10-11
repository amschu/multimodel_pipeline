# multicolormaize



## 1. OVERVIEW <a name="overview"></a>
### **Objective** <a name="objective"></a>

This repository addresses how to:
1) Build machine learning models that predict different phenotypic traits in Maize
2) Compare machine learning models performances to identify the most accurate prediction algorithm/model
3) Explore the extent each data types contribituion to the models performance in order to identigy to most informative data types

## 2. USAGE <a name="usage"></a>
**How to use the materials and sotfware in this repository**

a) Clone the repository:

    `git clone https://gitlab.msu.edu/schum193/cmse802-schumacher.git'`

b) set up conda enviroment 

- .yml file located in `multicolomaize/enviroment.yml`

    
        `conda create env -n MultiColorMaize_Env -f enviroment.yml`

c) run the pipeline and output to a results file

    `python multicolormaize.py > MCM_pipeline_output.txt`

SEE tutorial folder for a usage examples in a jupypter notebook

## 3. MultiColorMaize: Source Files <a name="sourcename"></a>
**This folder includes python source files (.py files).**

1) <details><summary>`_ _ init _ _ .py`</summary><blockquote>
     This lets Python know that this folder will become a Python Package.

2) <details><summary>`multicolormaize.py`</summary><blockquote>
    Establishes a workflow for using `datapreprocessing.py`, `runmodels.py`, and 'featureselection.py`to generate results for MaizeMultiColor predicitons. 

3) <details><summary>`datapreprocessing.py`</summary><blockquote>
    Reads in data and cleans it. Data cleaning include removing duplicates and idenitfying any NAs.
    Outputs a *cleaned* version of inputer data
    

4) <details><summary>`runmodels.py`</summary><blockquote>
    Takes cleaned input data and splits the data into two seperate dataframes. 
    This is done to maintain data intergrity for late evaulation of module perfomance. 
    Data is future dissected into a training data sub group and a validation set. 
    Multiple models will be explored and results are then added into an output file. 
    Additionally, a final containing each models feauture importances score is ouput to an output file for each individual model.


5) <details><summary>`featureselection.py`</summary><blockquote>
    Get the top performing model from the results out. Then gets corresponding feature importance file which is then used to generate new input dataset. The new input set comes from generating a subset from the orginal dataset features only grabbing the top features. 


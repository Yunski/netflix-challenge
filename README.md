# netflix-challenge
Predicting user ratings for films with the Netflix Prize competition dataset.

### Getting Started
Create the conda environment with: 
```
$ conda env create -f environment.yml 
```
If machine has gpu, use environment-gpu.yml.

### Usage
For a [script].py taking command line args, run the following for help   
```
$ python [script].py -h
```
To train a model, run {model}_train.py 
ex: ```python mf_tf_train.py```

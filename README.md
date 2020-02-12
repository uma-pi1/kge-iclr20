# You CAN Teach an Old Dog New Tricks! On Training Knowledge Graph Embeddings

This repository contains the data dumps of all the results we produced, as well as the scripts to reproduce the experiments conducted for the paper ["You CAN Teach an Old Dog New Tricks! On Training Knowledge Graph Embeddings"](https://openreview.net/forum?id=BkxSmlBFvr) published at ICLR 2020. In this work, we conduct an extensive experimental study on the effect of hyperparameter optimization on various Knowledge Graph Embedding (KGE) models. To conduct our experiments, we used the [LibKGE](https://github.com/uma-pi1/kge) framework.

## ICLR 2020 data dumps

The results of each of the trials we run for ICLR 2020 can be found in [data_dumps](data_dumps), where there is a CSV file for each dataset we tested.

## Reproducing our experiments

The following describes how to use the scripts provided in this repository to reproduce our experiments. All scripts can be easily adapted to your scenario.

### Code

To reproduce our experiments, you need to use the <em>libkge</em> framework we developed for them, available [here](www.github.com/uma-pi1/kge). The framework is controlled entirely by config files in YAML format, which we provide here for reproduction purposes.

### Generate the config files

The folder [config_files](config_files) contains the scripts to generate each config file we used in our experiments. The first phase of our experiments is a SOBOL (pseudo-random) hyperparameter search. The file [template_iclr2020.yaml](config_files/template_iclr2020.yaml) is a <em>template</em> that contains the general hyperparameter space used for all models on this phase. In addition, each model used in the experimental study may have specific hyperparameters that need to be set, which are specified in separate config files, e.g. [iclr2020_rescal.yaml](config_files/iclr2020_rescal.yaml). To generate the config files to run the hyperparameter optimization, you may run the following inside the [config_files](config_files) folder:

```sh
python create_config_files.py --prefix iclr2020
```

This takes as input all config files with the given prefix and combines them with [template_iclr2020.yaml](config_files/template_iclr2020.yaml). The output is a folder with the same name of the prefix that contains all configuration files, organized in their specific folders. For each dataset, there is a single config file for each combination of train type (negative sampling, 1vsAll, etc.) and loss function.

### Start the hyperparameter search

Once the config files are created, the following can be run on each of those folders to start the hyperparameter optimization specified in each config file:

```sh
python kge.py resume . --search.num_workers 4 --search.device_pool cuda:0,cuda:1
```

This command runs 4 trials (arms) of the search space simultaneously using two GPUs. The number of trials as well as the devices used can be specified at will, and they are distributed uniformly. For more details on our to use our <em>libkge</em> framework, see [here](www.github.com/uma-pi1/kge).

### Further tuning with Bayesian optimization

Once the SOBOL phase is done, we followed with a Bayesian phase, where we took the categorical settings of the best models found in the pseudo-random phase for each search, and further tuned the continuous hyperparameters with Bayesian optimization. The script [create_bayes_config_files](config_files/create_bayes_config_files) can be used to create the config files for this Bayesian phase. For example, to create the config file for the Bayes search of the model ComplEx on FB15K-237, you may run this on the  folder of that dataset:

```sh
python create_bayes_config_files.py --prefix complex --dump_best_model
```

This command goes through all searches for the ComplEx model and finds the most successful trial, according to the metric used for model selection in the SOBOL phase. It outputs the config file for the Bayesian search inside a new folder with the same name of the winning config file plus the <em>bo</em> suffix. Additionally, the <em>dump_best_model</em> parameter creates a readable file with the settings of the best trial found for the model.

### Include the best models in the Bayesian phase (optional)

The Bayesian phase includes 10 SOBOL trials at first. Optionally, you may want to make sure the best model found in the SOBOL phase is also included in these first trials of the Bayesian search. We provide the script [add_best_model_to_bayes_search.py](config_files/add_best_model_to_bayes_search) for such a purpose. To use it correctly, you need to have dumped the best model settings when creating the Bayes config files, as shown in the previous section. Then, you need to start the search, so the corresponding checkpoint file is created. Once the search is underway, you stop it and run the following:

```sh
python add_best_model_to_bayes_search.py --checkpoint checkpoint.pt --best_model dump_best_model
```

The <em>checkpoint</em> parameter is used to indicate the location of the checkpoint corresponding to the Bayes search. Similarly, the <em>best_model</em> parameter is used to indicate the location of the settings of the trial that is to be added to the search space, which is the file created with the [create_bayes_config_files](config_files/create_bayes_config_files) script.


### Create data dumps with all trials

Finally, the scripts [create_dumps.sh](scripts/create_dumps.sh) and [merge_csvs.sh](scripts/merge_csvs.sh) can be used to create a single CSV file per dataset with the results of all trials in the experiments. To do so, you may run this on the folder of each dataset:

```sh
sh create_dumps.sh kge.py scripts/iclr2020_keys.conf
```

The first parameter indicates the location of the <em>libkge</em> executable and the second indicates the set of keys to be used as columns in the output CSV file in this case [iclr2020_keys.conf](scripts/iclr2020_keys.conf). The output is a CSV file with the result of all trials for each combination of training type and loss function. To merge all of these entries into a single CSV, you may run the following on the folder of each dataset:

```sh
sh merge_csvs.sh
```

This produces a single CSV file with the result of each trial for that dataset, much like those we provide in the [data_dumps](data_dumps) folder.

### Plot the results

To produce the plots used in our work, you may feed your data dumps to the script [create_plots.py](scripts/create_plots.py) like so:

```sh
python create_plots.py --csv iclr2020-wnrr-all-trials.csv --output_folder iclr2020-plots
```

## How to cite

If you use our code or compare against our results please cite the following publication:

```
@inproceedings{
  ruffinelli2020you,
  title={You {\{}CAN{\}} Teach an Old Dog New Tricks! On Training Knowledge Graph Embeddings},
  author={Daniel Ruffinelli and Samuel Broscheit and Rainer Gemulla},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=BkxSmlBFvr}
}
```

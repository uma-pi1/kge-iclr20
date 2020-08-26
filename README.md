# You CAN Teach an Old Dog New Tricks! On Training Knowledge Graph Embeddings

This repository contains the data dumps of all the results we produced, as well as the scripts to reproduce the experiments conducted for the paper ["You CAN Teach an Old Dog New Tricks! On Training Knowledge Graph Embeddings"](https://openreview.net/forum?id=BkxSmlBFvr) published at ICLR 2020. In this work, we conduct an extensive experimental study on the effect of hyperparameter optimization on various Knowledge Graph Embedding (KGE) models. 

To conduct our experiments, we used the [LibKGE](https://github.com/uma-pi1/kge) framework. Since this framework is under continuous development, the instructions and models given here are *tied to a specific version* of that framework (see below). **The [LibKGE website](https://github.com/uma-pi1/kge) contains updated models and configurations options for more recent version of LibKGE.**

## Virtual Poster in ICLR2020

The video presentation for this paper is available [here](https://iclr.cc/virtual_2020/poster_BkxSmlBFvr.html).

## Results

Best performance of all the models tested in our empirical study:

#### FB15K-237 (Freebase)

|                                                                                                       |   MRR | Hits@1 | Hits@3 | Hits@10 |                                                                                      Config file |                                                                              Pretrained model |
|-------------------------------------------------------------------------------------------------------|------:|-------:|-------:|--------:|-------------------------------------------------------------------------------------------------:|----------------------------------------------------------------------------------------------:|
| [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf)                                           | 0.356 |  0.263 |  0.393 |   0.541 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-rescal.yaml) |    [1vsAll-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-rescal.pt) |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) | 0.313 |  0.221 |  0.347 |   0.497 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-transe.yaml) |   [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-transe.pt) |
| [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf)  | 0.343 |  0.250 |  0.378 |   0.531 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-distmult.yaml) | [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-distmult.pt) |
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf)                                           | 0.348 |  0.253 |  0.384 |   0.536 |  [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-complex.yaml) |  [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-complex.pt) |
| [ConvE](https://arxiv.org/abs/1707.01476)                                                             | 0.339 |  0.248 |  0.369 |   0.521 |    [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-conve.yaml) |     [1vsAll-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-conve.pt) |

#### WN18RR (Wordnet)

|                                                                                                       |   MRR | Hits@1 | Hits@3 | Hits@10 |                                                                                 Config file |                                                                        Pretrained model |
|-------------------------------------------------------------------------------------------------------|------:|-------:|-------:|--------:|--------------------------------------------------------------------------------------------:|----------------------------------------------------------------------------------------:|
| [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf)                                           | 0.467 |  0.439 |  0.480 |   0.517 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-rescal.yaml) |   [KvsAll-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-rescal.pt) |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) | 0.228 |  0.053 |  0.368 |   0.520 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-transe.yaml) |  [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-transe.pt) |
| [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf)  | 0.452 |  0.413 |  0.466 |   0.530 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-distmult.yaml) | [KvsAll-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-distmult.pt) |
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf)                                           | 0.475 |  0.438 |  0.490 |   0.547 |  [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-complex.yaml) |  [1vsAll-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-complex.pt) |
| [ConvE](https://arxiv.org/abs/1707.01476)                                                             | 0.442 |  0.411 |  0.451 |   0.504 |    [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-conve.yaml) |    [KvsAll-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-conve.pt) |

## ICLR 2020 data dumps

The results of each of the trials we run for ICLR 2020 can be found in [data_dumps](data_dumps), where there is a CSV file for each dataset we tested.

## Reproducing our experiments

The following describes how to use the scripts provided in this repository to reproduce our experiments. These scripts are highly specific, and you will likely need to adapt them to your setting.

### Code

To conduct our experiments, we used the <em>LibKGE</em> framework available [here](https://github.com/uma-pi1/kge). The framework is controlled entirely by config files in YAML format, which we provide [here](data_dumps) for reproduction purposes.

To retrieve the version that we used in the experiments use the following code

```sh
git clone https://github.com/uma-pi1/kge
cd kge
git checkout 4b2414a
```

### Generate the config files

The folder [config_files](config_files) contains the scripts to generate each config file we used in our experiments as well as the config files themselves. The first phase of our experiments is a SOBOL (pseudo-random) hyperparameter search. The file [template_iclr2020.yaml](config_files/templates_iclr2020.yaml) is a <em>template</em> that contains the general hyperparameter space used for all models on this phase. In addition, each model used in the experimental study may have specific hyperparameters that need to be set, which are specified in separate config files, e.g. [iclr2020_rescal.yaml](config_files/iclr2020_rescal.yaml). To generate the config files to run the hyperparameter optimization, run the following inside the [config_files](config_files) folder:

```sh
python create_config_files.py --prefix iclr2020
```

This takes as input all config files with the given prefix and combines them with [templates_iclr2020.yaml](config_files/templates_iclr2020.yaml). The output is a folder with the same name of the prefix that contains all configuration files, organized in their specific folders. For each dataset, there is a single config file for each combination of train type (negative sampling, 1vsAll, etc.) and loss function. Note that we set the loss to "kl" for KL-divergence in our experiments, which is equivalent to "ce" for cross-entropy, as reported in the paper.

### Adding a new model to the search space

To optimize a new model with the same search space, add the model to [LibKGE](https://github.com/uma-pi1/kge), and specify a config file with the specific additional settings for the search space required by your model, e.g. [iclr2020_conve.yaml](config_files/iclr2020_conve.yaml). Then create the config file as described in the previous section.

### Start the hyperparameter search

Once the config files are created, run the following on each of those folders to start the hyperparameter optimization specified in each config file:

```sh
kge resume . --search.num_workers 4 --search.device_pool cuda:0,cuda:1
```

This command runs 4 trials (arms) of the search space simultaneously using two GPUs. The number of trials as well as the devices used can be specified at will, and they are distributed uniformly. For more details on our to use our <em>LibKGE</em> framework, see [here](https://github.com/uma-pi1/kge).

### Further tuning with Bayesian optimization

Once the SOBOL phase is done, we followed with a Bayesian phase, where we took the categorical settings of the best models found in the pseudo-random phase for each search, and further tuned the continuous hyperparameters with Bayesian optimization. The script [create_bayes_config_files](config_files/create_bayes_config_files.py) can be used to create the config files for this Bayesian phase. For example, to create the config file for the Bayes search of the model ComplEx on FB15K-237, you may run this on the folder of that dataset:

```sh
python create_bayes_config_files.py --prefix complex --dump_best_model
```

This command goes through all searches for the ComplEx model and finds the most successful trial, according to the metric used for model selection in the SOBOL phase. It outputs the config file for the Bayesian search inside a new folder with the same name of the winning config file plus the <em>bo</em> suffix. Additionally, the <em>dump_best_model</em> parameter creates a readable file with the settings of the best trial found for the model.

### Include the best models in the Bayesian phase (optional)

The Bayesian phase includes 10 SOBOL trials at first. Optionally, it is possible to make sure the best model found in the SOBOL phase is also included in these first trials of the Bayesian search. We provide the script [add_best_model_to_bayes_search.py](config_files/add_best_model_to_bayes_search.py) for such a purpose. To use it correctly, you need to have dumped the best model settings when creating the Bayes config files, as shown in the previous section. Then, start the search, so the corresponding checkpoint file is created. Once the search is underway, you stop it and run the following:

```sh
python add_best_model_to_bayes_search.py --checkpoint checkpoint.pt --best_model dump_best_model
```

The <em>checkpoint</em> parameter is used to indicate the location of the checkpoint corresponding to the Bayes search. Similarly, the <em>best_model</em> parameter is used to indicate the location of the settings of the trial that is to be added to the search space, which is the file created with the [create_bayes_config_files](config_files/create_bayes_config_files.py) script.

### Create data dumps with all trials

Finally, the scripts [create_dumps.sh](scripts/create_dumps.sh) and [merge_csvs.sh](scripts/merge_csvs.sh) can be used to create a single CSV file per dataset with the results of all trials in the experiments. To do so, you may run this on the folder of each dataset:

```sh
sh create_dumps.sh scripts/iclr2020_keys.conf
```

The first parameter indicates the location of the <em>LibKGE</em> executable and the second indicates the set of attributes in the output CSV file ([iclr2020_keys.conf](scripts/iclr2020_keys.conf). The output is a CSV file with the result of all trials for each combination of training type and loss function. To merge all of these entries into a single CSV, run the following on the folder of each dataset:

```sh
sh merge_csvs.sh
```

This produces a single CSV file with the result of each trial for that dataset, like those we provide in the [data_dumps](data_dumps) folder.

### Train best models 5 times

To produce the config file to train the best models 5 times, run the following script [create_best_models_search_files.py](config_files/create_best_models_search_files.py) on each dataset's folder:

```sh
python create_best_models_search_files.py --prefix complex
```

Using [create_dumps.sh](scripts/create_dumps.sh) and [merge_csvs.sh](scripts/merge_csvs.sh) of these folders, you can produce a CSV that includes only the best trials, which are useful for the script that generates tables.

### Produce config files for the best models

To generate config files for training the best models obtained in the search, you need [create_best_models_config_files.sh](scripts/create_best_models_config_files.sh) and [get_best_trial.py](scripts/get_best_trial.py) in each dataset folder. Then run:

```sh
sh create_best_models_config_files.sh
```

This script assumes you've created and trained the best models 5 times as instructed in the previous section. The output files appear in the corresponding dataset's folder under the name "dataset_model_config_checkpoint_best.yaml".

### Plot the results

To produce the plots used in our work, feed your data dumps to the script [create_plots.py](scripts/create_plots.py) like so:

```sh
python create_plots.py --csv iclr2020-fb15k-237-all-trials.csv,iclr2020-wnrr-all-trials.csv \
--output_folder iclr2020-plots
```

### Create the tables

To produce the tables used in our work, feed your data dumps to the script [create_tables.py](scripts/create_tables.py) like so:

```sh
python create_tables.py --all_trials iclr2020-fb15k-237-all-trials.csv,iclr2020-wnrr-all-trials.csv \
--best_trials iclr2020-fb15k-237-best-trials,iclr2020-wnrr-best-trials
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

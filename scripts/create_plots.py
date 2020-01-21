import argparse
import pandas
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_category_order_labels(category_order, attribute, model, row):
    """ Hand-crafted category order and corresponding labels for x-axis in box plots """

    labels = []
    if attribute == "train_type_loss":
        if current_model == 'transe':
            category_order = ['negative_sampling-margin_ranking', 'negative_sampling-kl']
            if not len(labels):
                labels = ['NegSamp+MR', 'NegSamp+CE']
        else:
            category_order = ['negative_sampling-margin_ranking',
                    'negative_sampling-bce',
                    '1vsAll-bce',
                    'KvsAll-bce',
                    'negative_sampling-kl',
                    '1vsAll-kl',
                    'KvsAll-kl'
                    ]
            if not len(labels):
                labels = ['NegSamp+MR', 
                        'NegSamp+BCE', '1vsAll+BCE', 'KvsAll+BCE', 
                        'NegSamp+CE', '1vsAll+CE', 'KvsAll+CE']
    elif attribute == "reciprocal":
        if current_model == "conve":
            category_order = [1]
            labels = ["Reciprocal"]
        else:
            category_order = [0, 1]
            labels = ["No Reciprocal", "Reciprocal"]
    elif attribute == "dropout_e":
        category_order = [0, 1]
        labels = ["No Dropout", "Dropout"]
    elif attribute == "dropout_r":
        category_order = [0, 1]
        labels = ["No Dropout", "Dropout"]
    elif attribute == "emb_initialize":
        category_order = ["normal_", "uniform_", "xavier_normal_", "xavier_uniform_"]
        labels = ["Normal", "Unif.", "XvNorm", "XvUnif"]
    elif attribute == "emb_regularize_p":
        category_order = [0.0, 1.0, 2.0, 3.0]
        labels = ["None", "L1", "L2", "L3"]
    else:
        labels = category_order

    # no labels for top row
    if row == 0:
        labels = []
        for n in range(len(category_order)):
            labels.append('')

    return category_order, labels


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help="csvs created with dump command, comma separated")
    parser.add_argument('--output_folder', type=str, required=True, help="name of output folder")
    args, _ = parser.parse_known_args()

    # determine which attributes to plot
    attributes = ['train_batch_size',
            'train_type_loss',   
            'train_optimizer',
            'emb_e_dim', 
            'emb_regularize_p', 
            'emb_initialize',
            'reciprocal',
            'dropout_e',
            'dropout_r',
            ]

    # load input CSVs
    csvs = []
    for input_file in args.csv.split(","):
        csvs.append(pandas.read_csv(input_file))
    all_data = pandas.concat(csvs)
    
    # add train type and loss combination column
    all_data['train_type_loss'] = all_data['train_type'] + '-' +  all_data['train_loss']
    all_data = all_data.drop(columns=["train_type", "train_loss"])

    # add dropout/no dropout column
    all_data['dropout_e'] = np.where(all_data.emb_e_dropout > 0, 1, 0)
    all_data['dropout_r'] = np.where(all_data.emb_r_dropout > 0, 1, 0)

    # deal with empty string in emb_regularize_p
    all_data['emb_regularize_p'] = all_data['emb_regularize_p'].fillna(0)

    # create output folder
    output_folder = args.output_folder
    if os.path.isdir(output_folder):
        raise ValueError('Output folder already exists.')
    os.mkdir(output_folder)

    # set datasets 
    datasets = ["fb15k-237", "wnrr"]
    dataset_labels = ["FB15K-237", "WNRR"]

    # set models
    models = all_data.model.unique()

    # set metric label
    metric_label = "MRR"

    # order and label for models
    model_order = ['rescal', 'transe', 'distmult', 'complex', 'conve']
    model_labels = ['RESCAL', 'TransE', 'DistMult', 'ComplEx', 'ConvE']

    # plot MRR per model
    print("Plotting metric per model...")
    f, axes = plt.subplots(1, len(datasets), sharex=True, sharey=True)
    f.set_size_inches(12, 5)
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    for i in range(len(datasets)):
        current_dataset = datasets[i]
        title = dataset_labels[i]

        # get data for current dataset
        dataset_data = all_data.loc[all_data['dataset'] == current_dataset]

        # create plots
        mrr_per_model = sns.boxplot(y=dataset_data['metric'], 
                x=dataset_data['model'], 
                order=model_order,
                width=0.8,
                ax=axes[i]).set_title(title)

    for ax in axes.flat:
        ax.set(xlabel='', ylabel=metric_label)
        ax.set_xticklabels(model_labels, fontsize=10)

    mrr_per_model.get_figure().savefig(output_folder + '/metric_per_model.png', dpi=300)

    # plot each attribute vs metric
    for attribute in attributes:
        print("Plotting {}...".format(attribute))

        # create plot
        label_rotation=30
        num_rows = len(datasets)
        num_cols = len(models)
        f, axes = plt.subplots(num_rows, num_cols, sharex=False, sharey=True)
        if attribute == "train_type_loss":
            f.set_size_inches(10, 5)
        else:
            f.set_size_inches(8, 6)
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        plt.xticks(rotation=label_rotation)

        # each model is a column in the main plot
        for col in range(len(model_order)):
            current_model = model_order[col]
            title = model_labels[col]

            # get data for current model
            model_data = all_data.loc[all_data['model'] == current_model]

            # create each row of the image
            for row in range(len(datasets)):
                current_dataset = datasets[row]

                # set order and labels for categories in x axis of box plot
                category_order = sorted(model_data[attribute].fillna(0).unique())
                category_order, labels = get_category_order_labels(category_order, attribute, current_model, row)

                dataset_data = model_data.loc[model_data['dataset'] == current_dataset]
                box = sns.boxplot(x=dataset_data[attribute],
                        y=dataset_data['metric'],
                        order=category_order,
                        ax=axes[row][col]).set_title(title)

                # add xticks labels
                axes[row][col].set_xticklabels(labels, ha='right')

        for ax in axes.flat:
            plt.sca(ax)
            plt.xticks(rotation=label_rotation)
            ax.set(xlabel='', ylabel=metric_label)

        # save figure for current attribute            
        box.get_figure().savefig(output_folder + '/' + attribute + '.png', dpi=300)

    print('DONE! Find plots in folder:', output_folder)


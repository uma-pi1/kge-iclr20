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
    elif attribute == "dropout":
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
            'dropout'
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
    all_data['dropout'] = np.where((all_data.emb_e_dropout > 0) | (all_data.emb_r_dropout > 0), 1, 0)

    # deal with empty string in emb_regularize_p
    all_data['emb_regularize_p'] = all_data['emb_regularize_p'].fillna(0)

    # make sure no Bayesian trials are included
    all_data = all_data.loc[~all_data['folder'].str.contains("-bo")]
    
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
    metric_label = "Validation MRR"

    # order and label for models
    model_order = ['rescal', 'transe', 'distmult', 'complex', 'conve']
    model_labels = ['RESCAL', 'TransE', 'DistMult', 'ComplEx', 'ConvE']

    # plot MRR per model
    print("Plotting metric per model...")
    f, axes = plt.subplots(1, len(datasets), sharex=True, sharey=True)
    f.set_size_inches(6.4, 2.4)
    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    font_size = 7.5
    label_rotation=15
    for i in range(len(datasets)):
        current_dataset = datasets[i]
        title = dataset_labels[i]

        # get data for current dataset
        dataset_data = all_data.loc[all_data['dataset'] == current_dataset]

        # create plots
        mrr_per_model = sns.boxplot(y=dataset_data['metric']*100, 
                x=dataset_data['model'], 
                order=model_order,
                linewidth=0.5,
                fliersize=1,
                width=0.8,
                ax=axes[i])
        mrr_per_model.set_title(title, size=font_size)
        mrr_per_model.tick_params(labelsize=font_size)
        if i == 0:
            mrr_per_model.set_ylabel(metric_label, fontsize=font_size)
        else:
            mrr_per_model.set_ylabel('')

    for ax in axes.flat:
        ax.set_xticklabels(model_labels)
        ax.set(xlabel='')

    # save figure
    mrr_per_model.get_figure().savefig(
            output_folder + '/metric_per_model.pdf', 
            dpi=300,
            bbox_inches='tight',
            pad_inches=0)

    # plot each attribute vs metric
    for attribute in attributes:
        attribute_name = attribute
        print("Plotting {}...".format(attribute_name))

        # create plot
        label_rotation=30
        font_size = 7.5
        num_rows = len(datasets)
        num_cols = len(models)

        # manage with/no penalty dropouts
        if "_no_penalty" in attribute_name:
            attribute = attribute[:-len("_no_penalty")]
        elif "_with_penalty" in attribute_name:
            attribute = attribute[:-len("_with_penalty")]

        # skip Conve if reciprocal
        if attribute == "reciprocal":
            num_cols = len(models) - 1

        f, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
        if attribute == "train_type_loss":
            f.set_size_inches(6.1, 3.5)
        elif attribute == "reciprocal":
            f.set_size_inches(4.5, 4)
        else:
            f.set_size_inches(6, 4)
        plt.subplots_adjust(hspace=0.1, wspace=0.2)
        plt.xticks(rotation=label_rotation)

        # add dataset names to subplot rows
        row_names = ["FB15K-237", "WNRR"]
        pad = 5
        for ax, row in zip(axes[:,0], row_names):
            ax.annotate(row, xy=(0, 0.5), rotation=90, xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size=font_size, ha='right', va='center')

        # each model is a column in the main plot
        for col in range(len(model_order)):
            current_model = model_order[col]
            title = model_labels[col]

            # skip Conve if reciprocal
            if attribute == "reciprocal" and current_model == "conve":
                continue

            # get data for current model
            model_data = all_data.loc[all_data['model'] == current_model]

            # manage with/no penalty dropouts
            if "_no_penalty" in attribute_name:
                model_data = model_data.loc[model_data['emb_regularize_p'] == 0]
            elif "_with_penalty" in attribute_name:
                model_data = model_data.loc[model_data['emb_regularize_p'] > 0]

            # each dataset is a row in the main plot
            for row in range(len(datasets)):
                current_dataset = datasets[row]

                # set order and labels for categories in x axis of box plot
                category_order = sorted(model_data[attribute].fillna(0).unique())
                category_order, labels = get_category_order_labels(category_order, attribute, current_model, row)
                if attribute == "train_type_loss":
                    global_legends = labels
                    labels = ['', '', '', '', '', '', '']

                dataset_data = model_data.loc[model_data['dataset'] == current_dataset]
                if attribute == "train_type_loss":
                    colors=sns.color_palette()
                    box = sns.boxplot(x=dataset_data[attribute],
                            y=dataset_data['metric']*100,
                            order=category_order,
                            linewidth=0.5,
                            fliersize=1,
                            palette=colors,
                            ax=axes[row][col]
                            )
                else:
                    box = sns.boxplot(x=dataset_data[attribute],
                            y=dataset_data['metric']*100,
                            order=category_order,
                            linewidth=0.5,
                            fliersize=1,
                            ax=axes[row][col]
                            )

                if row != len(datasets) - 1:
                    box.set_title(title, size=font_size)
                box.tick_params(labelsize=font_size)
                if col == 0:
                    box.set_ylabel(metric_label, fontsize=font_size)
                else:
                    box.set_ylabel('')

                # add xticks labels
                axes[row][col].set_xticklabels(labels, ha='right')

        # add labels to box plots
        for ax in axes.flat:
            plt.sca(ax)
            plt.xticks(rotation=label_rotation)
            ax.set(xlabel='')

        # add legend
        if attribute == "train_type_loss":
            proxies = []
            for i in range(len(global_legends)):
                proxies.append(plt.Rectangle(
                    (0,0), 
                    1, 
                    1, 
                    ec='k', 
                    fc=colors[i], 
                    linewidth=0.5,
                    label=global_legends[i]))
            f.legend(
                    prop={'size': 5.5},
                    handles=proxies,
                    bbox_to_anchor=(0.01, 0.02), 
                    loc='lower left', 
                    borderaxespad=0.,
                    columnspacing=1,
                    ncol=7, 
                    frameon=False,
                    )

        # save figure for current attribute            
        box.get_figure().savefig(
                output_folder + '/' + attribute_name + '.pdf', 
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)

    print('DONE! Find plots in folder:', output_folder)


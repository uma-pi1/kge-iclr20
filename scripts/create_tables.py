import argparse
import pandas


def mean_std_table(
        datasets, 
        dataset_labels, 
        metrics, 
        metric_labels, 
        model_order, 
        model_labels, 
        all_data,
        output_file):

        # set output_file
        output_file = open(output_file, "w")

        # stats
        stats = ["Mean", "Median", "Peak"]

        # keep relevant columns
        all_data = all_data[["model", "dataset", metrics[0], metrics[1]]]

        # write table headers
        output_file.write("\\begin{table}[h!]\n")
        output_file.write("  \\begin{center}\n")
        output_file.write("    \\begin{tabular}{\n")
        output_file.write("      l\n")
        output_file.write("      p{1cm}\n")
        output_file.write(" c@{\hspace{0.5cm}}\n")
        output_file.write(" c@{\hspace{0.5cm}}\n")
        output_file.write(" c@{\hspace{0.5cm}}\n")
        output_file.write(" c@{\hspace{0.5cm}}\n")
        output_file.write(" c\n")
        output_file.write("    }\n")
        output_file.write("    \\toprule\n")
        output_file.write("    &        & {")
        for model in model_order:
            if model == model_order[-1]:
                output_file.write(model_labels[model] + "}        \\\\ \midrule\n")
            else:
                output_file.write(model_labels[model] + "}        & {")

        # get data
        mean = all_data.groupby(["model", "dataset"], as_index=False).mean()
        std = all_data.groupby(["model", "dataset"]).std().reset_index()

        # populate table
        for dataset in range(len(dataset_labels)):
            output_file.write("    \\multirow{5}*{\\begin{turn}{90}\\emph{" + dataset_labels[dataset] + "}\\end{turn}}\n")
            output_file.write("    \\\\[0.2cm]\n")
            for metric in range(len(metrics)):
                output_file.write("    & " + metric_labels[metric] + " & ")
                for model in model_order:
                    entry_mean = mean.loc[(mean["model"] == model) & (mean["dataset"] == datasets[dataset])]
                    entry_std = std.loc[(std["model"] == model) & (std["dataset"] == datasets[dataset])]
                    output_file.write("{:.1f}$\\pm${:.1f}".format(
                        round(entry_mean[metrics[metric]].item()*100, 1),
                        round(entry_std[metrics[metric]].item()*100, 1))
                        )
                    if model == model_order[-1]:
                        output_file.write("  \\\\")
                        if metrics[metric] == metrics[-1]:
                            if datasets[dataset] == datasets[-1]:
                                output_file.write("\n    \\\\[0.2cm] \n    \\bottomrule\n")
                            else:
                                output_file.write("\n    \\\\[0.2cm] \n    \\midrule\n")
                        else:
                            output_file.write("\n")
                    else:
                        output_file.write(" & ")

        # write end of table
        output_file.write("  \\end{tabular}\n")
        output_file.write("  \\end{center}\n")
        output_file.write("  \\caption{Mean and standard deviation of the performance on validation data of the best performing models.}\n")
        output_file.write("\\end{table}\n\n")


def hyperparameters_table(
        datasets, 
        dataset_labels, 
        model_order, 
        model_labels, 
        hyperparameters,
        hyperparameter_labels,
        metric,
        metric_label,
        all_trials,
        best_trials,
        output_file,
        appendix,
        table_label,
        bayes):
    
        # take either all Bayes trials, or none
        if bayes:
            all_trials = all_trials.loc[all_trials['folder'].str.contains("-bo")]
        else:
            all_trials = all_trials.loc[~all_trials['folder'].str.contains("-bo")]

        # set printable values of some hyperparameters
        # floats with printable will not be treated as float, meaning it will show a delta
        print_values = {}
        print_values["train_type"] = {
                "negative_sampling": "NegSamp",
                "1vsAll": "1vsAll",
                "KvsAll": "KvsAll",
                }
        print_values["reciprocal"] = {
                0: "No",
                1: "Yes"
                }
        print_values["emb_regularize_p"] = {
                0.0: "None",
                1.0: "L1",
                2.0: "L2",
                3.0: "L3"
                }
        print_values["emb_regularize_weighted"] = {
                0: "No",
                1: "Yes",
                }
        print_values["transe_l_norm"] = {
                1: "L1",
                2: "L2",
                }
        print_values["transe_normalize_e"] = {
                -1.0: "No",
                1.0: "L1",
                2.0: "L2",
                }
        print_values["transe_normalize_r"] = {
                -1.0: "No",
                1.0: "L1",
                2.0: "L2",
                }
        print_values["emb_initialize"] = {
                "normal_": "Normal",
                "uniform_": "Unif.",
                "xavier_normal_": "XvNorm",
                "xavier_uniform_": "XvUnif"
                }
        print_values["train_loss"] = {
                "kl": "CE",
                "bce": "BCE",
                "margin_ranking": "MR"
                }
        
        # set rounding for floats (defaults to 2 if not determined here)
        # Not a pretty solution, couldn't quickly parameterize {:.2f}
        round_5 = ["train_lr", "emb_initialize_normal_std"]
        round_0 = ["num_negs_s", "num_negs_o"]
        scientific = ["emb_e_regularize_weight", "emb_r_regularize_weight"]

        # set compabitility between hyperparameters (determines when hyperparameter should be printed)
        # {attribute_1: (attribute_2, [list of values])}
        # Show attribute_1 iff value of attribute_2 is in list of values
        compatibility = {
                "num_negs_s":("train_type", ["negative_sampling"]),
                "num_negs_o":("train_type", ["negative_sampling"]),
                "label_smoothing":("train_type", ["KvsAll"]),
                "margin":("train_loss", ["margin_ranking"]),
                "transe_l_norm":("model", ["transe"]),
                "transe_normalize_e":("model", ["transe"]),
                "transe_normalize_r":("model", ["transe"]),
                "conve_projection_dropout":("model", ["conve"]),
                "conve_feature_map_dropout":("model", ["conve"]),
                "emb_initialize_normal_std":("emb_initialize", ["normal_"]),
                "emb_initialize_uniform_interval":("emb_initialize", ["uniform_"]),
                "emb_e_regularize_weight":("emb_regularize_p", [1, 2, 3]),
                "emb_r_regularize_weight":("emb_regularize_p", [1, 2, 3])
                }

        # set hyperparameters on the far left if table from appendix
        far_left_params = [
                "emb_e_dim", 
                "train_type", 
                "train_loss",
                "train_optimizer",
                "emb_regularize_p", 
                "emb_initialize", 
                ]

        # set hyperparameters that trigger a multicolumn row before them
        multicol_params = {
                "emb_e_dropout":"Dropout",
                "transe_normalize_e":"Embedding normalization (TransE)"              
                }

        # open output_file
        output_file = open(output_file, "w")

        # write table headers
        if appendix and not bayes:
            output_file.write("\\begin{sidewaystable}[h!]\n")
        else:
            output_file.write("\\begin{table}[t]\n")
        output_file.write("  \\begin{center}\n")
        output_file.write("  \\begin{tabular}{\n")
        if appendix:
            if not bayes:
                output_file.write("    l@{\hspace{0.2cm}}\n")
                output_file.write("    l@{\hspace{-0.2cm}}\n")
                output_file.write("    r@{\hspace{0.2cm}}\n")
                output_file.write("    c@{\hspace{0.2cm}}\n")
                output_file.write("    r@{\hspace{0.2cm}}\n")
                output_file.write("    c@{\hspace{0.2cm}}\n")
                output_file.write("    r@{\hspace{0.2cm}}\n")
                output_file.write("    c@{\hspace{0.2cm}}\n")
                output_file.write("    r@{\hspace{0.2cm}}\n")
                output_file.write("    c@{\hspace{0.2cm}}\n")
                output_file.write("    r@{\hspace{0.2cm}}\n")
                output_file.write("    c")
            else:
                output_file.write("    l@{\hspace{0.2cm}}\n")
                output_file.write("    l@{\hspace{-0.2cm}}\n")
                output_file.write("    r@{\hspace{0.1cm}}\n")
                output_file.write("    c@{\hspace{0.1cm}}\n")
                output_file.write("    r@{\hspace{0.1cm}}\n")
                output_file.write("    c@{\hspace{0.1cm}}\n")
                output_file.write("    r@{\hspace{0.1cm}}\n")
                output_file.write("    c@{\hspace{0.1cm}}\n")
                output_file.write("    r@{\hspace{0.1cm}}\n")
                output_file.write("    c@{\hspace{0.1cm}}\n")
                output_file.write("    r@{\hspace{0.1cm}}\n")
                output_file.write("    c")
        else:
            output_file.write("    l@{\hspace{0.2cm}}\n")
            output_file.write("    l@{\hspace{-0.05cm}}\n")
            output_file.write("    r@{\hspace{0.1cm}}\n")
            output_file.write("    c@{\hspace{0.2cm}}\n")
            output_file.write("    r@{\hspace{0.1cm}}\n")
            output_file.write("    c@{\hspace{0.2cm}}\n")
            output_file.write("    r@{\hspace{0.1cm}}\n")
            output_file.write("    c@{\hspace{0.2cm}}\n")
            output_file.write("    r@{\hspace{0.1cm}}\n")
            output_file.write("    c@{\hspace{0.2cm}}\n")
            output_file.write("    r@{\hspace{0.1cm}}\n")
            output_file.write("    c\n")
        output_file.write("}\n")
        output_file.write("    \\toprule\n")
        if not bayes:
            output_file.write("&                                       & \multicolumn{2}{c}{")
        else:
            output_file.write("&                                       & \multicolumn{1}{c}{")
        for model in model_order:
            if model == model_order[-1]:
                output_file.write(model_labels[model] + "}        \\\\ \\midrule\n")
            else:
                if not bayes:
                    output_file.write(model_labels[model] + "}        & \multicolumn{2}{c}{")
                else:
                    output_file.write(model_labels[model] + "} &       & \multicolumn{1}{c}{")

        # populate table
        model_means = {}
        for dataset in range(len(dataset_labels)):
            if not bayes:
                output_file.write("    \\multirow{" + str((len(hyperparameters) + len(multicol_params) + 1)) + "}*{\\begin{turn}{90}\\emph{" + dataset_labels[dataset] + "}\\end{turn}}\n")
            else:
                output_file.write("    \\multirow{" + str((len(hyperparameters))) + "}*{\\begin{turn}{90}\\emph{" + dataset_labels[dataset] + "}\\end{turn}}\n")

            # write mean MRR of each model
            if not bayes:
                mean = best_trials.groupby(["model", "dataset"], as_index=False).mean()
                count = all_trials.groupby(["model", "dataset"], as_index=False).count()

                output_file.write("& " + metric_label + "                      &               & \\emph{")
                for model in model_order:
                    entry_mean = mean.loc[(mean["model"] == model) & (mean["dataset"] == datasets[dataset])]
                    model_means[model] = round(entry_mean[metric].item()*100, 1)
                    output_file.write("{:.1f}".format(model_means[model]))
                    if model == model_order[-1]:
                        output_file.write("}   \\\\\n")
                    else:
                        output_file.write("}   &               & \emph{")

            # hyperparameter rows
            for i in range(len(hyperparameters)):
                hp = hyperparameters[i]
                if bayes:
                    show_delta = False
                else:
                    show_delta = True

                # insert multicolum row if necessary
                if appendix and hp in multicol_params and not bayes:
                    output_file.write("&\multicolumn{6}{l}{" + multicol_params[hp] + "} \\\\\n")

                # hyperparameter name
                if appendix and hp in far_left_params and not bayes:
                    output_file.write("& " + hyperparameter_labels[i] + "                        & ")
                else:
                    output_file.write("& $\\quad$" + hyperparameter_labels[i] + "                        & ")

                for model in model_order:
                    # get model trials
                    model_trials = all_trials.loc[
                            (all_trials['model'] == model) &
                            (all_trials['dataset'] == datasets[dataset])
                            ]

                    # value
                    max_entry = model_trials.loc[model_trials[metric] == model_trials[metric].max()]
                    value = max_entry[hp].item()

                    # check compatibility (whether it should be printed or not)
                    compatible = True
                    if hp in compatibility and max_entry[compatibility[hp][0]].item() not in compatibility[hp][1]:
                        compatible = False
                        show_delta = False
                    if compatible:
                        if isinstance(value, float) and hp not in print_values:
                            show_delta = False
                            if hp == "emb_initialize_uniform_interval":
                                output_file.write("[{:.2f}, {:.2f}]   & ".format(round(value, 2), round(value, 2) * -1))
                            else:
                                if hp in round_5:
                                    output_file.write("{:.5f}  & ".format(round(value, 5)))
                                elif hp in round_0:
                                    output_file.write("{:.0f}  & ".format(round(value, 0)))
                                elif hp in scientific:
                                    output_file.write(scientific_notation(value) + " & ")
                                else:
                                    output_file.write("{:.2f}  & ".format(round(value, 2)))

                        else:
                            if hp in print_values:
                                printable_value = print_values[hp][max_entry[hp].item()]
                            else:
                                printable_value = value
                            output_file.write("{}           & ".format(printable_value))
                        if show_delta:
                                output_file.write("\emph{")

                        # delta
                        if show_delta:
                            delta_trials = model_trials.loc[model_trials[hp] != value]
                            if len(delta_trials.index):
                                max_entry = delta_trials.loc[delta_trials[metric] == delta_trials[metric].max()]
                                delta = round((max_entry[metric].item() * 100) - model_means[model], 1)
                                output_file.write("({:.1f})".format(delta))
                            else:
                                output_file.write("--")
                            output_file.write("}")
                    else:
                        output_file.write("-- & ")

                    # close line
                    if model == model_order[-1]:
                        output_file.write(" \\\\")
                        if hp == hyperparameters[-1]:
                            if datasets[dataset] == datasets[-1]:
                                output_file.write("\n    \\bottomrule\n")
                            else:
                                output_file.write("\n    \\midrule\n")
                        else:
                            output_file.write("\n")
                    else:
                        output_file.write(" & ")

        # write end of table
        output_file.write("  \\end{tabular}\n")
        output_file.write("  \\end{center}\n")
        output_file.write("  \\caption{Insert caption here}\n")
        output_file.write("  \\label{" + table_label + "}\n")
        if appendix and not bayes:
            output_file.write("\\end{sidewaystable}\n")
        else:
            output_file.write("\\end{table}\n")


def scientific_notation(number):
    number_str = "{:.2E}".format(number).split("E")
    return r"$" + number_str[0] + r"^{" + number_str[1] + "}$"


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--all_trials', 
        type=str, 
        required=True, 
        help="csvs of all trials created with dump command, comma separated")
    parser.add_argument(
        '--best_trials', 
        type=str, 
        required=True, 
        help="csvs of best trials created with dump command, comma separated")
    args, _ = parser.parse_known_args()

    # load input CSVs
    csvs = []
    for input_file in args.all_trials.split(","):
        csvs.append(pandas.read_csv(input_file))
    all_trials = pandas.concat(csvs)
    csvs = []
    for input_file in args.best_trials.split(","):
        csvs.append(pandas.read_csv(input_file))
    best_trials = pandas.concat(csvs)

    # deal with empty string in emb_regularize_p
    all_trials['emb_regularize_p'] = all_trials['emb_regularize_p'].fillna(0)
    best_trials['emb_regularize_p'] = best_trials['emb_regularize_p'].fillna(0)

    # change negative dropout to zero
    all_trials['emb_e_dropout'] = all_trials['emb_e_dropout'].clip(lower=0)
    best_trials['emb_e_dropout'] = best_trials['emb_e_dropout'].clip(lower=0)
    all_trials['emb_r_dropout'] = all_trials['emb_r_dropout'].clip(lower=0)
    best_trials['emb_r_dropout'] = best_trials['emb_r_dropout'].clip(lower=0)

    # make sure it's only validation data
    all_trials = all_trials.loc[all_trials["split"] == "valid"]
    best_trials = best_trials.loc[best_trials["split"] == "valid"]

    # order and label for models
    model_order = ['rescal', 'transe', 'distmult', 'complex', 'conve']
    model_labels = {'rescal': 'RESCAL', 'transe': 'TransE', 'distmult': 'DistMult', 'complex': 'ComplEx', 'conve': 'ConvE'}

    # set datasets
    datasets = ["fb15k-237", "wnrr"]
    dataset_labels = ["FB15K-237", "WNRR"]

    # set 2 metrics and corresponding attributes in CSVs
    metrics = ["fwt_mrr", "fwt_hits@10"]
    metric_labels = ["MRR", "Hits@10"]

    # MEAN+STD TABLE IN APPENDIX
    mean_std_table(
        datasets, 
        dataset_labels, 
        metrics, 
        metric_labels, 
        model_order, 
        model_labels, 
        best_trials,
        "table_mean_std.tex")

    # HYPERPARAMETER TABLE IN MAIN PAPER
    # set datasets
    datasets = ["fb15k-237", "wnrr"]
    dataset_labels = ["FB15K-237", "WNRR"]

    # set hyperparameters and corresponding labels
    hyperparameters = [
            "emb_e_dim", 
            "train_batch_size", 
            "train_type", 
            "train_loss", 
            "train_optimizer", 
            "emb_initialize", 
            "emb_regularize_p", 
            "reciprocal"
            ]
    hyperparameter_labels = [
            "Emb. size", 
            "Batch size", 
            "Train type", 
            "Loss", 
            "Optimizer", 
            "Initializer", 
            "Regularizer", 
            "Reciprocal"
            ]

    hyperparameters_table(
        datasets, 
        dataset_labels, 
        model_order, 
        model_labels, 
        hyperparameters,
        hyperparameter_labels,
        "fwt_mrr",
        "Mean MRR",
        all_trials,
        best_trials,
        "table_hyperparameters_main_paper.tex",
        False,
        "tab:hyperparameters-best-models",
        False)

    # HYPERPARAMETER TABLES IN APPENDIX
    # set datasets
    datasets = ["fb15k-237"]
    dataset_labels = ["FB15K-237"]

    # set hyperparameters and corresponding labels
    hyperparameters = [
            "emb_e_dim", 
            "train_type", 
            "reciprocal",
            "num_negs_s",
            "num_negs_o",
            "label_smoothing",
            "train_loss", 
            "margin",
            "transe_l_norm",
            "train_optimizer", 
            "train_batch_size",
            "train_lr",
            "train_lr_scheduler_patience",
            "emb_regularize_p", 
            "emb_e_regularize_weight",
            "emb_r_regularize_weight",
            "emb_regularize_weighted",
            "transe_normalize_e",
            "transe_normalize_r",
            "emb_e_dropout",
            "emb_r_dropout",
            "conve_projection_dropout",
            "conve_feature_map_dropout",
            "emb_initialize", 
            "emb_initialize_normal_std",
            "emb_initialize_uniform_interval"   
            ]
    hyperparameter_labels = [
            "Embedding size", 
            "Training type", 
            "Reciprocal",
            "No. subject samples (NegSamp)",
            "No. object samples (NegSamp)",
            "Label Smoothing (KvsAll)",
            "Loss", 
            "Margin (MR)",
            "$L_p$-norm (TransE)",
            "Optimizer", 
            "Batch size", 
            "Learning rate",
            "Scheduler patience",
            "$L_p$ regularization", 
            "Entity emb. weight",
            "Relation emb. weight",
            "Frequency weighting",
            "Entity",
            "Relation",
            "Entity embedding",
            "Relation embedding",
            "Projection (ConvE)",
            "Feature map (ConvE)",
            "Embedding initialization", 
            "Std. deviation (Normal)",
            "Interval (Unif)"
            ]

    hyperparameters_table(
        datasets, 
        dataset_labels, 
        model_order, 
        model_labels, 
        hyperparameters,
        hyperparameter_labels,
        "fwt_mrr",
        "Mean MRR",
        all_trials,
        best_trials,
        "table_hyperparameters_fb15k-237.tex",
        True,
        "tab:hyperparameters-best-models-full-fb15k-237",
        False)

    # set datasets
    datasets = ["wnrr"]
    dataset_labels = ["WNRR"]

    hyperparameters_table(
        datasets, 
        dataset_labels, 
        model_order, 
        model_labels, 
        hyperparameters,
        hyperparameter_labels,
        "fwt_mrr",
        "Mean MRR",
        all_trials,
        best_trials,
        "table_hyperparameters_wnrr.tex",
        True,
        "tab:hyperparameters-best-models-full-wnrr",
        False)

    # BAYES HYPERPARAMETERS IN APPENDIX
    # set datasets
    datasets = ["fb15k-237", "wnrr"]
    dataset_labels = ["FB15K-237", "WNRR"]

    # set hyperparameters and corresponding labels
    hyperparameters = [
            "train_lr",
            "train_lr_scheduler_patience",
            "emb_e_regularize_weight",
            "emb_r_regularize_weight",
            "emb_e_dropout",
            "emb_r_dropout",
            "conve_projection_dropout",
            "conve_feature_map_dropout",
            ]
    hyperparameter_labels = [
            "Learning rate",
            "Scheduler patience",
            "Entity reg. weight",
            "Relation reg. weight",
            "Entity emb. dropout",
            "Relation emb. dropout",
            "Projection dropout (ConvE)",
            "Feature map dropout (ConvE)",
            ]

    hyperparameters_table(
        datasets, 
        dataset_labels, 
        model_order, 
        model_labels, 
        hyperparameters,
        hyperparameter_labels,
        "fwt_mrr",
        "Mean MRR",
        all_trials,
        best_trials,
        "table_hyperparameters_bayes.tex",
        True,
        "tab:hyperparameters-best-models-full-bayes",
        True)


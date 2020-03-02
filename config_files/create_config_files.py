import argparse
import yaml
import os


if __name__ == "__main__":
    datasets = ["fb15k-237", "wnrr"]
    train_types = ["1vsAll", "KvsAll", "negative_sampling"]
    template_filename = "templates_iclr2020.yaml"

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=False, help="config file of single model"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=False,
        help="prefix in config files of all input models",
    )
    args, _ = parser.parse_known_args()
    if (args.model and args.prefix) or (not args.model and not args.prefix):
        print("ERROR: must specify either model or prefix, not both")
        exit(1)

    # load input config files
    loaded_models = []
    if args.prefix:
        for file in os.listdir("."):
            filename = os.fsdecode(file)
            if filename.startswith(args.prefix) and filename.endswith(".yaml"):
                with open(filename, "r") as f:
                    loaded_models.append(yaml.load(f, Loader=yaml.SafeLoader))
        # create output folder
        root_folder = args.prefix
    else:
        with open(args.model, "r") as file:
            loaded_models.append(yaml.load(file, Loader=yaml.SafeLoader))
        # create output folder
        root_folder = os.path.splitext(args.model)[0]

    # create output folder
    os.mkdir(root_folder)

    # create config files
    for dataset in datasets:
        # set output folder for dataset
        os.mkdir(os.path.join(root_folder, dataset))
        for model_dict in loaded_models:
            model_name = model_dict["model"]
            # add reciprocal relations as an option if applicable
            if model_name in ["conve"]:
                models = "# model\n    - name: model\n      type: fixed\n      value: reciprocal_relations_model"
            else:
                models = "# model\n    - name: model\n      type: choice\n      values: [" + model_name + ", reciprocal_relations_model]"
                #models = "[" + model_name + ", reciprocal_relations_model]"
            model_specific_entries = []
            if "ax_search" in model_dict:
                model_specific_entries = model_dict["ax_search"]["parameters"]
            for train_type in train_types:
                # determine set of loss functions
                # no margin ranking in all train types
                if train_type == "KvsAll" or train_type == "1vsAll":
                    losses = ["kl", "bce"]
                # no bce in transe
                elif train_type == "negative_sampling" and model_name == "transe":
                    losses = ["kl", "margin_ranking"]
                else:
                    losses = ["kl", "bce", "margin_ranking"]
                for loss in losses:
                    # skip transe if loss = bce
                    if loss == "bce" and model_name == "transe":
                        continue

                    # skip transe unless train type is negative sampling
                    if (train_type == "1vsAll" or train_type == "KvsAll") and model_name == "transe":
                        continue

                    # set output folder for model-train_type-loss combo
                    output_folder = (
                        model_name
                        + "-"
                        + train_type
                        + "-"
                        + loss
                    )
                    first_line = dataset + "-" + output_folder
                    output_folder = os.path.join(root_folder, dataset, output_folder)
                    os.mkdir(output_folder)
                    # create config file from template
                    output_filename = "config.yaml"
                    output_file = open(
                        os.path.join(output_folder, output_filename), "w"
                    )
                    output_file.write("# " + first_line)
                    with open(template_filename, "r") as template_file:
                        for line in template_file:
                            # handle entries for specific train types
                            if "#train_type" in line and "___train_type___" not in line:
                                if train_type in line:
                                    output_file.write("\n" + line.strip("\n"))
                                else:
                                    continue
                            else:
                                # set model, dataset, train_type and losses
                                new_line = line.strip("\n").replace(
                                    "___dataset___", dataset
                                )
                                new_line = new_line.replace("___model___", model_name)
                                new_line = new_line.replace("___models___", models)
                                new_line = new_line.replace(
                                    "___train_type___", train_type
                                )
                                new_line = new_line.replace("___loss___", loss)
                                # write new line
                                output_file.write("\n" + new_line)

                        # append margin if applicable
                        if (
                            loss == "margin_ranking"
                            and train_type == "negative_sampling"
                        ):
                            output_file.write("\n    # margin\n")
                            output_file.write("    - name: train.loss_arg\n")
                            output_file.write("      type: range\n")
                            output_file.write("      bounds: [0.0, 10.0]\n")

                        # append model specific entries given by user
                        output_file.write("\n    # model-specific entries\n")
                        for entry in model_specific_entries:
                            for key in entry:
                                if key == "name":
                                    output_file.write(
                                        "    - "
                                        + str(key)
                                        + ": "
                                        + str(entry[key])
                                        + "\n"
                                    )
                                else:
                                    output_file.write(
                                        "      "
                                        + str(key)
                                        + ": "
                                        + str(entry[key])
                                        + "\n"
                                    )

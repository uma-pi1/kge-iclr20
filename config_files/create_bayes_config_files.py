import argparse
import yaml
import os
import pandas
import torch


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="prefix in folders to read",
    )
    parser.add_argument(
        "--dump_best_model",
        required=False,
        action="store_true",
        help="dumps best model settings",
    )
    args, _ = parser.parse_known_args()

    # load checkpoints for each folder
    loaded_searches = {}
    for folder in os.listdir("."):
        foldername = os.fsdecode(folder)
        if foldername.startswith(args.prefix) and os.path.isdir(folder):
            loaded_searches[foldername] = torch.load(foldername + "/checkpoint_00001.pt")
    
    # turn checkpoints into dataframes
    dataframes = []
    parameters = []
    for search in loaded_searches:
        # add parameters if new 
        for param in loaded_searches[search]['parameters'][0].keys():
            if param not in parameters:
                parameters.append(param)

        # create dataframe for current search
        data = [(
                search if r is not None else 0,
                r['mean_reciprocal_rank_filtered_with_test'] if r is not None else 0,
                r['hits_at_10_filtered_with_test'] if r is not None else 0,
                r['epoch'] if r is not None else 0,
                *[p[param] if param in p else 0 for param in parameters]) 
                for r,p in zip(loaded_searches[search]['results'], loaded_searches[search]['parameters'])]
        dataframes.append(pandas.DataFrame(data))

    # combine all data
    all_data = pandas.concat(dataframes)
    all_data.columns = ["folder", "mrr", "hits@10", "epoch"] + parameters
    
    # get best model
    best_model = all_data.sort_values("mrr").iloc[-1]

    # create folder for bayes search
    output_folder = best_model["folder"] + "-bo"
    os.mkdir(output_folder)

    # dump best model
    if args.dump_best_model:
        best_model.to_csv(
                os.path.join(output_folder, best_model["folder"] + "-best-model.csv"),
                header=True
                )
        print("Dumped best model settings to file {} in folder {}".format(
            best_model["folder"] + "-best-model.csv", best_model["folder"]
            )
        )

    # create config file for bayes search with best model settings
    output_file = open(
        os.path.join(output_folder, "config.yaml"), "w"
    )
    output_file.write("# " + output_folder)
    with open(os.path.join(best_model["folder"], "config.yaml"), "r") as best_model_config:
        prevLine = ""
        for line in best_model_config:
            new_line = line.strip("\n")
            # adjust trials
            if "num_sobol_trials" in new_line:
                new_line = new_line.replace("  num_sobol_trials: 30", "  num_sobol_trials: 10")
            # get current ax param
            if " name: " in new_line:
                ax_param = new_line.split(": ")[-1]
            # replace choice parameters
            if "choice" in new_line:
                next_line = next(best_model_config)
                if "values" in next_line:
                    # add fixed type
                    output_file.write("\n" + "      type: fixed")
                    # drop choice values 
                    next_line = next(best_model_config)
                    #while "   - " in next_line:
                    #    next_line = next(best_model_config)
                    # add fixed value
                    if isinstance(best_model[ax_param], str) and not best_model[ax_param]:
                        print("this is being treated as an empty string:", ax_param)
                        output_file.write("\n      value: ''")
                    else:
                        output_file.write("\n      value: " + str(best_model[ax_param]))
                    new_line = next_line.strip("\n")
                    if " name: " in new_line:
                        ax_param = new_line.split(": ")[-1]
            # write line and update previous line
            output_file.write("\n" + new_line)
            prevLine = line.strip('\n')

    # done        
    print("Created file {} in folder {}".format("config.yaml", output_folder))


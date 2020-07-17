import argparse
import pandas
import torch
import math 
import os


from shutil import copyfile
from copy import deepcopy

if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="location of checkpoint file of bayes search",
    )
    parser.add_argument(
        "--best_model",
        type=str,
        required=True,
        help="location of dumped best model",
    )
    args, _ = parser.parse_known_args()

    # load checkpoint file
    cp = torch.load(args.checkpoint)

    # load dumped best model
    bm = pandas.read_csv(args.best_model, index_col=0)
    bm = pandas.DataFrame(bm).T
    
    # backup original checkpoint file if not done already
    if not os.path.isfile(args.checkpoint + ".bak"):
        copyfile(args.checkpoint, args.checkpoint + ".bak")
        print("Original checkpoint file stored in {}.bak".format(args.checkpoint))

    # add best model to trials
    cp["parameters"].append(deepcopy(cp["parameters"][-1]))
    cp["results"].append(None)
    for key in cp["parameters"][-1]:
        param_type = type(cp["parameters"][-1][key])

        if isinstance(bm[key][0], float) and math.isnan(bm[key][0]):
            cp["parameters"][-1][key] = ''
        else:
            if param_type == int:
                cp["parameters"][-1][key] = param_type(float(bm[key][0]))
            elif param_type == bool:
                if bm[key][0].lower() == 'false':
                    cp["parameters"][-1][key] = False
                else:
                    cp["parameters"][-1][key] = True
            else:
                cp["parameters"][-1][key] = param_type(bm[key][0])

    torch.save(cp, args.checkpoint)

    # done
    print("Added best model settings from {} as a trial in {}".format(
        args.best_model, args.checkpoint
        )
    )


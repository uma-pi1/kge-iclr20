import argparse
import json
import pickle
import glob
import matplotlib as plt
import seaborn
import pandas
import numpy
from tqdm import tqdm

from collections import defaultdict, OrderedDict


def collect(base_path, dataset, ):

    for i, trace_file_name in enumerate(
            sorted(glob.glob('{}/{}/*/*/trace.yaml'.format(base_path, dataset)))
    ):

        model_ds = trace_file_name.split('/')[-3]

        print('{}###{}'.format(dataset, model_ds))

        with open(trace_file_name, 'r') as f:
            for line in tqdm(f.readlines()):
                entry = dict(filter(lambda x: len(x) == 2, [kv.split(': ') for kv in line[1:-1].split(', ')]))
                if 'folder' in entry:
                    if len(entry['folder']) > 1:
                        folder = entry['folder']
                    else:
                        print(line)
                if entry['job'] == 'eval':
                    if 'mean_reciprocal_rank_filtered_with_test' in entry:
                        try:
                            collected_results['{}###{}'.format(dataset, model_ds)][folder][int(entry['epoch'])] = float(entry['mean_reciprocal_rank_filtered_with_test'])
                        except:
                            print(model_ds, folder)
                            print(line)

    return collected_results


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True, help="csvs created with dump command, comma separated")
    parser.add_argument('--output_folder', type=str, required=True, help="name of output folder")
    args, _ = parser.parse_known_args()

    collected_results = defaultdict(lambda: defaultdict(lambda: dict()))

    collected_results = collect(args.base_path, 'fb15k-237', collected_results)
    collected_results = collect(args.base_path, 'wnrr', collected_results)

    first_epoch_same_as_last_epoch = list()

    best_collected_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

    # collect the best result in each epoch for each mode
    for model_ds in collected_results.keys():
        dataset, model_traintype_loss = model_ds.split('###')
        if len(model_traintype_loss.split('-')) == 4 or 'bo-best' in model_traintype_loss:
            continue
        model, traintype, loss = model_traintype_loss.split('-')
        for trial in collected_results[model_ds].keys():
            for (epoch, mrr) in collected_results[model_ds][trial].items():
                best_collected_results[dataset][model][epoch] = max(best_collected_results[dataset][model][epoch], mrr)

    sky_collected_results = defaultdict(lambda: defaultdict(list))

    # now compute the skyline
    for dataset in best_collected_results.keys():
        for model in best_collected_results[dataset].keys():
            sorted_best_collected_results = sorted(best_collected_results[dataset][model].items(), key=lambda kv: kv[0])
            skyline_list = [sorted_best_collected_results[0]]
            best = 0
            for i in range(1, len(sorted_best_collected_results)):
                best = max(sorted_best_collected_results[i][1], best)
                skyline_list.append(
                    (sorted_best_collected_results[i][0], best)
                )
            sky_collected_results[dataset][model] = skyline_list

    fig, axes = plt.subplots(nrows=1, ncols=2)

    trans = OrderedDict({
        'rescal': 'RESCAL',
        'transe': 'TransE',
        'distmult': 'DistMult',
        'complex': 'ComplEx',
        'conve': 'ConvE',
        'fb15k-237': 'FB15k 237',
        'wnrr': 'WNRR',
        'fb15k': 'FB15k',
        'wn18': 'WN18',
    })

    ylim = {
        'wnrr': (0.19, 0.48),
        'fb15k-237': (0.19, 0.48),
    }

    for dataset, ax in zip(sky_collected_results.keys(), axes):
        ax.set_ylabel("MRR")
        for model in trans.keys():
            if model in sky_collected_results[dataset]:
                df = pandas.DataFrame(sky_collected_results[dataset][model], columns=['epoch', trans[model]])
                df.plot.line(x='epoch', y=trans[model], ax=ax, ylim=ylim[dataset], figsize=(12, 4), title=trans[dataset])

    fig.savefig("{}/best_mrr_per_epoch.pdf".format(args.output_folder), bbox_inches='tight', dpi=300)
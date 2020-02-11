import argparse
import json
import pickle
import glob
import matplotlib.pyplot as plt
import seaborn
import pandas
import numpy
from tqdm import tqdm

from collections import defaultdict, OrderedDict


def collect(base_path, dataset, collected_results, ):

    for i, trace_file_name in enumerate(
            sorted(glob.glob('{}/{}/*/*/trace.yaml'.format(base_path, dataset)))
    ):

        model_ds = trace_file_name.split('/')[-3]

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
    best_collected_trial = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

    for model_ds in collected_results.keys():
        print(model_ds)
        dataset, model_traintype_loss = model_ds.split('###')
        if len(model_traintype_loss.split('-')) == 4 or 'bo-best' in model_traintype_loss:
            continue
        #    print(model_traintype_loss)
        model, traintype, loss = model_traintype_loss.split('-')
        for trial in collected_results[model_ds].keys():
            # sorted_results_model_ds_trial = sorted(collected_results[model_ds][trial].items(), key=lambda kv: kv[1])
            for (epoch, mrr) in collected_results[model_ds][trial].items():
                if mrr > best_collected_results[dataset][model][epoch]:
                    best_collected_results[dataset][model][epoch] = mrr
                    best_collected_trial[dataset][model][epoch] = trial

    sky_collected_results = defaultdict(lambda: defaultdict(list))
    sky_collected_results_trial = defaultdict(lambda: defaultdict(list))

    # print(best_collected_trial)

    for dataset in best_collected_results.keys():
        for model in best_collected_results[dataset].keys():
            sorted_best_collected_results = sorted(best_collected_results[dataset][model].items(), key=lambda kv: kv[0])
            skyline_list = [sorted_best_collected_results[0]]
            trials_ticks_in_skyline = []
            best = 0
            best_trial = '-'
            for i in range(1, len(sorted_best_collected_results)):
                epoch = sorted_best_collected_results[i][0]
                if sorted_best_collected_results[i][1] > best:
                    best = sorted_best_collected_results[i][1]
                    if best_trial != best_collected_trial[dataset][model][epoch]:
                        best_trial = best_collected_trial[dataset][model][epoch]
                        trials_ticks_in_skyline.append((epoch, best))
                skyline_list.append((epoch, best))
            sky_collected_results[dataset][model] = skyline_list
            sky_collected_results_trial[dataset][model] = trials_ticks_in_skyline

    fig, axes = plt.subplots(nrows=1, ncols=2)

    trans = OrderedDict({
        'rescal': 'RESCAL',
        'transe': 'TransE',
        'distmult': 'DistMult',
        'complex': 'ComplEx',
        'conve': 'ConvE',
        'fb15k-237': 'FB15K-237',
        'wnrr': 'WNRR',
        'fb15k': 'FB15k',
        'wn18': 'WN18',
    })

    ylim = {
        'wnrr': (19, 48),
        'fb15k-237': (19, 48),
    }

    font_size = 7.5
    for dataset, ax in zip(sky_collected_results.keys(), axes):
        for model in trans.keys():
            if dataset not in ['wnrr']:
                ax.set_ylabel("Validation MRR", fontsize=font_size)
            else:
                ax.set_ylabel(" ", fontsize=font_size)
            ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
            ax.title.set_size(font_size)
            ax.tick_params(labelsize=font_size)
            if model in sky_collected_results[dataset]:
                data = [(x, y * 100) for x, y in sky_collected_results[dataset][model]]
                data_ticks = [(x, y * 100) for x, y in sky_collected_results_trial[dataset][model]]
                df = pandas.DataFrame(data, columns=['Epoch', trans[model]])
                ax = df.plot.line(x='Epoch', y=trans[model], ax=ax, ylim=ylim[dataset], figsize=(6.4, 2.4), title=trans[dataset], linewidth=1)
                color = ax.get_lines()[-1].get_color()
                df = pandas.DataFrame(data_ticks, columns=['Epoch', trans[model]])
                df.plot.scatter(x='Epoch', y=trans[model], ax=ax, ylim=ylim[dataset], figsize=(6.4, 2.4), title=trans[dataset], marker='o', s=8, c=color)
                ax.legend(prop={'size': font_size})
        if dataset not in ['wnrr']:
            ax.get_legend().remove()

    fig.savefig("{}/best_mrr_per_epoch.pdf".format(args.output_folder), bbox_inches='tight', dpi=300)


#!/bin/bash

dataset=${PWD##*/}
for d in *best*/ ; do
	cd $d
	IFS='-'	read -ra ADDR <<< "$d"
	model=${ADDR[0]}
	# commented out because files already there
	#python /work/druffine/kge/kge/cli.py dump trace . --keysfile /work/druffine/kge-configs/search/scripts/iclr2020_keys.conf > trace_dump.csv
	trial_folder=$(python ../get_best_trial.py)
	cd $trial_folder
	echo 'Dumping config of best '$model' for '$dataset'...'
	output_file=$dataset'-'$model'-config-checkpoint_best.yaml'
	/work/druffine/kge/kge/cli.py dump config checkpoint_best.pt --exclude search ax_search dataset.files > $output_file
	mv $output_file ../../$output_file
	cd ../..
done


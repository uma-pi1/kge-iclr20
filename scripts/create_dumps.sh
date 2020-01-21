#!/bin/sh

for d in */ ; do
	cd $d
	python /work/druffine/kge/kge.py dump trace . --keysfile /work/druffine/kge-configs/search/scripts/iclr2020_keys.conf > trace_dump.csv
	cd ..
done

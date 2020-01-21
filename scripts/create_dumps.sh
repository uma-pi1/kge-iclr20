#!/bin/sh

kge_location=$1
keys=$2

for d in */ ; do
	cd $d
	python $kge_location dump trace . --keysfile $keys > trace_dump.csv
	cd ..
done

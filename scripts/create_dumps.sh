#!/bin/sh

keys=$1

for d in */ ; do
	cd $d
	kge dump trace . --keysfile $keys > trace_dump.csv
	cd ..
done

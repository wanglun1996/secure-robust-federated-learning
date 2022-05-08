#!/bin/bash

COUNTER=0

for DATASET in "MNIST" "Fashion-MNIST"
do
	for ATTACK in "modelpoisoning" "backdoor" "noattack" "trimmedmean" "krum"
	do
		for AGG in "krum" "median" "bulyankrum" "bulyanmedian" "bulyantrimmedmean" # "mom_filterl2" "mom_ex_noregret" "filterl2" "ex_noregret" # "filterL2" "bulyankrum" "bulyantrimmedmean" "average" "krum" "trimmedmean" "ex_noregret"
		do
			DEVICE=$((COUNTER%4+4))
			COMMAND="conda init; python simulate.py --dataset ${DATASET} --device ${DEVICE} --attack ${ATTACK} --agg ${AGG} > log/${ATTACK}_${AGG}_${DATASET}.log;"
			echo $COMMAND
			screen -dm bash -c "conda init; python simulate.py --dataset ${DATASET} --device ${DEVICE} --attack ${ATTACK} --agg ${AGG} > log/${ATTACK}_${AGG}_${DATASET}.log;"
			COUNTER=$((COUNTER+1))
		done
	done
done

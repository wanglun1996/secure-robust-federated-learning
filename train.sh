#!/bin/bash

COUNTER=0

mkdir log

for DATASET in "MNIST" "Fashion-MNIST"
do
	for ATTACK in "backdoor" "krum" "modelpoisoning" "noattack" "trimmedmean"
	do
		for AGG in "average" "bulyankrum" "bulyanmedian" "bulyantrimmedmean" "ex_noregret" "filterl2" "krum" "median" "mom_ex_noregret" "mom_filterl2" "trimmedmean"
		do
			DEVICE=$((COUNTER%8))
			COMMAND="conda init; python src/simulate.py --dataset ${DATASET} --device ${DEVICE} --attack ${ATTACK} --agg ${AGG} > log/${ATTACK}_${AGG}_${DATASET}.log;"
			echo $COMMAND
			screen -dmS ${ATTACK}_${AGG}_${DATASET} bash -c "conda init; python src/simulate.py --dataset ${DATASET} --device ${DEVICE} --attack ${ATTACK} --agg ${AGG} > log/${ATTACK}_${AGG}_${DATASET}.log;"
			COUNTER=$((COUNTER+1))
		done
	done
done

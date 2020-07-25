#!/bin/bash


python run.py --transformer="/data/ficstamas/sense_bert/sensebert-base-uncased" \
 --in_file="/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml" --name="semcor" --gpu=0


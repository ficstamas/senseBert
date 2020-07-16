#!/bin/bash

transformers=("sensebert-base-uncased" "sensebert-large-uncased")
semcor="/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
ALL="/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"

for transformer in ${transformers[*]}
do
  python run.py --transformer="$transformer" --in_file="$semcor" --name="semcor" --gpu=0
  python run.py --transformer="$transformer" --in_file="$ALL" --name="semcor" --gpu=0
done

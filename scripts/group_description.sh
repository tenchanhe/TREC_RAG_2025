#!/usr/bin/env bash

python3 -m src.kg.group_description_by_dense
python3 src/retireval/group_description.py
python3 src/evaluation/cal_trec.py --input_path runs/retrieval/group_description/et_dense.txt
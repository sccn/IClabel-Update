#!/bin/bash

# update labels
rm -rf python/data/cache
scp labeling:~/iclabel/python/results/ICLabels_onlyluca.pkl python/data/labels/
scp labeling:~/iclabel/python/results/ICLabels_expert.pkl python/data/labels/

# select and train updated ICLabel model
cd python
python icl_final_cv_test.py
cd ../

# transfer model to MATLAB (matconvnet)
cd matlab
matlab -nodisplay -nosplash -nodesktop -r "run('construct_ICL.m');"
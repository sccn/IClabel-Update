# IClabel-Update
An automated pipeline to periodically update the ICLabel EEGLAB plugin from the current [ICLabel website](iclabel.ucsd.edu) databates.

### Pipeline
1.  The iclabel website runs crowd labeling algorithms on a nightly basis and provides the results as downloadable files.
2.  Those files are downloaded with python and used to train tensorflow models in a cross-validation scheme, output model weights.
3.  The new model weights are incorporated into the ICLabel EEGLAB plugin, the version number is incremented, and the zipped package update is pushed to the plugin manager.

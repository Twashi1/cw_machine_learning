import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio

# Goal: Logistic Regression
#   - Use multiple modalities to optimise
#   - Use grid search to optimise hyperparameters
#       one such hyperparameter: threshold
#       could also consider cross-validation, assuming would have its own set of hyperparameters
#   - then do some basic performance testing, compare to sklearn LogisticRegression
#   - then test with varying train/test splits

#   - (filter the outliers we saw in each modality)

# In the report
#   - analyse the available baseline models, and justify why logistic regression is best for this case
#       - lots of visualisations for the above
#       - bonus if we find some issues in class distribution
#       - talk about filtering outliers?
#       - explain simply how the outliers influence the model's performance, maybe feature scaling is relevant
#   - clearly explain the meaning of f1-score, accuracy, precision, and recall
#   - talk about how the outliers being eliminated may help model's performance, or maybe had no influence
#   - any other pattern in class distribution that could be accounted for??
#   - talk very basic about how train/test split helps paradigm?
#       - maybe use new paradigm (few-shot, zero-shot, etc.) and discuss value of said paradigm (easier?)
#       - how ato adjust model to the new paradigm (likely few-shot, zero-shot, federated, etc.)
#       - evaluate model with new paradigm

# Prepare dataset

# load data
data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
sbj = 'sub-10'
image_model = 'pytorch/cornet_s'
text_model = 'CLIPText'
roi = '17channels'
brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)
image_dir_seen = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', image_model, sbj)
image_dir_unseen = os.path.join(data_dir_root, 'visual_feature/ThingsTest', image_model, sbj)
text_dir_seen = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
text_dir_unseen = os.path.join(data_dir_root, 'textual_feature/ThingsTest/text', text_model, sbj)

brain_seen = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['data'].astype('double') * 2.0
brain_seen = brain_seen[:,:,27:60] # 70ms-400ms
brain_seen = np.reshape(brain_seen, (brain_seen.shape[0], -1))
image_seen = sio.loadmat(os.path.join(image_dir_seen, 'feat_pca_train.mat'))['data'].astype('double')*50.0
text_seen = sio.loadmat(os.path.join(text_dir_seen, 'text_feat_train.mat'))['data'].astype('double')*2.0
label_seen = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['class_idx'].T.astype('int')
image_seen = image_seen[:,0:100]

brain_unseen = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))['data'].astype('double')*2.0
brain_unseen = brain_unseen[:, :, 27:60]
brain_unseen = np.reshape(brain_unseen, (brain_unseen.shape[0], -1))
image_unseen = sio.loadmat(os.path.join(image_dir_unseen, 'feat_pca_test.mat'))['data'].astype('double')*50.0
text_unseen = sio.loadmat(os.path.join(text_dir_unseen, 'text_feat_test.mat'))['data'].astype('double')*2.0
label_unseen = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))['class_idx'].T.astype('int')
image_unseen = image_unseen[:, 0:100]
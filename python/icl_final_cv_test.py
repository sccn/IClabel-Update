from tfGAN_indvBN import *
from tfMLP import *
from icldata import ICLabelDataset  # available from https://github.com/lucapton/ICLabel-Dataset
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn import metrics
from os.path import isdir, join
import shutil
import pandas as pd
from scipy import interp
from scipy.io import savemat
from matplotlib import pyplot as plt
import itertools


def ndarr2latex(arr, caption=None, label=None, row_names=None, col_names=None):
    n_row, n_col = arr.shape
    latex = [
        '\\begin{table}',
        '\\centering',
        '\\begin{tabular}{|' + 'c|' * (n_col + (row_names is not None)) + '}',
        '\\hline'
    ]
    if col_names is not None:
        latex.append(' & ' + ' & '.join(col_names) + ' \\\\')
        latex.append('\\hline')

    for it in range(n_row):
        strarr = ['{:.1f}'.format(x) for x in arr[it]]
        if row_names is not None:
            latex.append(' & '.join(row_names[it:it+1] + strarr) + ' \\\\')
        else:
            latex.append(' & '.join(strarr) + ' \\\\')
        latex.append('\\hline')

    latex.append('\\end{tabular}')

    if caption is not None:
        latex.append('\\caption{' + caption + '}')

    if label is not None:
        latex.append('\\label{' + label + '}')

    latex.append('\\end{table}')

    return latex


def make_cm_strong_and(label, pred):
    assert label.shape == pred.shape, 'label and prediciton must have same shape'
    return np.nansum(np.maximum(label[:, :,  np.newaxis] + pred[:, np.newaxis] - 1, 0), 0)


def make_cm_weak_and(label, pred):
    assert label.shape == pred.shape, 'label and prediciton must have same shape'
    return np.nansum(np.minimum(label[:, :,  np.newaxis], pred[:, np.newaxis]), 0)


def make_cm_prod(label, pred):
    assert label.shape == pred.shape, 'label and prediciton must have same shape'
    return np.nansum(label[:, :,  np.newaxis] * pred[:, np.newaxis], 0)


def make_cm_all(label, pred):
    # get raw soft confusion matices (cm)
    strong = make_cm_strong_and(label, pred)
    weak = make_cm_weak_and(label, pred)
    prod = make_cm_prod(label, pred)
    # combine strong and weak AND cms into optimistic and pessimistic cms
    cm_pes = weak.copy()
    np.fill_diagonal(cm_pes, np.diag(strong))
    cm_opt = strong.copy()
    np.fill_diagonal(cm_opt, np.diag(weak))

    return cm_pes, prod, cm_opt


def perf_soft(label, pred):
    n_cls = label.shape[1]
    cm_pes, cm_prod, cm_opt = make_cm_all(label, pred)

    ce = -np.nansum(label * np.log(pred), 1).mean()

    # get perf stats
    acc, pre, rec, spe = np.zeros(3), np.zeros((3, n_cls)), np.zeros((3, n_cls)), np.zeros((3, n_cls))
    for it, cm in enumerate((cm_pes, cm_prod, cm_opt)):
        acc[it] = np.diag(cm).sum() / cm.sum()
        pre[it, :] = np.diag(cm) / cm.sum(0)  # precision / PPV
        rec[it, :] = np.diag(cm) / cm.sum(1)  # recall / sensitivity / TPR
        spe[it, :] = (np.diag(cm).sum() - np.diag(cm)) / (cm.sum() - cm.sum(1))  # Specifity / 1 - FPR

    return acc, pre, rec, spe, ce


def perf_hard(labels, pred):
    # remove nans
    ind_keep = np.logical_not(np.isnan(pred).any(1))
    labels = labels[ind_keep]
    pred = pred[ind_keep]

    # get argmax
    label_argmax = labels.argmax(1)
    pred_argmax = pred.argmax(1)

    # get perf stats
    ce = -(labels * np.log(pred)).sum(1).mean()
    acc = metrics.accuracy_score(label_argmax, pred_argmax)
    pre = metrics.precision_score(label_argmax, pred_argmax, average=None)
    rec = metrics.recall_score(label_argmax, pred_argmax, average=None)
    auc = np.array([])

    # roc and prc
    roc = []
    prc = []
    thresh = np.zeros((1, pred.shape[1]))
    spacing = np.linspace(0, 1, 101)
    for it in range(pred.shape[1]):
        auc = np.append(auc, metrics.roc_auc_score(label_argmax == it, pred[:, it]))
        temp_roc = metrics.roc_curve(label_argmax == it, pred[:, it])
        roc.append([interp(spacing, temp_roc[2][::-1], x[::-1]) for x in temp_roc])
        temp_prc = metrics.precision_recall_curve(label_argmax == it, pred[:, it])
        temp_prc = temp_prc[:2] + (np.concatenate((temp_prc[2], [1])),)
        thresh[0, it] = temp_prc[2][np.argmax(f_beta_prc(temp_prc[0], temp_prc[1], 1))]
        prc.append([interp(spacing, temp_prc[2], x) for x in temp_prc])

    micro_pre = metrics.precision_score(label_argmax, pred_argmax, average='micro')
    micro_rec = metrics.recall_score(label_argmax, pred_argmax, average='micro')
    macro_pre = pre.mean()
    macro_rec = rec.mean()
    macro_auc = auc.mean()

    return thresh, ce, acc, pre, rec, auc, roc, prc, micro_pre, micro_rec, macro_pre, macro_rec, macro_auc


def soft_perf_plot(vals, classes=None):
    plt.figure()
    marker = itertools.cycle((',', '+', '.', 'o', '*'))
    for it, vals in enumerate(vals):
        plt.plot(vals, linestyle='', marker=marker.next())
        if classes is not None:
            plt.xticks(range(n_cls), labels=classes)


def soft_perf_plot2(vals, labels=None, new_fig=True):
    n_cls = vals.shape[1]
    if new_fig:
        plt.figure()
    plt.errorbar(range(n_cls), vals[1], yerr=np.abs(vals[1:2] - vals[[0, 2], :]))
    plt.xlim((0.1, n_cls + 0.1))
    plt.ylim((0, 1))
    if labels is not None:
        plt.xticks(range(n_cls), labels, rotation=20)


def reduce_labels(labels, n_cls):
    if n_cls == 2:
        labels = np.concatenate((labels[:, 0:1], labels[:, 1:].sum(1, keepdims=True)), 1)
    elif n_cls == 3:
        labels = np.concatenate((labels[:, 0:1], labels[:, 2:3], labels[:, [1, 3, 4, 5, 6]].sum(1, keepdims=True)), 1)
    elif n_cls == 5:
        labels = np.concatenate((labels[:, :4], labels[:, 4:].sum(1, keepdims=True)), 1)
    elif n_cls == 7:
        pass
    else:
        raise ValueError('n_cls must be 2, 3, or 5')
    return labels


def acc_cr(fpr, tpr, class_ratio=1):
    return (tpr + class_ratio * (1 - fpr)) / (1 + class_ratio)


def f_beta_roc(fpr, tpr, beta=1):
    return 2 * tpr / (tpr + beta * fpr + 1)


def f_beta_prc(precision, recall, beta):
    return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

n_folds = 10

icl_archs = [WeightedConvMANN, ConvMANN, AltConvMSSGAN]
ilc_methods = [x.name + ' w/ acor'*y for x, y in itertools.product(icl_archs, range(2))]
other_archs = ICLabelDataset(datapath='data/').load_classifications(2, np.array([[1, 1]])).keys()
cls_map = {x: y for x, y in zip(ilc_methods + other_archs, range(len(ilc_methods + other_archs)))}
cls_imap = {y: x for x, y in cls_map.iteritems()}
cls_imap = [cls_imap[x] for x in range(len(cls_map))]

cols = ('n_cls', 'arch', 'fold',
        'cross_entropy', 'accuracy',
        'micro_precision', 'micro_recall',
        'macro_precision', 'macro_recall', 'macro_auc',
        'precision', 'recall', 'auc',
        'thresh', 'roc_fpr', 'roc_tpr', 'roc_thr', 'prc_pre', 'prc_rec', 'prc_thr',
        'soft_acccuracy_pessimistic', 'soft_acccuracy_expected', 'soft_acccuracy_optimistic',
        'soft_precision_pessimistic', 'soft_precision_expected', 'soft_precision_optimistic',
        'soft_recall_pessimistic', 'soft_recall_expected', 'soft_recall_optimistic',
        'soft_specificity_pessimistic', 'soft_specificity_expected', 'soft_specificity_optimistic')
scores = pd.DataFrame(columns=cols)
raw = {x: [[]] * n_folds for x in cls_imap}
raw.update({'label': [[]] * n_folds})
# train and extract performance statistics
for labels in ('all',):

    # load data
    icl = ICLabelDataset(label_type=labels, datapath='data/')
    icl_data = icl.load_semi_supervised()
    icl_data_val_labels = np.concatenate((icl_data[1][1][0], icl_data[3][1][0]), axis=0)
    icl_data_val_ilrlabels = np.concatenate((icl_data[1][1][1], icl_data[3][1][1]), axis=0)
    icl_data_val_ilrlabelscov = np.concatenate((icl_data[1][2][1], icl_data[3][2][1]), axis=0)

    # process topo maps
    topo_data = list()
    for it in range(4):
        temp = 0.99 * icl_data[it][0]['topo'] / np.abs(icl_data[it][0]['topo']).max(1, keepdims=True)
        topo_data.append(icl.pad_topo(temp).astype(np.float32).reshape(-1, 32, 32, 1))

    # generate mask
    mask = np.setdiff1d(np.arange(1024), icl.topo_ind)

    # K-fold
    kfold = StratifiedKFold(n_splits=n_folds)
    ind_fold = 0

    for ind_train_l, ind_test in kfold.split(icl_data_val_labels, icl_data_val_labels.argmax(1)):

        # create validation set
        sss = StratifiedShuffleSplit(1, len(ind_test))
        sss_gen = sss.split(icl_data_val_labels[ind_train_l], icl_data_val_labels[ind_train_l].argmax(1))
        ind_train_l_tr, ind_train_l_val = sss_gen.next()

        for use_autocorr in (False, True):

            # rescale features
            if use_autocorr:
                input_data = [[topo_data[x],
                               0.99 * icl_data[x][0]['psd'],
                               0.99 * icl_data[x][0]['autocorr'],
                               ] for x in range(4)]
            else:
                input_data = [[topo_data[x],
                               0.99 * icl_data[x][0]['psd'],
                               ] for x in range(4)]

            # create data fold
            temp = [np.concatenate((x, y), axis=0) for x, y in zip(input_data[1], input_data[3])]
            input_data[1] = [x[ind_train_l] for x in temp]                  # labeled train
            input_data[2] = [x[ind_train_l[ind_train_l_tr]] for x in temp]  # labeled train fold
            input_data[3] = [x[ind_train_l[ind_train_l_val]] for x in temp] # labeled validation fold
            input_data.append([x[ind_test] for x in temp])                  # test data
            test_ids = np.concatenate((icl_data[1][0]['ids'], icl_data[3][0]['ids']), axis=0)[ind_test]

            # create label fold
            train_labels = icl_data_val_labels[ind_train_l]
            train_labels_tr = icl_data_val_labels[ind_train_l[ind_train_l_tr]]
            train_labels_val = icl_data_val_labels[ind_train_l[ind_train_l_val]]
            test_labels = icl_data_val_labels[ind_test]

            train_ilrlabels = icl_data_val_ilrlabels[ind_train_l]
            train_ilrlabels_tr = icl_data_val_ilrlabels[ind_train_l[ind_train_l_tr]]
            train_ilrlabels_val = icl_data_val_ilrlabels[ind_train_l[ind_train_l_val]]
            test_ilrlabels = icl_data_val_ilrlabels[ind_test]

            train_ilrlabelscov = icl_data_val_ilrlabelscov[ind_train_l]
            train_ilrlabelscov_tr = icl_data_val_ilrlabelscov[ind_train_l[ind_train_l_tr]]
            train_ilrlabelscov_val = icl_data_val_ilrlabelscov[ind_train_l[ind_train_l_val]]
            test_ilrlabelscov = icl_data_val_ilrlabelscov[ind_test]

            # augment dataset by negating and/or horizontally flipping topo maps
            for it in range(5):
                input_data[it][0] = np.concatenate((input_data[it][0],
                                                    -input_data[it][0],
                                                    np.flip(input_data[it][0], 2),
                                                    -np.flip(input_data[it][0], 2)))
                for it2 in range(1, len(input_data[it])):
                    input_data[it][it2] = np.tile(input_data[it][it2], (4, 1))
            try:
                train_labels = np.tile(train_labels, (4, 1))
                train_labels_tr = np.tile(train_labels_tr, (4, 1))
                train_labels_val = np.tile(train_labels_val, (4, 1))
                test_labels = np.tile(test_labels, (4, 1))
                # ilr labels
                train_ilrlabels = np.tile(train_ilrlabels, (4, 1))
                train_ilrlabels_tr = np.tile(train_ilrlabels_tr, (4, 1))
                train_ilrlabels_val = np.tile(train_ilrlabels_val, (4, 1))
                test_ilrlabels = np.tile(test_ilrlabels, (4, 1))
                # ilr labels cov
                train_ilrlabelscov = np.tile(train_ilrlabelscov, (4, 1, 1))
                train_ilrlabelscov_tr = np.tile(train_ilrlabelscov_tr, (4, 1, 1))
                train_ilrlabelscov_val = np.tile(train_ilrlabelscov_val, (4, 1, 1))
                test_ilrlabelscov = np.tile(test_ilrlabelscov, (4, 1, 1))
            except ValueError:
                train_labels = 4 * train_labels
                train_labels_tr = 4 * train_labels_tr
                train_labels_val = 4 * train_labels_val
                test_labels = 4 * test_labels
                # ilr labels
                train_ilrlabels = 4 * train_ilrlabels
                train_ilrlabels_tr = 4 * train_ilrlabels_tr
                train_ilrlabels_val = 4 * train_ilrlabels_val
                test_ilrlabels = 4 * test_ilrlabels
                # ilr labels cov
                train_ilrlabelscov = 4 * train_ilrlabelscov
                train_ilrlabelscov_tr = 4 * train_ilrlabelscov_tr
                train_ilrlabelscov_val = 4 * train_ilrlabelscov_val
                test_ilrlabelscov = 4 * test_ilrlabelscov

            test_ids = np.tile(test_ids, (4, 1))

            # describe features and name
            additional_features = OrderedDict([('psd_med', input_data[1][1].shape[1])])
            name = 'ICLabel_' + labels
            if use_autocorr:
                additional_features['autocorr'] = input_data[1][2].shape[1]
                name += '_autocorr'

            name += '_cv' + str(ind_fold)

            raw['label'][ind_fold] = test_labels

            for arch in icl_archs:

                # reset graph
                tf.reset_default_graph()

                if arch is ConvMANN:
                    # instantiate model
                    model = arch(icl_data[1][1][0].shape[1], additional_features=additional_features,
                                 early_stopping=True, name=name)
                    # train
                    model.train(input_data[2], train_labels_tr, input_data[3], train_labels_val,
                                balance_labels=True, learning_rate=3e-4)

                elif arch is WeightedConvMANN:
                    # instantiate model
                    model = arch(icl_data[1][1][0].shape[1], additional_features=additional_features,
                                 early_stopping=True, name=name, weighting=np.array((2, 1, 1, 1, 1, 1, 1)))
                    # train
                    model.train(input_data[2], train_labels_tr, input_data[3], train_labels_val,
                                balance_labels=True, learning_rate=3e-4)

                else:
                    # instantiate model
                    model = arch(icl_data[1][1][0].shape[1], additional_features=additional_features,
                                 mask=mask, early_stopping=True, name=name)
                    # train
                    model.train(input_data[0], input_data[2], train_labels_tr, input_data[3], train_labels_val,
                                balance_labels=True, learning_rate=3e-4, label_strength=0.9, n_epochs=2)

                # calculate score
                model.load()
                out = model.pred(input_data[4])
                pred = out[0]
                if pred.shape[1] > 7:
                    pred = np.exp(out[1][:, :-1])
                    pred /= pred.sum(1, keepdims=True)
                for n_cls in (2, 3, 5, 7):
                    # get labels and predictions
                    temp_labels = reduce_labels(test_labels, n_cls)
                    temp_pred = reduce_labels(pred, n_cls)
                    # get perf stats
                    thresh, ce, acc, pre, rec, auc, roc, prc, micro_pre, micro_rec, macro_pre, macro_rec, macro_auc \
                        = perf_hard(temp_labels, temp_pred)
                    soft_acc, soft_pre, soft_rec, soft_spe, _ = perf_soft(temp_labels, temp_pred)
                    scores = scores.append(pd.DataFrame([[n_cls, arch.name + ' w/ acor' * use_autocorr, ind_fold, ce, acc,
                                                          micro_pre, micro_rec, macro_pre, macro_rec, macro_auc,
                                                          pre, rec, auc, thresh,
                                                          [x[0] for x in roc], [x[1] for x in roc], [x[2] for x in roc],
                                                          [x[0] for x in prc], [x[1] for x in prc], [x[2] for x in prc],
                                                          soft_acc[0], soft_acc[1], soft_acc[2],
                                                          soft_pre[0], soft_pre[1], soft_pre[2],
                                                          soft_rec[0], soft_rec[1], soft_rec[2],
                                                          soft_spe[0], soft_spe[1], soft_spe[2],
                                                          ]], columns=cols))
                raw[arch.name + ' w/ acor' * use_autocorr][ind_fold] = pred

                # delete temporary files
                shutil.rmtree(join(model.path))

        ind_fold += 1

scores = scores.reset_index(drop=True)
scores.to_pickle('output/scores.pkl')

print('ICLabel candidate crossvalidation complete.')

# n_cls = 2
# for meas in ('cross_entropy', 'accuracy', 'macro_precision', 'macro_recall', 'macro_auc'):
#     try:
#         scores.boxplot(meas, ('autocorr', 'balance_labels', 'arch'))
#         scores[scores['n_cls'] == n_cls].boxplot(meas, ('arch',))
#     except TypeError:
#         pass

pr = scores[scores['n_cls'] == 7].groupby('arch')[['macro_precision','macro_recall', 'macro_auc']].mean()
f1 = 2 / ((1 / pr.values[:, 0]) + (1 / pr.values[:, 1]))
best = f1.argmax()

# train final
labels = 'all'
arch = icl_archs[best/2]
for use_autocorr in (False, True):

    # rescale features
    if use_autocorr:
        input_data = [[topo_data[x],
                       0.99 * icl_data[x][0]['psd'],
                       0.99 * icl_data[x][0]['autocorr'],
                       ] for x in range(4)]
    else:
        input_data = [[topo_data[x],
                       0.99 * icl_data[x][0]['psd'],
                       ] for x in range(4)]

    # augment dataset by negating and/or horizontally flipping topo maps
    for it in range(len(input_data)):
        input_data[it][0] = np.concatenate((input_data[it][0],
                                            -input_data[it][0],
                                            np.flip(input_data[it][0], 2),
                                            -np.flip(input_data[it][0], 2)))
        for it2 in range(1, len(input_data[it])):
            input_data[it][it2] = np.tile(input_data[it][it2], (4, 1))
    try:
        train_labels, test_labels = np.tile(icl_data[1][1][0], (4, 1)), np.tile(icl_data[3][1][0], (4, 1))
    except ValueError:
        train_labels, test_labels = (4 * icl_data[1][1], 4 * icl_data[3][1])

    # describe features and name
    additional_features = OrderedDict([('psd_med', input_data[1][1].shape[1])])
    name = 'ICLabel_' + labels
    if use_autocorr:
        additional_features['autocorr'] = input_data[1][2].shape[1]
        name += '_autocorr'
    name += '_cvFinal'

    # reset graph
    tf.reset_default_graph()
    if arch is ConvMANN:
        # instantiate model
        model = arch(icl_data[1][1][0].shape[1], additional_features=additional_features,
                     early_stopping=True, name=name)
        # check if already exists, if not train
        if not isdir(join('output', arch.name, arch.name + '_' + name)):
            model.train(input_data[1], train_labels, input_data[3], test_labels,
                        balance_labels=True, learning_rate=3e-4)
    elif arch is WeightedConvMANN:
        # instantiate model
        model = arch(icl_data[1][1][0].shape[1], additional_features=additional_features,
                     early_stopping=True, name=name, weighting=np.array((2, 1, 1, 1, 1, 1, 1)))
        # check if already exists, if not train
        if not isdir(join('output', arch.name, arch.name + '_' + name)):
            model.train(input_data[1], train_labels, input_data[3], test_labels,
                        balance_labels=True, learning_rate=3e-4)
    else:
        # instantiate model
        model = arch(icl_data[1][1][0].shape[1], additional_features=additional_features,
                     mask=mask, early_stopping=True, name=name)
        # check if already exists, if not train
        if not isdir(join('output', arch.name, arch.name + '_' + name)):
            model.train(input_data[0], input_data[1], train_labels, input_data[3], test_labels,
                        balance_labels=True, learning_rate=3e-4, label_strength=0.9, n_epochs=2)

print('Done retraining ICLabel')

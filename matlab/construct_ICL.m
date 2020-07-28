%% setup matconvnet
run(fullfile('..', 'plugin', 'matconvnet', 'matlab', 'vl_setupnn'))

%% determine winner and find files
path_output = fullfile('..', 'python', 'output');
folders = dir(path_output);
folders = folders(3:end);
folders = folders([folders.isdir]);
assert(length(folders) == 1, 'ICLabel update failed: too many output folders')
arch = folders.name;
path_arch = fullfile(path_output, arch);
file_weights = fullfile(path_arch, [arch '_ICLabel_all_autocorr_cvFinal'], [arch '_inference.mat']);
file_weights_lite = fullfile(path_arch, [arch '_ICLabel_all_cvFinal'], [arch '_inference.mat']);
file_test_vals = fullfile(path_arch, [arch '_ICLabel_all_autocorr_cvFinal'], [arch '_test_vals.mat']);
file_test_vals_lite = fullfile(path_arch, [arch '_ICLabel_all_cvFinal'], [arch '_test_vals.mat']);


%% build cnn
weights = load(file_weights);
net = build_network(weights, arch, true);

weights_lite = load(file_weights_lite);
net_lite = build_network(weights_lite, arch, false);

%% eval net
test_vals = load(file_test_vals);
test_network(net, test_vals, arch, true);
test_vals_lite = load(file_test_vals_lite);
test_network(net_lite, test_vals_lite, arch, false);

%% save net
path_save = fullfile('..', 'plugin');
netStruct = net.saveobj();
save(fullfile(path_save, 'netICL_weekly.mat'), '-struct', 'netStruct');
netStruct_lite = net_lite.saveobj();
save(fullfile(path_save, 'netICL_lite_weekly.mat'), '-struct', 'netStruct_lite');

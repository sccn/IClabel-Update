function test_network(net, test, arch, autocorr)
net.conserveMemory = 0;
in = {
    'in_image', permute(test.in_image, [2 3 4 1]), ...
    'in_psd_med', permute(test.in_psd, [3 2 4 1])
};
if autocorr
    in(5:6) = {'in_autocorr', permute(test.in_autocorr, [3 2 4 1])};
end
net.eval(in);

% check final layer
if strcmp(arch, 'AltConvMSSGAN')
    test_out = bsxfun(@rdivide, test.discriminator__pred_probs(:, 1:end - 1), ...
        sum(test.discriminator__pred_probs(:, 1:end - 1), 2));
else
    test_out = test.discriminator__pred_probs;
end
difference = squeeze(net.getVar(net.getOutputs()).value)' - test_out;

if max(abs(difference(:))) > 1e-4
%     lucatext([], 'ICLabel update failed: matconvnet version does not match original')
    error('ICLabel update failed: matconvnet version does not match original')
end
    

% TODO: add assert statements checking that differences are about the same
% TODO: send email on failure
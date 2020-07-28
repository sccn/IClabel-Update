function net = build_network(weights, arch, autocorr)

%% build discriminator cnn
net = dagnn.DagNN;
names = {
    'discriminator__image__initial_1_128__conv__kernel' 'discriminator__image__initial_1_128__conv__bias'
    'discriminator__image__pyramid_128_256__conv__kernel' 'discriminator__image__pyramid_128_256__conv__bias'
    'discriminator__image__pyramid_256_512__conv__kernel' 'discriminator__image__pyramid_256_512__conv__bias'
};
in_layers = {
    'in_image'
    'discriminator_image_layer1_x'
    'discriminator_image_layer2_x'
};
nodes = [1 128 256 512];
for it = 1:size(names, 1)
    sit = num2str(it);
    
    % conv
    convBlock = dagnn.Conv('size', [4 4 nodes(it) nodes(it + 1)], ...
        'hasBias', true, 'stride', 2, 'pad', 1);
    params = {['discriminator_image_layer' sit '_conv_kernel'], ...
        ['discriminator_image_layer' sit '_conv_bias']};
    net.addLayer(['discriminator_image_layer' sit '_conv'], convBlock, ...
        in_layers(it), {['discriminator_image_layer' sit '_z']}, params);
    for it_p = 1:length(params)
        net.params(net.getParamIndex(params(it_p))).value ...
            = weights.(names{it, it_p});
    end
    
    % leaky relu
    reluBlock = dagnn.ReLU('leak', 0.2);
    net.addLayer(['discriminator_image_layer' sit '_relu'], reluBlock, ...
        {['discriminator_image_layer' sit '_z']}, {['discriminator_image_layer' sit '_x']}, {});
end

%% 1d CNNs
descriptors = {'psd_med'};
if autocorr
    descriptors{2} = 'autocorr';
end

names = {};
for it = 1:length(descriptors)
    names(it) = {{
        ['discriminator__' descriptors{it} '__initial_100_128__conv__kernel'] ...
            ['discriminator__' descriptors{it} '__initial_100_128__conv__bias']
        ['discriminator__' descriptors{it} '__hidden_128_256__conv__kernel'] ...
            ['discriminator__' descriptors{it} '__hidden_128_256__conv__bias']
        ['discriminator__' descriptors{it} '__final_256_1__conv__kernel'] ...
            ['discriminator__' descriptors{it} '__final_256_1__conv__bias']
    }};
end

nodes = [1 128 256 1];
% over networks
for it = 1:length(names)
    % over layers
    for it2 = 1:size(names{it}, 1)
        sit = num2str(it2);
        if it2 == 1
            in_layer = ['in_' descriptors{it}];
        else
            in_layer = ['discriminator_' descriptors{it} '_layer'  num2str(it2 - 1) '_conv_' 'x'];
        end
        name_start = ['discriminator_' descriptors{it} '_layer'  sit '_conv_'];
            
        % conv
        convBlock = dagnn.Conv('size', [1 3 nodes(it2) nodes(it2 + 1)], ...
            'hasBias', true, 'pad', [0 0 1 1]);
        params = {[name_start 'kernel'], [name_start 'bias']};
        net.addLayer([name_start 'conv'], convBlock, ...
            in_layer, {[name_start 'z']}, params);
        for it_p = 1:length(params)
            param_val = weights.(names{it}{it2, it_p});
            if it_p == 1
                param_val = permute(param_val, [4 1 2 3]);
            end
            net.params(net.getParamIndex(params(it_p))).value = param_val;
        end
        
        % leaky relu
        reluBlock = dagnn.ReLU('leak', 0.2);
        net.addLayer([name_start 'relu'], reluBlock, ...
            {[name_start 'z']}, {[name_start 'x']}, {});
    end
end

%% concatenate and convolve discriminator pre-outputs
n_in = 612 + 100*autocorr;
if strcmp(arch, 'AltConvMSSGAN')
    n_out = 8;
else
    n_out = 7;
end
names = {
    ['discriminator__final_' num2str(n_in) '_' num2str(n_out) '__conv__kernel'] ...
        ['discriminator__final_' num2str(n_in) '_' num2str(n_out) '__conv__bias']
};

% create blocks
reshapeBlock = dagnn.Reshape('size', [1 1 100]);
concat1dBlock = dagnn.Concat('inputSizes', repmat({[1 1 100]}, 1, 4), 'dim', 1);
concat2dBlock = dagnn.Concat('inputSizes', repmat({[4 1 100]}, 1, 4), 'dim', 2);
concat3dBlock = dagnn.Concat('inputSizes', {[4 4 1000], [4 4 100], [4 4 100]}, 'dim', 3);

for it = 1:length(descriptors)
    % tile non-image outputs
    net.addLayer(['discriminator_' descriptors{it} '_reshape'], ...
        reshapeBlock, ...
        ['discriminator_' descriptors{it} '_layer3_conv_x'], ...
        ['discriminator_' descriptors{it} '_layer3_reshape_x'])
    net.addLayer(['discriminator_' descriptors{it} '_concat1'], ...
        concat1dBlock, ...
        repmat({['discriminator_' descriptors{it} '_layer3_reshape_x']}, 1, 4), ...
        {['discriminator_' descriptors{it} '_concat1_z']})
    net.addLayer(['discriminator_' descriptors{it} '_concat2'], ...
        concat2dBlock, ...
        repmat({['discriminator_' descriptors{it} '_concat1_z']}, 1, 4), ...
        {['discriminator_' descriptors{it} '_concat2_z']})
end
% concatenate all 3 data streams in discriminator
inputs = {'discriminator_image_layer3_x'};
for it = 1:length(descriptors)
    inputs{it + 1} = ['discriminator_' descriptors{it} '_concat2_z'];
end
net.addLayer('discriminator_concat', concat3dBlock, inputs, ...
    {'discriminator_concat_x'})

% discriminator output conv (exclude fake label)
if strcmp(arch, 'AltConvMSSGAN')
    convBlock = dagnn.Conv('size', [4 4 1300 7], 'hasBias', true);
else
    convBlock = dagnn.Conv('size', [4 4 712 7], 'hasBias', true);
end 
params = {'discriminator_conv_kernel', 'discriminator_conv_bias'};
net.addLayer('discriminator_conv', convBlock, {'discriminator_concat_x'}, ...
    {'discriminator_conv_x'}, params);
net.params(net.getParamIndex(params(1))).value = weights.(names{1})(:, :, :, 1:7);
net.params(net.getParamIndex(params(2))).value = weights.(names{2})(:, 1:7);

% softmax
softmaxBlock = dagnn.SoftMax();
net.addLayer('discriminator_softmax', softmaxBlock, {'discriminator_conv_x'}, {'discriminator_probabilities'}, {});

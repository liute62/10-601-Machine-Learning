% #############################################################################################
% #########  YOU SHOULD MAINTAIN THE RETURN TYPE AND SHAPE AS PROVIDED IN STARTER CODE   ######
% #############################################################################################

function [input_od] = relu_backward(output, input, layer)
% RELU backward

% Args:
% output: a cell array contains output data and shape information
% input: a cell array contains input data and shape information
% layer: one cnn layer, defined in testLeNet.m

% Returns:
% input_od: gradients w.r.t input data

input_od = zeros(size(input.data));

% TODO: your implementation goes below this comment
% implementation begins

% implementation ends

assert(all(size(input.data) == size(input_od)), 'input_od does not have the right length');

end

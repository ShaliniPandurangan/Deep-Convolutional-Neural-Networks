function [c, s] = pdfb2vec(y)
%

n = length(y);

% Save the structure of y into s
s(1, :) = [1, 1, size(y{1})];

% Used for row index of s
ind = 1;

for l = 2:n
    nd = length(y{l});
    
    for d = 1:nd
        s(ind + d, :) = [l, d, size(y{l}{d})];
    end
    
    ind = ind + nd;
end

% The total number of PDFB coefficients
nc = sum(prod(s(:, 3:4), 2));

% Assign the coefficients to the vector c
c = zeros(1, nc);

% Variable that keep the current position
pos = prod(size(y{1}));

% Lowpass subband
c(1:pos) = y{1}(:);

% Bandpass subbands
for l = 2:n    
    for d = 1:length(y{l})
        ss = prod(size(y{l}{d}));
        c(pos+[1:ss]) = y{l}{d}(:);
        pos = pos + ss;
    end
end
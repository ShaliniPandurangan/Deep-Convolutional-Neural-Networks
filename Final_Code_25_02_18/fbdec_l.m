function [y0, y1] = fbdec_l(x, f, type1, type2, extmod)
% Modulate f
f(1:2:end) = -f(1:2:end);

if min(size(x)) == 1
    error('Input is a vector, unpredicted output!');
end

if ~exist('extmod', 'var')
    extmod = 'per';
end

% Polyphase decomposition of the input image
switch lower(type1(1))
    case 'q'
        % Quincunx polyphase decomposition
	    [p0, p1] = qpdec(x, type2);
	
    case 'p'
	    % Parallelogram polyphase decomposition
	    [p0, p1] = ppdec(x, type2);
	
    otherwise
	    error('Invalid argument type1');
end

% Ladder network structure
y0 = (1 / sqrt(2)) * (p0 - sefilter2(p1, f, f, extmod, [1, 1]));
y1 = (-sqrt(2) * p1) - sefilter2(y0, f, f, extmod);
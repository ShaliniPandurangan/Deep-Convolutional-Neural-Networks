function y = extend2(x, ru, rd, cl, cr, extmod)
	           
[rx, cx] = size(x);
    
switch extmod
    case 'per'
	I = getPerIndices(rx, ru, rd);
	y = x(I, :);
	
	I = getPerIndices(cx, cl, cr);
	y = y(:, I);
	
    case 'qper_row'
	rx2 = round(rx / 2);
		
	y = [[x(rx2+1:rx, cx-cl+1:cx); x(1:rx2, cx-cl+1:cx)], x, ...
	     [x(rx2+1:rx, 1:cr); x(1:rx2, 1:cr)]];
	
	I = getPerIndices(rx, ru, rd);
	y = y(I, :);	
	
    case 'qper_col'
	cx2 = round(cx / 2);
	
	y = [x(rx-ru+1:rx, cx2+1:cx), x(rx-ru+1:rx, 1:cx2); x; ...
	     x(1:rd, cx2+1:cx), x(1:rd, 1:cx2)];
	
	I = getPerIndices(cx, cl, cr);
	y = y(:, I);
	
    otherwise
	error('Invalid input for EXTMOD')
end	

%----------------------------------------------------------------------------%
% Internal Function(s)
%----------------------------------------------------------------------------%
function I = getPerIndices(lx, lb, le)

I = [lx-lb+1:lx , 1:lx , 1:le];

if (lx < lb) | (lx < le)
    I = mod(I, lx);
    I(I==0) = lx;
end
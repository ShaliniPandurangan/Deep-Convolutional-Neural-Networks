function [y,ylo] = DualTreeComplesCurvelet(x, pfilt, dfilt, nlevs)
if isempty(nlevs)
    y = {x};
    
else
    % Get the pyramidal filters from the filter name
    [h, g] = pfilters(pfilt);
    
    if nlevs(end) ~= 0
        % Laplacian decomposition
        [xlo, xhi] = lpdec(x, h, g);
    
        % DFB on the bandpass image
        switch dfilt        % Decide the method based on the filter name
            case {'pkva6', 'pkva8', 'pkva12', 'pkva'}   
                % Use the ladder structure (whihc is much more efficient)
                xhi_dir = dfbdec_l(xhi, dfilt, nlevs(end));
            
            otherwise       
                % General case
                xhi_dir = dfbdec(xhi, dfilt, nlevs(end));                
        end
        
    else        
        % Special case: nlevs(end) == 0
        % Perform one-level 2-D critically sampled wavelet filter bank
        [xlo, xLH, xHL, xHH] = wfb2dec(x, h, g);
        xhi_dir = {xLH, xHL, xHH};
    end
    
    % Recursive call on the low band
    ylo = DualTreeComplesCurvelet(xlo, pfilt, dfilt, nlevs(1:end-1));

    % Add bandpass directional subbands to the final output
    y = {ylo{:}, xhi_dir};
end
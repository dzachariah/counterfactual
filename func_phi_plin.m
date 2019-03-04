function [ phi ] = func_phi_plin( x, c_set )

% x row vector R^d
% Dave Zachariah

%% Initialize
D      = length(x);
[D0,dimC] = size(c_set);
M      = dimC - 1;
phi    = zeros(1,1 + M*D0 + D-D0);
vec1   = ones(1,M);

%% Construct
for d = 1:D0
   
    %row vector
    delta = x(d)*vec1 - c_set(d,1:M);
    reg   = delta .* (delta > 0);
    
    if x(d) > c_set(d,M+1)
        reg(M) = c_set(d,M+1);
    end
    
    %store
    idx_start = 2 + M*(d-1);
    phi(1,idx_start:idx_start+M-1) = reg;
end

phi(1) = 1;

end

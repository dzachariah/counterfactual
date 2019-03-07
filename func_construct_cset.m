function [ c_set ] = func_construct_cset( X, M )
%TODO:
%Dave Zachariah

%% Initialize
[~,N] = size(X); %set of continuous inputs (feature x sample)

%% Create knots

%Emp. quantile function for each dimension
F_inv_hat = sort( X, 2 );

%Quantiles
q_set = (((1:M+1)-1)/M);

%Indices
idx    = floor(q_set * N);
idx(1) = 1;

%Access F^-1 at quantiles
c_set = F_inv_hat(:,idx);

end

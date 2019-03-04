function [ C_min, C_max, y_hat ] = compute_spiceinterval( phix, beta, M, y_min, y_max, y, Phi, U, L, Gamma, rho, kappa, w_hat )
%Compute conformal prediction interval of SPICE predictor
%Dave Zachariah 2017-03-22

%Input:
% phix   - phi(x) rowvector with test point x
% beta   - target coverage
% M      - grid resolution of output

%Data:
% y      - nx1 outputs from training set
% Phi    - nxp regressor matrix
% U      - dimension of mean parameters
% L      - number of iterations per dimension

%SPICE: 
% Gamma  - pxp gram matrix
% rho    - px1
% kappa  - 1x1
% w_hat  - learned weights

%Output:
% C - 2x1 vector of interval boundaries

%Usage:
% Given phi(x_test) of a test point x_test, compute confidence interval
% C(x) for prediction


%% Initialize
N        = length(y);
P        = length(phix);
Y_grid   = linspace(y_min, y_max, M);
C_min    = -inf;
C_max    =  inf;
nc_min   = 1;
y_hat    =  inf;

%% Compute

k = 0;
for y_new = Y_grid;
    %Update model
    [w_new] = func_newsample_covlearn( y_new, phix, Gamma, rho, kappa, w_hat, N+1, L, P, U );

    %Compute residuals
    r           = abs(y - Phi*w_new);
    resid_count = sum( r <= (abs( y_new - phix*w_new )*ones(N,1)) );

    %Nonconformal statistic
    ncstat = (1 + resid_count) / (N+1);

    %Indicate
    ind_check = ( (N+1)*ncstat <= ceil(beta*(N+1)) );

    %Update interval (take into account rounding)
    k = k+1;
    if (ind_check == 1) && ((k>1) && (k<M))
        if (C_min == -inf) %initial
            C_min = Y_grid(k-1); %round
        end
        C_max = Y_grid(k+1); %round
    elseif (ind_check == 1) && (k==1)
       disp('CI warning: min(Y_grid) not low enough')
    elseif (ind_check == 1) && (k==M)
       disp('CI warning: max(Y_grid) not high enough')
    end
    
    %Track lowest stat
    if ncstat < nc_min
        nc_min  = ncstat;
        y_hat = y_new;
    end
    
end

if (C_min == -inf)
    disp('CI warning: refine {Y_grid}')
end



end


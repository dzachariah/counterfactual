clear
close all

%% Seed
%RandStream.setGlobalStream(RandStream('mt19937ar','seed',100));
RandStream.setGlobalStream(RandStream('mt19937ar','seed','shuffle'));

%% Initialize
mc_runs = 1;
n = 120;
d = 1;
beta = 0.90;

M = 8;

N_grid = 100;

%Allocate
y = zeros(1,n);
g = zeros(1,n);
x = zeros(d,n);


%% Generate data
x_star = 30;

for i = 1:n
   
    %Group
    g(i) = rand < 0.5;
    
    %Outcome
    if g(i) == 0
        x(i) = 20 + 10*randn;
        y(i) = 72 + 3*sqrt( abs(x(i)) )  + 1*randn;
        
    elseif g(i) == 1
        x(i) = 40 + 10*randn;
        y(i) = 90 + exp( 0.06*x(i) )  + 1*randn;
        
    else
        disp('ERROR')
    end
    
end



%% Learn predictors

%Adaptive basis
x_cont = x;
c_set  = func_construct_cset(x_cont,M);

%Construct subsets
idx0 = find(g == 0);
idx1 = find(g == 1);

y0 = y( idx0 ); 
x0 = x(:,idx0 );
n0 = length(y0);

y1 = y( idx1 ); 
x1 = x(:,idx1 );
n1 = length(y1);

%Learn SPICE predictors
Phi0 = zeros(n0,1+M*d);
Phi1 = zeros(n1,1+M*d); 

for k = 1:n0
    Phi0(k,:) = func_phi_plin( x0(:,k), c_set );
end
for k = 1:n1
    Phi1(k,:) = func_phi_plin( x1(:,k), c_set );
end

[ w0_hat, Gamma0, rho0, kappa0 ] = compute_spicepredictor( y0', Phi0, 1, 3 );
[ w1_hat, Gamma1, rho1, kappa1 ] = compute_spicepredictor( y1', Phi1, 1, 3 );

y_min = min(y);
y_max = max(y);


%% Plot

figure(1)
plot(x0, y0, 'ko' ), hold on, grid on
plot(x1, y1, 'b+' )
xlabel('$x$','Interpreter','latex'),
ylabel('$y$','Interpreter','latex'),

%figure(2)
phi_test_row = func_phi_plin( x_star, c_set );
 
y0_hat = phi_test_row*w0_hat;
y1_hat = phi_test_row*w1_hat;

%Compute intervals
[ y0_min, y0_max ] = compute_spiceinterval( phi_test_row, beta, N_grid, y_min, y_max, y0', Phi0, 1, 1, Gamma0, rho0, kappa0, w0_hat );
[ y1_min, y1_max ] = compute_spiceinterval( phi_test_row, beta, N_grid, y_min, y_max, y1', Phi1, 1, 1, Gamma1, rho1, kappa1, w1_hat );


figure(2)
plot( [y0_min y0_max], [0 0], 'LineWidth', 1.5 ), hold on, grid on
plot( [y1_min y1_max], [1 1], 'LineWidth', 1.5 )
plot( y0_hat, 0, 'o', 'LineWidth', 1.5 ), hold on
plot( y1_hat, 1, 'o','LineWidth', 1.5 )
axis( [ min([y0_min-1.2*(y0_hat-y0_min),y1_min-1.2*(y1_min-y1_hat)]),   max([y0_max+1.2*(y0_max-y0_hat),y1_max+1.2*(y1_max-y1_hat)]),  -0.5, 1.5 ]  )
xlabel('outcome $y$','Interpreter','latex'),
ylabel('group $g$','Interpreter','latex'),

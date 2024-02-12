N = 10000 ; % the length of the time series
x = zeros(N,1);
x(1) = 1;
x(2) = 1;
a0 = 0;
a1 = 0.7;
a2 = -0.5;


% The polynomial a2*z^2 + a1*z + a0 is represented as poly = [-a2 -a1 1]
% The roots of the polynomial are returned in a column vector.
poly = [-a2 -a1 1]
r = roots(poly)
abs(r)
sigma2 = 0.2; % variance for epsilon


for t=1:N
epsilon(t) = randn * sqrt(sigma2) ;
end


for t=3:N
x(t)= a0 + a1*x(t-1) + a2*x(t-2) + epsilon(t);
end


% Discard the first 100 samples to remove transient samples.
x(1:100) = [];
epsilon(1:100) = [];
mu = mean(x);
var_x = var(x); % variance of x
N = N - 100;


% autocorrelation
L = 15 ; % L denotes lag.
rho = zeros(1,L);


for k=1:L
for t=1:N-k
rho(k) = rho(k) + (x(t)-mu)*(x(t+k)-mu);
end
rho(k) = rho(k)/(N-k)/var_x;
end


subplot(1,2,1)
plot(x(1:500));
xlabel('t');
ylabel('x');
subplot(1,2,2)
bar(rho)
xlabel('k');
ylabel('ACF');
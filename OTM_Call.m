clc % clear workspace of prior output.
clear all % clear variables
clf; %clear figures

T = 5;
r = 0.15; % interest rate
mu = 0.15; % drift coefficient
sigma = 0.3; % volatility
S0 = 100; % asset price at time t=0

K = 110; % strike price
N = 100 ; % number of time steps
dt = T/N;
t_value = [0:dt:T];
W = zeros(1,N+1); % Brownian motion
S = zeros(1,N+1); % asset price
V = zeros(1,N+1); % option price
Delta = zeros(1,N+1); % Delta
D = zeros(1,N+1); % bank deposit
Pi = zeros(1,N+1); % portfolio
D(1) = 0.0; % Choose any number for the initial cash amount.
S(1)= S0;

for i=2:N+1
dW = sqrt(dt)*randn;
W(i) = W(i-1) + dW;
S(i) = S(i-1) + mu*S(i-1)*dt + sigma*S(i-1)*dW;
end

for i=1:N+1
tau = T-(i-1)*dt;
d1 = (log(S(i)/K) + (r+0.5*sigma^2)*tau)/sigma/sqrt(tau);
d2 = d1 - sigma*sqrt(tau);
V(i)= S(i)*normcdf(d1) - K*exp(-r*tau)*normcdf(d2);
Delta(i)=normcdf(d1);
end

for i = 1:N
D(i+1) = exp(r*dt)*D(i) + (Delta(i)-Delta(i+1))*S(i+1);
end

for i = 1:N+1
Pi(i) = -V(i) + D(i) + Delta(i)*S(i);
end

subplot(3,2,1);
plot([0:dt:T],W)
title('W')

subplot(3,2,2);
plot([0:dt:T],S)
title('S')
hold on
plot([0:dt:T],K*ones(1,N+1),'--')

subplot(3,2,3);
plot([0:dt:T],V)
title('V')

subplot(3,2,4);
for i=1:N
x = (i-1)*dt:dt/(500/N):i*dt;
y = Delta(i)*exp(0*x);
plot(x,y) % Plot the graph on each subinterval.
hold on
end
title('\Delta')

subplot(3,2,5);
for i=1:N
x = (i-1)*dt:dt/(500/N):i*dt;
y = D(i);
plot(x,y*exp(r*(x-(i-1)*dt))) % Plot the graph on each subinterval.
hold on
end
title('D')

subplot(3,2,6);
plot([0:dt:T],Pi(1)*exp(r*[0:dt:T]))
title('\Pi')
hold on
plot([0:dt:T],Pi)

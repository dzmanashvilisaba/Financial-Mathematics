%In this code we are generating the scenario when the asset price of PUT 
%option after given period equals to strike price. We are observing the 
%changes in option price, deltaas, bank deposits and portfolios and we are
%plotting it.

clc % clear workspace of prior output.
clear all; % clear variables
clf; %clear figures


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%     VARIABLES     %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


N = 100;    %Number of time stamps
M = 5;      %Number of sample paths
T = 5;      %End period 
dt = T/N;   %Length of time interval

mu = 0.15       %Drift coefficient
sigma = 0.3;    %Volatility
S0 = 100;       %Initial price

K = 110;                %Strike price
Delta = zeros(M,N+1);   %Delta - the quantity
D0 = 10.0;               %Initial cash amount
r = 0.15;               %
A = 1;                  %Payoff

%   Variables for plotting
time = 0:dt:T;

%   Matrices
W = zeros(M, N+1);  %Brownian Motions
S = zeros(M, N+1);  %Geometric Brownian Motion/Bridge (Asset Price)
X = zeros(M, N+1);  %Brownian Bridges
D = zeros(M,N+1);   %Bank Deposits
V = zeros(M,N+1);   %Option Prices
P = zeros(M,N+1);   %Portfolios



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
%%%%%%%%%%%%%    MATHEMATICS    %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%   Generate matrix for  brownian motions and geometric brownian motions
dW = sqrt(dt)*randn(M,N);
for i=1:N
    W(:,i+1) = W(:,i) + dW(:,i);
end

a = 0;
b = ( log(K/S0) - (mu-(sigma^2)/2)*T )/sigma
%brownian motion
for i=1:N+1
    X(:,i) = a + (b-a)*(i-1)*dt/T + W(:,i) - W(:,N+1)*(i-1)*dt/T;
end

%asset price with S_T = K
S(:,1) = S0;
for i=1:N+1
    S(:,i) = S0*exp((mu-(sigma^2)/2)*(i-1)*dt + sigma*X(:,i));
end

%option prices and delta
for i= 1:N
    for j = 1:M
        tau = T-(i-1)*dt;
        d1 = (log(S(j,i)/K) + (r+0.5*sigma^2)*tau)/sigma/sqrt(tau);
        d2 = d1 - sigma*sqrt(tau);
        V(j,i)= K*exp(-r*tau)*normcdf(-d2) - S(j,i)*normcdf(-d1);
        Delta(j,i)=normcdf(d1)-1;   
    end
end
V(:,N+1) = A/2;

%bank deposits
D(:,1) = D0;
for i = 1:N
        D(:,i+1) = exp(r*dt)*D(:,i) + (Delta(:,i)-Delta(:,i+1)).*S(:,i+1);
end

%portfolios
for i = 1:N+1
    P(:,i) = -V(:,i) + D(:,i) + Delta(:,i).*S(:,i);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
%%%%%%%%%%%%%       PLOTS       %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%   Plot the  brownian bridges X
subplot(3,2,1);
for i=1:M
    plot(time,X(i,:),'color',hsv2rgb([1-i/M 1 1]));
    hold on;
end
plot(time,zeros(1,N+1),'k-.');
plot(time,ones(1,N+1)*b,'k-.');
title('Brownian Bridges');

%   Plot the  geometric brownian bridges S
subplot(3,2,2);
for i=1:M
    plot(time,S(i,:),'color',hsv2rgb([1-i/M 1 1]));
    hold on;
end
plot(time,ones(1,N+1)*S0,'k-.');
plot(time,ones(1,N+1)*K,'k-.');
title('Asset Prices');

%   Plot the  option prices V
subplot(3,2,3);
for i=1:M
    plot(time,V(i,:),'color',hsv2rgb([1-i/M 1 1]));
    hold on;
end
title('Option Prices');
set(gca,'xlim',[0,T],'xtick',[0:T/4:T]);

%   Plot the deltas Delta
subplot(3,2,4);
for i=1:M
    plot(time,Delta(i,:),'color',hsv2rgb([1-i/M 1 1]));
    hold on;
end
title('Delta');

%   Plot the bank deposits D
subplot(3,2,5);
for i=1:M
    plot(time,D(i,:),'color',hsv2rgb([1-i/M 1 1]));
    hold on;
end
title('Bank Deposits');


%   Plot the portfolio P
subplot(3,2,6);
for i=1:M
    plot(time,P(i,:),'color',hsv2rgb([1-i/M 1 1]));
    hold on;
end
plot([0:dt:T],P(1)*exp(r*[0:dt:T]),'k-.','LineWidth',2);
title('Portfolios');
% Binomial tree method for the American put option
clf
clc
clear all

T = 1; % maturity date
N = 6; % number of time periods
S0 = 50; % asset price at time 0
S = zeros(N+1,N+1); % asset price
S(1,:) = S0;
P = zeros(N+1,N+1); % American put price
K = 60; % strike Price
r = 0.05; % interest rate
sigma = 0.3; % volatility
dt = T/N;

u = exp(sigma*sqrt(dt) + (r-0.5*sigma^2)*dt);
d = exp(-sigma*sqrt(dt) + (r-0.5*sigma^2)*dt);

q = (exp(r*dt)-d)/(u-d);

% i for time, j for asset price level
for i = 2:N+1
    for j = 1:i
        S(i,j) = S0*u^(j-1)*d^(i-j); % asset price
    end
end

for j = 1:N+1
    P(N+1,j) = max(K-S(N+1,j),0); % American put payoff at T
end

for i = N:-1:1 % backward in time
    for j = 1:i
        P(i,j) = max(exp(-r*dt)*(q*P(i+1,j+1)+(1-q)*P(i+1,j)), K-S(i,j));
    end
end

subplot(1,2,1)
for i = 1:N+1 % backward in time
plot(i-1,S(i,1:i),'or','MarkerSize',16,'LineWidth',1)
for j = 1:i
text(i-1,S(i,j),num2str(S(i,j),'%.2f'),'Color','k')
end
hold on
end
line([0 N],[K K],'Color','b')
xlabel('Time Steps')
ylabel('Asset Price')
title('Binomial Tree for Asset Price');

hold off
subplot(1,2,2)
for i = 1:N+1
plot(i-1,S(i,1:i),'sb','MarkerSize',16,'LineWidth',1)
for j = 1:i
text(i-1,S(i,j),num2str(P(i,j),'%.2f'),'Color','k')
end
hold on
end
xlabel('Time Steps')
ylabel('American Put Price')
title('Binomial Tree for American Put Option Price');

% Optimal exercise boundary for american put option is
D=P+S-K*ones(N+1,N+1)

for i=1:N+1
for j=1:N+1
if(j>i) 
D(i,j)=0
end
end
end
[A B] = find(D)
coords = zeros(1,N+1);
for j=1:N+1
for i=length(A):-1:1
    if (A(i)==j)
    coords(j)=B(i);
    end
end
end



V = coords %- ones(1,length(coords))
plot(0:N,S(V),'r--', 'LineWidth',1.5);
American_put_price = P(1,1)
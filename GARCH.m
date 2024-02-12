clc 
clear all 
clf

sigma2(1) = 0.1;
a0 = 0.01;
alpha = 0.05;
beta = 0.85;
T = 100000; % Tail estimation requires large T

for t = 1:1:T
    u(t) = sqrt(sigma2(t))*randn;
    sigma2(t+1) = a0 + alpha * u(t)^2 + beta * sigma2(t);
end
u(1:100) = []; % Discard the first 100 samples to remove transient samples
sigma2(1:100) = [];

subplot(2,2,1)
plot(u(1:300));
xlabel('u');
title("Time series for u")

subplot(2,2,2)
plot(sigma2(1:300));
xlabel('\sigma^2');
title("Time series for \sigma^2")

realKurtosis = 3 * (1 - (alpha + beta)^2) / (1 - (alpha + beta)^2 - 2*alpha^2)
estKurtosis = mean(u.^4)/mean(u.^2)^2

u = u./std(u); % Normalize u
for k = 1:1:4
    P(1,k) = 1 - normcdf(k); % Tail probability of the normal distribution
    P(2,k) = mean(u > k); % Estimate the tail probability of u
end
format long
P

subplot(2,2,3)
totCategories = 30;
len = (max(u) - min(u)) / (totCategories - 1);
bars = min(u):len:max(u);
data = zeros(1, totCategories);
j = 1;
for i = 1:length(u)
    while (u(i) > bars(j))
       j = j + 1;
    end
    data(j) = data(j) + 1;
    j = 1;
end
p1 = plot(bars, data);
ylabel('Quantity')
title("Distribution of u")


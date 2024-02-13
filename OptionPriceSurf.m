K = 10;
S0 = 10;
r = 0.05;
sigma = 0.3;
mu = 0.1;


T = 1;
N = 51;
dt = T/(N-1);
S_path = zeros(N,1);
S_path(1)= S0; % asset price at time 0

for i=1:N-1
dW = sqrt(dt)*randn;
S_path(i+1) = S_path(i) + mu*S_path(i)*dt + sigma*S_path(i)*dW;
end

C_path = zeros(N,1);
P_path = zeros(N,1);


for i = 1:N
S = S_path(i);
tau = T-(i-1)*dt;
d1 = (log(S/K) + (r+0.5*sigma^2)*tau)/(sigma*sqrt(tau));
d2 = d1 - sigma*sqrt(tau);
C_path(i) = S*normcdf(d1) - K*exp(-r*tau)*normcdf(d2);
P_path(i) = C_path(i) - S + K*exp(-r*tau);
end


t_value = linspace(0,T,N);
S_value = linspace(0,20,N);
C = zeros(N,N);
P = zeros(N,N);


for j=1:N
S = S_value(j);
for i = 1:N-1
tau = T-t_value(i);
d1 = (log(S/K)+(r+0.5*sigma^2)*tau)/(sigma*sqrt(tau));
d2 = d1-sigma*sqrt(tau);
N1 = normcdf(d1);
N2 = normcdf(d2);
C(i,j) = S*N1 - K*exp(-r*tau)*N2;
P(i,j) = C(i,j) + K*exp(-r*tau) - S;
end
C(N,j) = max(S-K,0); % payoff at T
P(N,j) = max(K-S,0); % payoff at T
end


% Plot the superimposed image of asset price movement.
[S_grid,t_grid] = meshgrid(S_value,t_value);
figure(1);
surf(S_grid,t_grid,C)
hold on
plot3(S_path,t_grid,C_path);
xlabel('S'), ylabel('t'), zlabel('Call')
hold off

figure(2);
surf(S_grid,t_grid,P)
hold on
plot3(S_path,t_grid,P_path);
xlabel('S'), ylabel('t'), zlabel('Put')
hold off
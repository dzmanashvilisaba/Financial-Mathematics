# Financial-Mathematics

## Brownian Motion
A stochastic process $W_t$, $t \geq 0$, is called a
Brownian motion if it has the following properties:
  1. $W_0 = 0$ and $t\mapsto W_t$, $t \geq 0$, is continuous with probability 1.
  2. For $0 \leq s \leq t$ the increment $W_t - W_s$ has normal distribution with mean 0 and
variance $t - s$.
  3. For $0 \leq t_1 < t_2 \leq t_3 < t4 \leq \cdots \leq t_{2n-1} < t_{2n}$ the increments

$$ W(t_2) - W(t_1),  W(t_3) - W(t_2), \cdots , W(t_{2n}) - W(t_{2n-1}) $$

are independent.

Stocastic differential equation for Brownian motion will be:

$$ dW_t = \sqrt{dt} Z_t $$

where $Z_t$ is standard Brownian motion increment that follows normal distribution. Stocastic differential equation for Geometric Brownian motion will be:

$$ dS_t = \mu S_tdt + \sigma S_t dW_t $$

where and $\mu$, the percentage drift, and $\sigma$, the percentage volatility, are constants

|  Brownian Motion     |   Geometric Brownian Motion   | 
| -------------- | -------------- |
| ![]( https://github.com/dzmanashvilisaba/Financial-Mathematics/blob/main/graphs/Brownian.png )    |  ![]( https://github.com/dzmanashvilisaba/Financial-Mathematics/blob/main/graphs/GeometricBrownian.png ) |   








## American Put Option Demo

In this demo, the price V of an American option is considered as a function of the 
stock value S and time t, V = V(S,t). The financial parameters like strike, volatility,
etc. are assumed to be constants. The demo computes the option price for a range of discrete 
stock values and a range of discrete time values. The demo also computes the optimal exercise boundary as a function of time. The results are visualized in three figures. 

  1.  The first figure is a graph  of the American option price at the initial time. For comparison reasons, this figure also shows a graph of the corresponding European option and a graph of the payoff.
  2.  The second figure displays a surface of the  option price as a function of the stock value and time.
  3.  The third graph displays the optimal exercise boundary.

|        |      |      |
| -------------- | -------------- | -------------- |
| ![]( https://github.com/dzmanashvilisaba/Financial-Mathematics/blob/main/graphs/AmericanPut1.png )    |  ![]( https://github.com/dzmanashvilisaba/Financial-Mathematics/blob/main/graphs/AmericanPut2.png ) |    ![]( https://github.com/dzmanashvilisaba/Financial-Mathematics/blob/main/graphs/AmericanPut3.png ) |





##  Multiperiod Bionamial Tree Method for American Option
It is a numerical technique used in finance to value options, where the price of the underlying asset can change over multiple periods. It is a discrete-time method for option pricing developed by Cox, Ross, and Rubinstein.

![](  https://github.com/dzmanashvilisaba/Financial-Mathematics/blob/main/graphs/BinomialTree.png  )  







##  Time Series Models

A time series is a sequential set of data points, measured typically over successive times.
In time series analysis there are models that are fitted to data either to better understand the data or to predict future points in the series. ARIMA is one such model.
  
**ARIMA** is an acronym that stands for Auto-Regressive Integrated Moving Average. Specifically,

* **AR** Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.

* **I** Integrated. The use of differencing of raw observations in order to make the time series stationary.

* **MA** Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.

A standard notation is used of ARIMA(p, d, q) where the parameters are substituted with integer values to quickly indicate the specific ARIMA model being used.
* p The number of lag observations included in the model, also
called the lag order.
* d The number of times that the raw observations are
differenced, also called the degree of differencing.
* q The size of the moving average window, also called the order
of moving average.

In other words, ARIMA model can be configured to perform the function of an **ARMA** model, and even a simple **AR**, **I**, or **MA** model

###  AR Model
- Autoregressive models are based on the idea that current value $X_t$ of the series $(X)_ {t=1} ^{\infty}$, can be explained as a linear combination of past $p$ values, $X_{t-1}, X_{t-2}, \cdots, X_{t-p}$, together with a random
error $\omega_t$ in the same series.

$$ AR(p) \qquad : \qquad X_t = \phi_1X_{t-1} + \phi_2X_{t-2} + \cdots + \phi_pX_{t-p}  + \omega_t   $$


- where $X_t$ is stationary, $\omega_t \sim \omega N(0, \sigma^2_{\omega})$,  and $\phi_1, \phi_2, \cdots , \phi_p$ are model parameters. The hyperparameter $p$
represents the length of the “direct look back” in the series.


###  MA Model
- This model assumes the  correlated noise structures in the time series.

$$ MA(q) \qquad : \qquad X_t = \omega_t + \omega_{t-1}\theta_1 + \omega_2\theta_2 + \cdots + \omega_q\theta_q  $$

- where $\omega_t \sim \omega N(0, \sigma^2 _{\omega} )$, and $\theta_1, \theta_2, \cdots , \theta_q$  are parameters.
- Although it looks like a regression model, the difference is that the $(\omega)$ is not observable.
- Contrary to AR model, finite MA model is always stationary, because the observation is just a weighted moving average over past forecast errors.

###  ARMA Model
- Autoregressive and moving average models can be combinedtogether to form ARMA models.

$$        ARMA(p, q) \qquad : \qquad       X_t =   \omega_t + \omega_{t-1}\theta_1 +  \phi_1X_{t-1} + \omega_2\theta_2 + \phi_2X_{t-2} + \cdots + \omega_q\theta_q  + \cdots + \phi_pX_{t-p}       $$


### ARIMA Model
- Differences of order $d$ are defined as $\nabla X_t = X_t − X_{t−1} = (1-B)X_t$, and $\nabla d = (1 − B)^d$, where $(1 − B)^d$ can be expanded algebraically for higher integer values of $d$.
- The backshift operator $B$ is defined as $BX_t = X_{t−1}$. It can be extended: $B^kX_t = X_{t-k}$
- A process $X_t$ is said to be $ARIMA(p, d, q)$ if following is $ARMA(p,q)$

$$ \nabla X_t = (1 − B)^dX_t $$ 


When analyzing and forecasting the sizes of error for models, certain assumption are made. One such assumption is that, the average of all error terms squared is same at any point. This assumption is called **heteroskedasticity**, and is of focus in **ARCH/GARCH** models where the volatility of time series is modeled with mathematical formulation, in the context of financial data.



###  ARCH Model
To reflect the influence of past squared error terms in the current volatility of time series, there is ARCH model. ARCH model includes an **Autoregressive (AR)** component: like in the ARIMA (Autoregressive Integrated Moving Average) model,  current value of the series is modeled as a linear combination of its past values. 

Other component is called: **Conditional Heteroskedasticity**. The term "heteroskedasticity" refers to the situation where the variability of a time series is not constant over time. Formulation:

$$ h_t = \alpha_0 + \sum_{i=1}^p \alpha_i\epsilon^2_{t-i} $$

- $h_t$ is conditional variance at time $t$, representing the estimate of the variance of the time series at that specific time given the past information.
- $\epsilon_{t-i}^2$ is the squared error term at lag $i$, indicating the squared difference between the observed value and the predicted value at time 
-  $\alpha_0$ is a constant term in the model, representing the baseline level of conditional variance.
-  $\alpha_{t-i}$ parameters  associated with the squared error terms at different lags. These coefficients determine the impact of past squared error terms on the current conditional variance. 
- $p$ is the order of the model, representing the number of lagged squared error terms considered.




###  GARCH Model
**Generalized Autoregressive Conditional Heteroskedasticity** includes lagged values of both the conditional variance and the squared observations in the model. This allows GARCH to capture not only the autoregressive nature of volatility (as in ARCH) but also the persistence of volatility shocks.

$$ h_t = \alpha_0 + \sum_{i=1}^p \alpha_i\epsilon^2_{t-i}  +  \sum_{j=1}^q \beta_j h_{t-j} $$




## Black-Scholes-Merton Formulation
Assumptions on Underlying Asset:
  1.  Asset follows a geometric Brownian motion with constant volatility
  2.  There are no dividens or stock splits

Assumptions on the Financial Market:
  1.  It is possible to buy and sell any amount of asset at any time
  2.  Bid and ask prices are equal. i.e. the bid-ask spread is zero
  3.  There are no transaction costs or taxes
  4.  Short selling is allowed without any cost. Borrowing money is possible at any time
  5.  The risk-free interest rate is known and costant.

The option value, $V$, is depended on the time, $t$, and underlying asset (stock) price, $S$. As, time changes from $t$ to $t+\delta t$, $V$ also changes. Exapnd the $V(S, t)$ using taylor expansion of second order:

$$ \delta V = V_t\delta t + V_S\delta S + \frac{1}{2}V_{tt}(\delta t)^2 + V_{St}\delta S\delta t + \frac{1}{2}V_{SS}(\delta S)^2 $$

The increment $\delta t$ is close to 0 and we ignore any term whose order is greater than 1 by Ito's lemma. (practical rule of thumb: $(\delta t)^2 = 0$, $(\delta W)^2 = \delta t$, $\delta t\times \delta W = (\delta t)^{3/2}$).

$$ (\delta S)^2 = \sigma^2S^2(\delta W)^2 + \mathrm{higher\ order\ terms} \approx \sigma^2S^2\delta t $$

we can classify the increments as follows:

$$\delta V = (V_t + \frac{1}{2}\sigma^2 S^2 V_{SS})\delta t + V_S\delta S$$

(above, first term in parantheses is risk-free, second term is risky)

To derive a partial differential equation for $V$ we take the viewpoint of a fund
manager of the portfolio $\Pi$ based on the idea of hedging, and construct a portfolio
$\Pi$ that is self-financing and risk-free as follows:

$$\Pi (S, t) = -V(S,t) + D (S, t) +\Delta (S,t) S $$

In other words, $\Pi consists of an option that has been sold, a bank deposit or riskfree
asset $D$, and $\Delta$ shares of risky asset $S$. Here $\Delta$ is a function of $t$ and $S$, and is
called the hedge ratio. If $\Delta < 0 $, it represents short selling.

As a fund manager we maintain the same number of shares of a stock from time
$t$ to $t + \delta t$, which fits common sense. More precisely, since we do not know how
much $S$ would change, we wait until we obtain the information on the stock value at
time $t + \delta t$, and make an investment decision upon that information. Suppose that
while the hedge ratio $\Delta t$ is fixed, $S_t$ changes to $S_{t+\delta t}$, and $\Pi_t$ changes to $\Pi_{t+\delta t}$.

The risk-free asset Dt gains interest
$$ \delta D_t = rD\delta t$$

Hence:

$$ \delta Pi = -\delta V + \delta D + \Delta \delta S$$

$$ \delta Pi = -\delta V + rD\delta t + \Delta \delta S$$

$$  \delta Pi = (-V_t - \frac{1}{2}\sigma^2 S^2 V_{SS} + rD)\delta t + \left( \Delta - V_S \right)\delta S $$

If we take
	$$\Delta_t = V_S(t, S_t)$$

for every $t$, then we have
 $$ \delta\Pi = (-V_t - \frac{1}{2}\sigma^2 S^2 V_{SS} + rD)\delta t  $$

Now the $\delta S$ term has disappeared and $\delta \Pi$ is risk-free, and hence it is equivalent to a
bank deposit for a time duration $\delta t$. Thus we obtain

$$ \delta \Pi = r\Pi \delta t$$

By the no arbitrage principle:

$$  (-V_t - \frac{1}{2}\sigma^2 S^2 V_{SS} + rD)\delta t  =   r\Pi \delta  $$

Therefore

$$  \left(-V_t - \frac{1}{2}\sigma^2 S^2 V_{SS} + rD \right)\delta t = (-V + D  +V_S S )\delta t$$

and we obtain the Black–Scholes–Merton equation after canceling the $rD$ terms.

$$\boxed{\quad \frac{\delta V}{\delta t} + \frac{1}{2}  \sigma^2S^2\frac{\delta^2V}{\delta S^2} + rS\frac{\delta V}{\delta S} = rV\quad }$$


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




##  Multiperiod Bionamial Tree Method 
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




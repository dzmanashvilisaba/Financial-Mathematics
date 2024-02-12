# Financial-Mathematics





## American Put Option Demo

In this demo, the price V of an American option is considered as a function of the 
stock value S and time t, V = V(S,t). The financial parameters like strike, volatility,
etc. are assumed to be constants. The demo computes the option price for a range of discrete 
stock values and a range of discrete time values. The demo also computes the optimal exercise boundary as a function of time. The results are visualized in three figures. 

  1.  The first figure is a graph  of the American option price at the initial time. For comparison reasons, this figure also shows a graph of the corresponding European option and a graph of the payoff.
  2.  The second figure displays a surface of the  option price as a function of the stock value and time.
  3.  The third graph displays the optimal exercise boundary.






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


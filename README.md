# hurst_exponent
Calculate the Hurst exponent of a time series based on the rescaled range analysis method.

Paste time series data into the raw_data vector to calculate the Hurst exponent of a time series.
The Hurst exponent indicates the intensity of long-range dependence in the time series and overall captures volatility persistence within the stochastic process, which can be represented through variable H ⋲ (0, 1). 

The value of the Hurst exponent can characterize price movement in three scenarios:
- If H ⋲ (0, ½), each increment is positively correlated, and the movement is characterized as exhibiting short-term memory dependency
- If H = ½, each increment is considered independent from one another, and the movement is characterized as classic Brownian motion
- If H ⋲ (½, 1), each increment is negatively correlated, and the movement is characterized as exhibiting long-term memory dependency

There exist numerous methods to estimate the Hurst exponent, namely the rescaled range analysis method (simple, no underlying assumptions needed), the Higuchi method, the periodogram method, and the variance method. In this script, I used the rescaled range method to calculate the Hurst exponent, though this method has its drawbacks with its high sensitivity to short-range dependency.

Details on using the rescaled range method can be viewed at: https://en.wikipedia.org/wiki/Rescaled_range.

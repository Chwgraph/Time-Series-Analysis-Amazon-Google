# Time-Series-Analysis-Amazon-Google
Time series anlysis of stock prices of Amazon and Google

# Introduction
Amazon and Google are two world-leading companies in artificial intelligence. Thanks to their extraordinary performances in technologies, both Amazon and Google tended to become members of the most popular stocks in the US stock market. In this report, we apply techniques of time series analysis to the adjusted close prices and daily returns of these two stocks from 2008 to 2024 for the purpose of analyzing their performances theoretically. Data used in this paper is obtained from Yahoo Finance: https://finance.yahoo.com/.

# Data Preprocessing
We calculate the daily returns of these two stocks by $X(t)=\log \frac{S(t)}{S(t-1)}$ where $S(t)$ is the adjusted close price at time $t$.

# Analysis

## Stationarity and Autocorrelation
### Dickey-Fuller Test
In statistics, an augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root is present in a time series sample. The alternative hypothesis is different depending on which version of the test is used, but is usually stationarity or trend-stationarity.
It is an augmented version of the Dickey–Fuller test for a larger and more complicated set of time series models. The augmented Dickey–Fuller (ADF) statistic, used in the test, is a negative number. The more negative it is, the stronger the rejection of the hypothesis that there is a unit root at some level of confidence. See Wiki for a brief introduction: https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test.

### (Partial) Autocorrelation Function
The partial autocorrelation function (PACF) gives the partial correlation of a stationary time series with its own lagged values, regressed the values of the time series at all shorter lags. It contrasts with the autocorrelation function, which does not control for other lags. See Wiki for a brief introduction: https://en.wikipedia.org/wiki/Partial_autocorrelation_function. 

## Fractal Behavior
### Hurst Exponent
The Hurst exponent is used as a measure of long-term memory of time series. It quantifies the relative tendency of a time series either to regress strongly to the mean or to cluster in a direction. See Wiki for a brief introduction: https://en.wikipedia.org/wiki/Hurst_exponent

### Detrended Fluctuation Analysis
Detrended fluctuation analysis (DFA) is a method for determining the statistical self-affinity of a signal. The obtained exponent is similar to the Hurst exponent, except that DFA may also be applied to signals whose underlying statistics (such as mean and variance) or dynamics are non-stationary (changing with time). See Wiki for a brief introduction: https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis.

### Granger Causality
The Granger causality test is a statistical hypothesis test for determining whether one time series is useful in forecasting another. In this project we study the Granger causality between these two stocks by fitting a $\textbf{VARMA}$ model. See Wiki for a brief introduction: https://en.wikipedia.org/wiki/Granger_causality.

## Empirical Mode Decomposition
Using the EMD method, any complicated data set can be decomposed into a finite and often small number of components. These components form a complete and nearly orthogonal basis for the original signal. In addition, they can be described as intrinsic mode functions (IMF). An intrinsic mode function (IMF) is defined as a function that satisfies the following requirements:

1. In the whole data set, the number of extrema and the number of zero-crossings must either be equal or differ at most by one.
2. At any point, the mean value of the envelope defined by the local maxima and the envelope defined by the local minima is zero.

It represents a generally simple oscillatory mode as a counterpart to the simple harmonic function. 

See Wiki for a brief introduction: https://en.wikipedia.org/wiki/Hilbert%E2%80%93Huang_transform#cite_note-2.

# References
1. Mushtaq, Rizwan. "Augmented dickey fuller test." (2011).
2. Qian, Bo, and Khaled Rasheed. "Hurst exponent and financial market predictability." IASTED conference on Financial Engineering and Applications. Proceedings of the IASTED International Conference. Chicago Cambridge, MA, 2004.
3. Hu, Kun, et al. "Effect of trends on detrended fluctuation analysis." Physical Review E 64.1 (2001): 011114.
4. Zeiler, Angela, et al. "Empirical mode decomposition-an introduction." The 2010 international joint conference on neural networks (IJCNN). IEEE, 2010.

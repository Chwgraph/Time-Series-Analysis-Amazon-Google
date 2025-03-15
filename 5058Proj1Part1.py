# -*- coding: utf-8 -*-
"""
5058 Proj1

@author: 17100
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df2=pd.read_csv('C:/Users/17100/.spyder-py3/5058/data/GOOG.csv')
df4=pd.read_csv('C:/Users/17100/.spyder-py3/5058/data/AMZN.csv')
print(df2.columns)
returns=[]
returns2=[]
clos=np.array(list(df2['Adj Close']))
clos2=np.array(list(df4['Adj Close']))
for i in range(1,len(clos)):
    returns.append(np.log(clos[i]/clos[i-1]))
for i in range(1, len(clos2)):
    returns2.append(np.log(clos2[i]/clos2[i-1]))
    
x1=np.mean(returns)
x2=np.mean(returns2)
r1=np.array(returns)-x1
r2=np.array(returns2)-x2

fig=plt.figure(figsize=(16,8), dpi=200)
n=len(r1)
m=len(r2)
tline=np.linspace(0, n, n)
tline2=np.linspace(0, m, m)
plt.plot(tline, r1, color='blue', label='GOOG')
plt.plot(tline2, r2, color='r', label='AMZN')
plt.title('Daily Return')
plt.xlim(0, n)
plt.ylim(-0.2, 0.25)
plt.legend()
plt.xlabel('t')
plt.ylabel('X(t)')
plt.show()

fig=plt.figure(figsize=(16,8), dpi=200)
n=len(r1)
m=len(r2)
tline=np.linspace(0, n, n)
tline2=np.linspace(0, m, m)
plt.plot(tline, r1, color='blue')
plt.title('Daily Return of Google')
plt.xlim(0, n)
plt.ylim(-0.2, 0.25)
plt.xlabel('t')
plt.ylabel('X(t)')
plt.show()

fig=plt.figure(figsize=(16,8), dpi=200)
n=len(r1)
m=len(r2)
tline=np.linspace(0, n, n)
tline2=np.linspace(0, m, m)
plt.plot(tline2, r2, color='r')
plt.title('Daily Return of Amazon')
plt.xlim(0, n)
plt.ylim(-0.2, 0.25)
plt.xlabel('t')
plt.ylabel('X(t)')
plt.show()

fig2=plt.figure(figsize=(16,8), dpi=200)
tline3=np.linspace(0, n+1, n+1)
tline4=np.linspace(0, m+1, m+1)
plt.plot(tline3, clos, color='blue', label='GOOG')
plt.plot(tline4, clos2, color='r', label='AMZN')
plt.title('Adjusted Close Price')
plt.xlim(0, n)
plt.ylim(0, 200)
plt.legend()
plt.xlabel('t')
plt.ylabel('S(t)')
plt.show()

from arch.unitroot import ADF
adf1 = ADF(r1)
# print(adf.pvalue)
print(adf1.summary().as_text())

adf1 = ADF(clos)
# adf1.trend = 'ct'
print(adf1.summary().as_text())

adf2 = ADF(r2)
# print(adf.pvalue)
print(adf2.summary().as_text())

adf2 = ADF(clos2)
# adf2.trend = 'ct'
print(adf2.summary().as_text())

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig=plt.figure(figsize=(16,8), dpi=100)
plot_acf(r1, title='ACF of Google')
plt.xlabel('lag')
plt.ylabel('Correlation Coefficient')
plt.show()

fig2=plt.figure(figsize=(16,8), dpi=100)
plot_pacf(r1, title='PACF of Google')
plt.xlabel('lag')
plt.ylabel('Correlation Coefficient')
plt.show()

plot_acf(r2, title='ACF of Amazon')
plt.xlabel('lag')
plt.ylabel('Correlation Coefficient')
plt.show()


plot_pacf(r2, title='PACF of Amazon')
plt.xlabel('lag')
plt.ylabel('Correlation Coefficient')
plt.show()

clos3=abs(r1)
clos4=abs(r2)

plot_acf(clos3, title='ACF of Absolute Value of Google')
plt.xlabel('lag')
plt.ylabel('Correlation Coefficient')
plt.show()

plot_acf(clos4, title='ACF of Absolute Value of Amazon')
plt.xlabel('lag')
plt.ylabel('Correlation Coefficient')
plt.show()

# import statsmodels as sm
# from statsmodels.tsa.api import ARIMA
# data=r1.copy()
# data2=r2.copy()
# aic_val=[]
# for ari in range(1,5):
#     for mij in range(1,5):
#         try:
#             arma_obj=ARIMA(data, order=(ari,0,mij)).fit()
#             aic_val.append([ari,mij,arma_obj.aic])
#         except Exception as e:
#             print(e)

# print(aic_val)

# aic_val2=[]
# for ari in range(1,5):
#     for mij in range(1,5):
#         try:
#             arma_obj2=ARIMA(data2, order=(ari,0,mij)).fit()
#             aic_val2.append([ari,mij,arma_obj2.aic])
#         except Exception as e:
#             print(e)

# print(aic_val2)


def hurst(price, min_lag=3, max_lag=100):
  lags = np.arange(min_lag, max_lag + 1)
  tau = [np.std(np.subtract(price[lag:], price[:-lag])) 
    for lag in lags]
  m = np.polyfit(np.log10(lags), np.log10(tau), 1)
  return m, lags, tau

m, lag, rs=hurst(clos)
print(m[0])

m2, lag2, rs2=hurst(clos2)
print(m2[0])

fig=plt.figure(figsize=(16,8), dpi=100)
plt.plot(np.log10(lag), np.log10(rs), label='$\log$ R(n)')
plt.scatter(np.log10(lag), np.log10(rs))
plt.plot(np.log10(lag), np.log10(lag)*m[0]+m[1], label='$H \log n$')
plt.xlabel('Log of Window Size')
plt.ylabel('Log of R(n)')
plt.legend()
plt.title('Log of R(n) Against Log of Window Size For Google')

fig2=plt.figure(figsize=(16,8), dpi=100)
plt.plot(np.log10(lag2), np.log10(rs2), label='$\log$ R(n)')
plt.scatter(np.log10(lag2), np.log10(rs2))
plt.plot(np.log10(lag2), np.log10(lag2)*m2[0]+m2[1], label='$H \log n$')
plt.xlabel('Log of Window Size')
plt.ylabel('Log of R(n)')
plt.legend()
plt.title('Log of R(n) Against Log of Window Size For Amazon')


def DFA(data,ni,fittime):
    n = len(data)//ni
    nf = int(n*ni)
 
    n_mean =np.mean(data[:nf])
    y = []
    y_hat = []
    for i in range(nf):
        y.append(np.sum(data[:i+1]-n_mean))
    for i in range(int(n)):
        x = np.arange(1,ni+1,1)
        y_temp = y[int(i*ni+1)-1:int((i+1)*ni)]
        coef = np.polyfit(x,y_temp,deg=fittime)
        y_hat.append(np.polyval(coef,x))
    fn = np.sqrt(sum((np.asarray(y)-np.asarray(y_hat).reshape(-1))**2)/nf)
    return fn

lags = np.arange(3,101)


f = []
for i in range(len(lags)):
    f.append(DFA(r1,lags[i],1))

m3 = np.polyfit(np.log10(lags), np.log10(f), 1)

f2 = []
for i in range(len(lags)):
    f2.append(DFA(r2,lags[i],1))

m4 = np.polyfit(np.log10(lags), np.log10(f2), 1)


fig3=plt.figure(figsize=(16,8),dpi=100)
plt.plot(np.log10(lags),np.log10(f), label=r'$\log$ f(n)')
plt.scatter(np.log10(lags),np.log10(f))
plt.plot(np.log10(lags), np.log10(lags)*m3[0]+m3[1], label=r'$\alpha$ $\log$ n')
plt.legend()
plt.title('$\log f(n)$ Against Window Sizes For Google')
plt.xlabel('Log of Window Size')
plt.ylabel('$\log$ f(n)')
plt.show()

fig4=plt.figure(figsize=(16,8), dpi=100)
plt.plot(np.log10(lags), np.log10(f2), label=r'$\log$ f(n)')
plt.scatter(np.log10(lags), np.log10(f2))
plt.plot(np.log10(lags), np.log10(lags)*m4[0]+m4[1], label=r'$\alpha$ $\log$ n')
plt.legend()
plt.title('$\log f(n)$ Against Window Sizes For Amazon')
plt.xlabel('Log of Window Size')
plt.ylabel('$\log f(n)$')
plt.show()

print(m3[0])
print(m4[0])

y1=np.log(clos)
y2=np.log(clos2)
qarray=np.array([1,2,3, 5, 10, 15, 20, 25, 30, 35, 40])
M=[]
M2=[]
tau=np.arange(1, 100)
for q in qarray:
    Mt=np.array([np.mean([abs(y1[i+t]-y1[i])**q for i in range(0, len(y1)-t)]) for t in tau])
    Mt2=np.array([np.mean([abs(y2[i+t]-y2[i])**q for i in range(0, len(y2)-t)]) for t in tau])
    M.append(Mt)
    M2.append(Mt2)
fig5=plt.figure(figsize=(16,8), dpi=100)
plt.plot(tau, pow(M[0], 1/qarray[0]), label='q=1')
plt.plot(tau, pow(M[1], 1/qarray[1]), label='q=5')
plt.plot(tau, pow(M[2], 1/qarray[2]), label='q=10')
plt.plot(tau, pow(M[3], 1/qarray[3]), label='q=20')
plt.plot(tau, pow(M[4], 1/qarray[4]), label='q=40')
plt.scatter(tau, pow(M[0], 1/qarray[0]))
plt.scatter(tau, pow(M[1], 1/qarray[1]))
plt.scatter(tau, pow(M[2], 1/qarray[2]))
plt.scatter(tau, pow(M[3], 1/qarray[3]))
plt.scatter(tau, pow(M[4], 1/qarray[4]))
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$M(q, \tau)^{1/q}$')
plt.xlim(0,)
plt.ylim(0,)
plt.title(r'$M(q, \tau)^{1/q}$ Against $\tau$ For Google')
plt.show()


fig5=plt.figure(figsize=(16,8), dpi=100)
plt.plot(tau, pow(M2[0], 1/qarray[0]), label='q=1')
plt.plot(tau, pow(M2[1], 1/qarray[1]), label='q=5')
plt.plot(tau, pow(M2[2], 1/qarray[2]), label='q=10')
plt.plot(tau, pow(M2[3], 1/qarray[3]), label='q=20')
plt.plot(tau, pow(M2[4], 1/qarray[4]), label='q=40')
plt.scatter(tau, pow(M2[0], 1/qarray[0]))
plt.scatter(tau, pow(M2[1], 1/qarray[1]))
plt.scatter(tau, pow(M2[2], 1/qarray[2]))
plt.scatter(tau, pow(M2[3], 1/qarray[3]))
plt.scatter(tau, pow(M2[4], 1/qarray[4]))
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$M(q, \tau)^{1/q}$')
plt.ylim(0,)
plt.xlim(0,)
plt.title(r'$M(q, \tau)^{1/q}$ Against $\tau$ For Amazon')
plt.show()

FB=[]
FB2=[]

for i in range(11):
    m5 = np.polyfit(np.log10(tau), np.log10(pow(M[i], 1/qarray[i])), 1)
    m6 = np.polyfit(np.log10(tau), np.log10(pow(M2[i], 1/qarray[i])), 1)
    FB.append(m5[0])
    FB2.append(m6[0])

fig6=plt.figure(figsize=(16,8), dpi=100)
plt.plot(qarray, np.array(FB), label='GOOG')
plt.plot(qarray, np.array(FB2), label='AMAZ')
plt.scatter(qarray, np.array(FB), marker='*')
plt.scatter(qarray, np.array(FB2), marker='x')
plt.title(r'Figure For $\frac{f(q)}{q}$ Against $q$')
plt.xlabel('q')
plt.ylabel(r'$\frac{f(q)}{q}$')
plt.legend()
plt.show()
    
def MDFA(data,ni,fittime, q):
    n = len(data)//ni
    nf = int(n*ni)
 
    n_mean =np.mean(data[:nf])
    y = []
    y_hat = []
    for i in range(nf):
        y.append(np.sum(data[:i+1]-n_mean))
    for i in range(int(n)):
        x = np.arange(1,ni+1,1)
        y_temp = y[int(i*ni+1)-1:int((i+1)*ni)]
        coef = np.polyfit(x,y_temp,deg=fittime)
        y_hat.append(np.polyval(coef,x))
    fn = pow(sum((np.asarray(y)-np.asarray(y_hat).reshape(-1))**q)/nf, 1/q)
    return fn

qarray2=[1, 2,3, 4,5,8, 10, 15, 20, 25,30, 35, 40]
M4=[]
lag=[]
for q in qarray2:
    f3 = []
    for i in range(len(lags)):
        f3.append(MDFA(r1,lags[i],1, q))
    b=pd.DataFrame([], columns=['lags', 'f3'])
    b['lags']=np.log(lags)
    b['f3']=np.log(np.array(f3))
    b=b.dropna()

    m6 = np.polyfit(b['lags'], b['f3'], 1)
    M4.append(m6[0])

fig7=plt.figure(figsize=(16,8), dpi=100)
plt.plot(qarray2, M4)
plt.scatter(qarray2, M4, marker='*', color='r')
plt.xlabel('q', fontsize=20)
plt.ylabel(r'$\alpha(q)$', fontsize=20)
plt.title('Scaling Exponents of Google', fontsize=20)
plt.xlim(0,40)
plt.ylim(0, 1)
plt.show()

M5=[]
lag2=[]
for q in qarray2:
    f4 = []
    for i in range(len(lags)):
        f4.append(MDFA(r2,lags[i],1, q))
    b=pd.DataFrame([], columns=['lags', 'f3'])
    b['lags']=np.log(lags)
    b['f3']=np.log(np.array(f4))
    b=b.dropna()

    m7 = np.polyfit(b['lags'], b['f3'], 1)
    if m7[0]<0:
        m7[0]=0
    M5.append(m7[0])

fig8=plt.figure(figsize=(16,8), dpi=100)
plt.plot(qarray2, M5)
plt.scatter(qarray2, M5, marker='*', color='r')
plt.xlabel('q', fontsize=20)
plt.ylabel(r'$\alpha(q)$', fontsize=20)
plt.title('Scaling Exponents of Amazon', fontsize=20)
plt.xlim(0,40)
plt.ylim(0, 1)
plt.show()

from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import mse,rmse
from statsmodels.tsa.statespace.varmax import VARMAX,VARMAXResults
df2=pd.DataFrame([], columns=['r1', 'r2'])
df2['r1']=r1[:4026]
df2['r2']=r2


df3=pd.DataFrame([], columns=['r2', 'r1'])
df3['r1']=r1[:4026]
df3['r2']=r2
AIC2=[]
for i in range(1,6):
    for j in range(1,6):
        model = VARMAX(df2, order=(i,j), trend='c') # c indicates a constant trend
        results = model.fit(maxiter=1000, disp=False)
        AIC2.append([i,j,results.aic])
print(AIC2)

model2 = VARMAX(df2, order=(1,2), trend='c') # c indicates a constant trend
results2 = model2.fit(maxiter=1000, disp=False)

from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(df2, maxlag=4)
grangercausalitytests(df3, maxlag=4)

ffr1=np.fft.fft(r1)
ffr2=np.fft.fft(r2)
ffrr1=np.fft.fftfreq(len(r1))
ffrr2=np.fft.fftfreq(len(r2))
n1=len(r1)
n2=len(r2)
fig8=plt.figure(figsize=(16,8),dpi=100)
plt.plot(ffrr1[:n1//2], abs(ffr1)[:n1//2])
plt.xlabel('Frequency')
plt.ylabel('Magnitude of Coefficients')
plt.xlim(0,0.5)
plt.ylim(0)
plt.title('Magnitude of Coefficients of Fourier Transform of Google')

fig9=plt.figure(figsize=(16,8), dpi=100)
plt.plot(ffrr2[:n2//2], abs(ffr2)[:n2//2])
plt.xlabel('Frequency')
plt.ylabel('Magnitude of Coefficients')
plt.xlim(0,0.5)
plt.ylim(0,)
plt.title('Magnitude of Coefficients of Fourier Transform of Amazon')

from scipy import signal

freqs, psd = signal.welch(r1)
plt.figure(figsize=(16, 8))
plt.plot(freqs, psd)
plt.title("PSD of Google")
plt.xlabel("Frequency")
plt.xlim(0,0.5)
plt.ylim(0,)
plt.ylabel("Power")
plt.tight_layout()
plt.show()

freqs2, psd02 = signal.welch(r2)
plt.figure(figsize=(16, 8))
plt.plot(freqs2, psd02)
plt.title("PSD of Amazon")
plt.xlabel("Frequency")
plt.xlim(0,0.5)
plt.ylim(0,)
plt.ylabel("Power")
plt.tight_layout()
plt.show()

freqs, psd = signal.welch(r1)
m6 = np.polyfit(np.log10(freqs[1:]), np.log10(psd[1:]), 1)
plt.figure(figsize=(16, 8))
plt.loglog(freqs, psd, label='psd')
plt.loglog(freqs, (10**m6[1])*freqs**m6[0], label='Fitting Curve')
plt.title("PSD of Google")
plt.xlabel("Frequency")
plt.xlim(0,0.5)
plt.ylim(0,)
plt.ylabel("Power")
plt.legend()
plt.tight_layout()
plt.show()



freqs2, psd02 = signal.welch(r2)
m7 = np.polyfit(np.log10(freqs2[1:]), np.log10(psd02[1:]), 1)
plt.figure(figsize=(16, 8))
plt.loglog(freqs2, psd02, label='psd')
plt.loglog(freqs2, (10**m7[1])*freqs2**m7[0], label='Fitting Curve')
plt.title("PSD of Amazon")
plt.xlabel("Frequency")
plt.xlim(0,0.5)
plt.ylim(0,)
plt.legend()
plt.ylabel("Power")
plt.tight_layout()
plt.show()

import emd
t=np.arange(len(r1))
imf = emd.sift.sift(r1)
print(imf.shape)
ind=np.array([0,1,3, 5,8])
imfs=imf.T[ind].T
emd.plotting.plot_imfs(imfs)
plt.title('IMFs of Google')

imf2 = emd.sift.sift(r2)
print(imf2.shape)
ind2=np.array([0,1,3, 5,7])
imfs2=imf2.T[ind2].T
emd.plotting.plot_imfs(imfs2)
plt.title('IMFs of Amazon')

Hur1=[]
for i in range(5):
    m, lag, rs=hurst(imfs.T[i])
    if m[0]<0:
        m[0]=0
    elif m[0]>1:
        m[0]=1
    Hur1.append(m[0])
Hur2=[]
for i in range(5):
    m, lag, rs=hurst(imfs2.T[i])
    if m[0]<0:
        m[0]=0
    elif m[0]>1:
        m[0]=1
    Hur2.append(m[0])

fig2=plt.figure(figsize=(16,8), dpi=100)
plt.plot(ind+1,Hur1, label='GOOG')
plt.plot(ind2+1, Hur2, label='AMAZ')
plt.title('Hurst Exponents of Each IMF')
plt.legend()
plt.ylim(0,1)
plt.xlabel('Order')
plt.ylabel('Hurst Exponent')


fr1, psd1=signal.welch(imfs.T[0])
fr2, psd2=signal.welch(imfs.T[1])
fig3=plt.figure(figsize=(16,8), dpi=100)
plt.plot(fr1, psd1, label='IMF 1')
plt.plot(fr2, psd2, label='IMF 2')
plt.legend()
plt.ylabel('Power')
plt.xlabel('Frequency')
plt.xlim(0,0.5)
plt.ylim(0,)
plt.title('PSD of First Two IMFs of Google')

fr3, psd3=signal.welch(imfs2.T[0])
fr4, psd4=signal.welch(imfs2.T[1])
fig4=plt.figure(figsize=(16,8), dpi=100)
plt.plot(fr3, psd3, label='IMF 1')
plt.plot(fr4, psd4, label='IMF 2')
plt.legend()
plt.ylabel('Power')
plt.xlabel('Frequency')
plt.xlim(0,0.5)
plt.ylim(0,)
plt.title('PSD of First Two IMFs of Amazon')


r3=r1-imfs.T[0]
r4=r1-imfs.T[1]-imfs.T[0]
r5=r2-imfs2.T[0]
r6=r2-imfs2.T[1]-imfs2.T[0]

fr1, psd1=signal.welch(r3)
fr2, psd2=signal.welch(r4)
fig3=plt.figure(figsize=(16,8), dpi=100)
plt.plot(fr1, psd1, label=r'X-$c_1$')
plt.plot(fr2, psd2, label=r'X-$c_1$-$c_2$')
plt.plot(freqs, psd, label='X')
plt.legend()
plt.ylabel('Power')
plt.xlabel('Frequency')
plt.xlim(0,0.5)
plt.ylim(0,)
plt.title('PSD Comparision of Google')

fr3, psd3=signal.welch(r5)
fr4, psd4=signal.welch(r6)
fig4=plt.figure(figsize=(16,8), dpi=100)
plt.plot(fr3, psd3, label=r'X-$c_1$')
plt.plot(fr4, psd4, label=r'X-$c_1$-$c_2$')
plt.plot(freqs2, psd02, label='X')
plt.legend()
plt.ylabel('Power')
plt.xlabel('Frequency')
plt.xlim(0,0.5)
plt.ylim(0,)
plt.title('PSD Comparision of Amazon')


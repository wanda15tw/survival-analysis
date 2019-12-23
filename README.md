
# Predictive Maintenance -- Survival Analysis 

###### tags: `tech blog`, `tutorials`, `predictive maintenance`, `python`, `lifelines`
[complete notebook](https://github.com/wanda15tw/survival-analysis/blob/master/1223%20Survival%20Analysis%20-%20confidential%20removed.ipynb)

**Predictive maintenance** is to predict *which machinery at which condition needs preventative maintenance* so as to eliminate outages and the costs associated with it. Instead of predicting each individual part's failure, **Survival analysis** is a statistics approach to estimate failure rate. It is also called **reliability analysis** and **event history analysis**. Simply put, it intends to answer -- "How long until an event occurs?"

For example, 
* How long patients survive?
* How long mechanical parts last?

In this tutorial, we exemplify the second question with 28,529 censored parts which had no failure history, 1334 failed parts and their time to event (fail or exit the censorship). Input table is prepared as below with `event` and `time_to_event` columns.

![input_table](https://i.imgur.com/tUZJwqi.png)

```
df.event.value_counts()

# censor    28529
# fail       1334
# Name: event, dtype: int64
```
## Kaplan-Meier Estimation
At each **t + 1**, compute how many **at risk**, which is the last entrance substracted by parts failed at **t** and also those did not fail but monitored until **t** (i.e. # censored) 
```
from lifelines import KaplanMeierFitter

durations = df['time_to_event'].apply(lambda x: 0.5 if x==0 else x) # workarounds for DOA
event_observed = df['event'].apply(lambda x: 1 if x=='fail' else 0)

km = KaplanMeierFitter()

km.fit(durations, event_observed, label='KM')
```
* Event Table:

`km.event_table`

![event_table](https://i.imgur.com/r9daMYo.png)




## Survival Function
* What fraction survive past t?
* e.g., 5-yr survival rates, median survival time
* Also called **Reliability Function**

![survival function](https://i.imgur.com/OBTq3Nc.png)

### Estimated by event table 
Survival function can be estimated by the following formula where $d_i$ stands for number of defects/observed and $n_i$ is number of parts entering period $i$ / **at risk**.
![km survival function](https://i.imgur.com/gu15mJ6.png)

`km.survival_function_`

![km.survival_function](https://i.imgur.com/SeKYnVm.png)

## Hazard Function
* Of the people who survive until t, what fraction die at t?
* Conditional density probability

$$
\lambda(t) = {f(y)\over S(t)}
$$

* The hazard function might be of more intrinsic interest than the
p.d.f. to a patient who had survived a certain time period and wanted to
know something about their prognosis.

## Popular Distributions
### Weibull

```
from lifelines import * 

wbf = WeibullFitter().fit(durations[1:], event_observed[1:], label='WeibullFitter')

wbf.summary
```
![](https://i.imgur.com/FZpn2YZ.png)

`wbf.plot_hazard()`
![weibull_hazard](https://i.imgur.com/Tu84f6d.png)

```
plt.figure(figsize=(10, 24))
plt.subplot(4, 1, 1)
wbf.plot_survival_function()
plt.title('Survival Curve')
plt.xlabel('Time to Event (days)')
plt.ylabel('Survival Probability')

plt.subplot(4, 1, 2)
wbf.plot_cumulative_density()
plt.title('Cumulative Failure Density')
plt.xlabel('Time to Event (days)')
plt.ylabel('Cum. Density Prob. (failure)')


plt.subplot(4, 1, 3)
wbf.plot_hazard()
plt.title('Hazard Function')
plt.xlabel('Time to Event (days)')
plt.ylabel('Hazard Function (pdf)')

plt.subplot(4, 1, 4)
wbf.plot_cumulative_hazard()
plt.title('Cumulative Hazard Function')
plt.xlabel('Time to Event (days)')
plt.ylabel('Cumulative Hazard Function')

plt.show()
```
![](https://i.imgur.com/vMpKXpj.png)


### QQPlot

In fact, weibull distribution fits our data poortly, and we can observe that by looking at qqplot as below. None falls on the diagonal line. Instead, Log Normal seems to be a better fit.

```
from lifelines.plotting import qq_plot

durations = df['time_to_event'].apply(lambda x: 0.5 if x==0 else x)
event_observed = df['event'].apply(lambda x: 1 if x=='fail' else 0)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.reshape(4, )

for i, model in enumerate([WeibullFitter(), LogNormalFitter(), LogLogisticFitter(), ExponentialFitter()]):
    model.fit(durations, event_observed)
    qq_plot(model, ax=axes[i])
```
![qq_plot](https://i.imgur.com/9Iq471K.png)

### LogNormal

```
lnf = LogNormalFitter().fit(durations, event_observed)
lnf.summary
```
![](https://i.imgur.com/63svVIc.png)

```
plt.figure(figsize=(10, 24))
plt.subplot(4, 1, 1)
lnf.plot_survival_function()
plt.title('Survival Curve')
plt.xlabel('Time to Event (days)')
plt.ylabel('Survival Probability')

plt.subplot(4, 1, 2)
lnf.plot_cumulative_density()
plt.title('Cumulative Failure Density')
plt.xlabel('Time to Event (days)')
plt.ylabel('Cum. Density Prob. (failure)')


plt.subplot(4, 1, 3)
lnf.plot_hazard()
plt.title('Hazard Function')
plt.xlabel('Time to Event (days)')
plt.ylabel('Hazard Function (pdf)')

plt.subplot(4, 1, 4)
lnf.plot_cumulative_hazard()
plt.title('Cumulative Hazard Function')
plt.xlabel('Time to Event (days)')
plt.ylabel('Cumulative Hazard Function')

plt.show()
```
![](https://i.imgur.com/yIc1vip.png)

### Poisson 

```
from scipy.stats import poisson

mu = num_active_parts * AFR
x = np.arange(0, 1000, 1)
plt.plot(x, poisson.pmf(x, mu))
plt.title('Number of failures occur in a year in Prob.')
```
![](https://i.imgur.com/NLDXu3L.png)
```
plt.plot(x, poisson.cdf(x, mu))
plt.title('Cumu. Demand Distribution (Probability of number of failures less than x)')
plt.xlabel('x (demand)')
plt.show()
```
![](https://i.imgur.com/1BuH6tk.png)
```
i = np.arange(0.01, 1, 0.01)
plt.plot(i, poisson.isf(1-i, mu)) # cdf modified from isf, for some reason, there is no inverse CDF, but inverse survival function
plt.title('Inverse CDF')
plt.show()
```
![](https://i.imgur.com/ooJdvsh.png)


## (Bonus) Spares Demand Forecast using Newsvendor Model

### Newsvendor Model
The optimal spare quantiy given underage cost $C_u$ and overage cost $C_o$ is to minimize expected overall cost:
To minize:
$$
Expected\ Overage\ Cost + Expected\ Underage\ Cost\ \\
= E(Demand<Q)\times C_o + E(Demand>Q) \times C_u \\
$$

Equals to derivative = 0, equals to
$$
P(Demand<Q*) = {C_u\over C_u+C_o} \\
==Q* = F^{-1}({C_u \over C_u+C_o})
$$

## Results
```
SLAs = [0.50, 0.75, 0.95, 0.96, 0.97, 0.98, 0.99, 0.9999999, 1]
print('{:<30}{:<40}'.format('Reliability', '# Spares'))
for sla in SLAs:
    print('{:<30}{:<40}'.format(sla, poisson.isf(1-sla, mu)))
```
![](https://i.imgur.com/ypUvju4.png)
This simplified model (constant failure rate = 0.0003 per day / AFR = 203 per year) fitted in poisson distribution recommends 237 spares in a year to achieve 99% reliability or when ${C_u\over C_u+C_o} = 0.99$.


## Reference
* https://www.youtube.com/watch?v=XHYFNraQEEo
* https://towardsdatascience.com/survival-analysis-intuition-implementation-in-python-504fde4fcf8e
* https://lifelines.readthedocs.io/en/latest/Quickstart.html

# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fitter import Fitter
import collections as ct
import scipy.stats as ss
from scipy import stats
from scipy.stats import norm
from scipy.stats import burr
import scipy.stats as ss


distribution_car_price = pd.read_csv('/Users/gordontsai/googledrive/projects/weg/transportation/data/CarPriceDistribution 20170305gt Master.csv', header= 0)
distribution_car_price = pd.DataFrame(distribution_car_price)
distribution_car_price = distribution_car_price.iloc[:,2]
distribution_car_price = distribution_car_price[np.isfinite(distribution_car_price)]
data = distribution_car_price
data = data[np.isfinite(data)]

f = Fitter(data, timeout= 2400)


# f.hist()
# print data.isnull().values.any()
# print data.isnull().sum().sum()



f.distributions
f.fit()
# may take some time since by default, all distributions are tried
# but you call manually provide a smaller set of distributions



f.summary()



# f.fitted_param['dgamma']v
# f.fitted_pdf['dgamma']
# f.summary()


a = f.fitted_param['nct']
b = f.fitted_param['genlogistic']
c = f.fitted_param['gumbel_r']
d = f.fitted_param['f']
e = f.fitted_param['johnsonsu']


#Print distributions that we fit
print a
print b
print c
print d
print e



carprice_model_fit = ct.OrderedDict()
carprice_model_fit['nct'] = [7.3139456577106312, 3.7415255108348946, -46.285705145385577, 7917.0860181436065]
carprice_model_fit['genlogistic'] = [10.736440967148635, 3735.7049978006107, 10095.421377235754]
carprice_model_fit['gumbel_r'] = [26995.077239517472, 10774.370808211244]
carprice_model_fit['f'] = [24168.523476867485, 35.805656864712923, -21087.314142557225, 51154.0328397044]
carprice_model_fit['johnsonsu'] = [-1.7479864366935538, 1.8675670208081987, 14796.793096897647, 14716.575397771712]




a = [7.3139456577106312, 3.7415255108348946, -46.285705145385577, 7917.0860181436065]
b = [10.736440967148635, 3735.7049978006107, 10095.421377235754]




plt.hist(distribution_car_price, bins = [i for i in range(1, 50, 5)])




plt.hist(distribution_car_price, normed = True,bins = 1000)
plt.show()


#a = f.get_best()
#var = []

# for i in range(0,len(list(a.values())[0])):
#     var.append(list(a.values())[0][i]) #Returns tuple in zero index

distribution = st.gennorm.pdf(a[0], a[1],a[2],a[3])

#plot(distribution)

x = [i for i in range(0, 121, 5)]
#plt(x,distribution, 'k-')
plt.hist(distribution_car_price, bins = [i for i in range(1, 50, 5)])

plt.show()



fig, ax = plt.subplots(1, 1)
c, d = 10.5, 4.3
mean, var, skew, kurt = burr.stats(c, d, moments='mvsk')

x = np.linspace(burr.ppf(0.01, c, d),
                burr.ppf(0.99, c, d), 100)
burr.ppf



print(stats.norm.__doc__)



alpha, loc, beta=b[0], b[1], b[2]
pdf = nct.pdf()
data=ss.genlogistic.rvs(alpha,loc=loc,scale=beta,size=5000)
myHist = plt.hist(distribution_car_price, 500, normed=True)
rv = ss.genlogistic(alpha,loc,beta)
x = np.linspace(0,500000)
h = plt.plot(x, rv.pdf(x), lw=2)

axes = plt.gca()
axes.set_xlim([0,150000])
plt.show()


alpha, loc, beta=b[0], b[1], b[2]
data=ss.genlogistic.rvs(alpha,loc=loc,scale=beta,size=10000)
myHist = plt.hist(distribution_car_price, 500, normed=True)
rv = ss.nct(a[0],a[1],a[2],a[3])
x = np.linspace(0,500000)
h = plt.plot(x, rv.pdf(x), lw=2)

axes = plt.gca()
axes.set_xlim([0,150000])
plt.show()


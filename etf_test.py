
import pandas as pd
import os
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import matplotlib.ticker as mtick

os.chdir('/home/andrey/Документы/RL/swarm')
from swarm_example import Space, Particle

# iShares S&P 500 Value ETF
ive = pd.read_csv('IVE.csv', usecols = ['Date', 'Adj Close'], index_col = 'Date')
ive.rename(columns = {'Adj Close':'IVE'}, inplace= True)

# iShares Edge MSCI USA Momentum Factor ETF
mtum = pd.read_csv('MTUM.csv', usecols = ['Date', 'Adj Close'], index_col = 'Date')
mtum.rename(columns = {'Adj Close':'MTUM'}, inplace= True)

# iShares Core MSCI Emerging Markets ETF (USD)
iemg = pd.read_csv('IEMG.csv', usecols = ['Date', 'Adj Close'], index_col = 'Date')
iemg.rename(columns = {'Adj Close':'IEMG'}, inplace= True)

# iShares Core U.S. Aggregate Bond ETF (USD)
agg= pd.read_csv('IEMG.csv', usecols = ['Date', 'Adj Close'], index_col = 'Date')
agg.rename(columns = {'Adj Close':'AGG'}, inplace= True)

# iShares Core Growth Allocation ETF
aor = pd.read_csv('AOR.csv', usecols = ['Date', 'Adj Close'], index_col = 'Date')
aor.rename(columns = {'Adj Close':'AOR'}, inplace= True)

etf_list = [ive, mtum, iemg, agg, aor]

final = etf_list[0].join(etf_list[1:],  how = 'inner')
final = final.pct_change()
final.dropna(inplace=True)

# select year

df = final.filter(like= '2017', axis = 0)
cov_matrix = np.cov(df, rowvar=False)*df.shape[0]
returns = (df+1).cumprod(axis=0).iloc[-1,:].values-1


dic_pos = {}
best_values, sharpes, disp = [], [], []

for a in range(0,100,1):
    alpha = a/100
    search_space = Space(50, 5, returns, cov_matrix, alpha, w=0.5, c1=0.8, c2=0.9)
    particles_vector = [Particle(search_space.n_stocks) for _ in range(search_space.n_particles)]
    search_space.particles = particles_vector
    n_iterations = 1000
        # algo
    iteration = 0
    prev_gbest_value = -10**4
    while(iteration < n_iterations):
        search_space.set_pbest()    
        search_space.set_gbest()

        if(abs(search_space.gbest_value - prev_gbest_value) < 0.0001) and (iteration>3):
            break
        prev_gbest_value = search_space.gbest_value
        print('Iteration: {0} Position {1}, Value: {2:.3f}'.format(iteration, 
                                                                search_space.gbest_position, search_space.gbest_value))
        search_space.move_particles()
        iteration += 1
    print('finished {} ---------'.format(alpha))
        
    best_values.append(search_space.gbest_value)
    sharpes.append(search_space.sharpe)
    disp.append(search_space.vol_dis)
    dic_pos[a] = search_space.gbest_position


a = np.array(list(dic_pos.values()))
ive = a[:,0]
mtum = a[:,1]
iemg = a[:,2]
agg = a[:,3]
aor = a[:,4]
etf_list = [ive, mtum, iemg, agg, aor]
etf_list_sf = [savgol_filter(i, 7, 3) for i in etf_list]
total_sf = np.array(etf_list_sf)
total = np.array(etf_list)
# separate window
%matplotlib auto 

# plt.stackplot(range(0,100),total, labels=['ive','mtum','iemg', 'agg', 'aor'])

fig = plt.figure(1, (14,7))
ax = fig.add_subplot(1,1,1)

ax.stackplot(np.linspace(0,1,100),total*100, labels=['ive','mtum','iemg', 'agg', 'aor'])
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), ncol=1, fontsize=12)
plt.xlabel('Alpha', fontsize=15)
plt.ylabel('% Portfolio', fontsize=15)

fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)

ax.margins(0)
plt.title('Portfolio weighs transformation', pad=10, fontsize=20)
plt.style.use('seaborn-darkgrid')
plt.show()

# need to calculate sharpe ratio and 
plt.plot(best_values)

# Several graphs
fig = plt.figure(1,(17,8))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
# Sharpe
ax1.plot(np.linspace(0,1,100), sharpes, label='original SR')
ax1.plot(np.linspace(0,1,100), savgol_filter(sharpes, 7, 3), linestyle ='--', label='smooth SR')
ax1.legend(loc='upper left')
ax1.set_xlabel('Alpha', fontsize=15)
ax1.set_ylabel('Sharpe coef.', fontsize=15)
ax1.set_title('Sharpe ratios', pad = 10, fontsize=20)
# Vol dispersion
ax2.plot(np.linspace(0,1,100), disp, label='original dispersion')
ax2.plot(np.linspace(0,1,100), savgol_filter(disp, 7, 3), linestyle ='--', label='smooth dispersion')
ax2.legend(loc='upper left')
ax2.set_xlabel('Alpha', fontsize=15)
ax2.set_ylabel('Volatility coef.', fontsize=15)
ax2.set_title('Volatility dispersion', pad = 10, fontsize=20)
# optimization coef.
ax3.plot(np.linspace(0,1,100), best_values, label = 'values')
ax3.legend(loc='upper left')
ax3.set_xlabel('Alpha', fontsize=15)
ax3.set_ylabel('Best values', fontsize=15)
ax3.set_title('Best values', pad = 10, fontsize=20)

plt.tight_layout()
plt.style.use('seaborn-darkgrid')
plt.show()



#diffrent portfolios

pos = [1,5,9,13,17]

fig = plt.figure(1,(15,8))
ax = fig.add_subplot(1,1,1)
_ = ax.bar(pos, total[:,39]*100, color='blue', label='blended')

for i,v in zip(pos, total[:,39]*100):
    ax.text(i, v+2, "{:.2f}%".format(v), ha ='center', color = 'blue', fontsize = 13)

_ = ax.bar([i+1 for i in pos], total[:,0]*100, color='darkcyan', label='risk-parity')

for i,v in zip([i+1 for i in pos], total[:,0]*100):
    ax.text(i, v+2, "{:.2f}%".format(v), ha ='center', color = 'darkcyan', fontsize = 13)

_ = ax.bar([i+2 for i in pos], total[:,-1]*100, color = 'darkviolet', label = 'efficeint')

for i,v in zip([i+2 for i in pos], total[:,-1]*100):
    ax.text(i, v+2, "{:.2f}%".format(v), ha ='center', color = 'darkviolet', fontsize = 13)

ax.set_xticks([i+1 for i in pos])
ax.set_xticklabels(('ive','mtum','iemg', 'agg', 'aor'), fontsize = 13)
ax.set_xlabel('Assets', fontsize=18)

fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)
ax.set_ylabel('Weights, %', fontsize = 18)

ax.legend(loc='upper left')
ax.set_title('Portfolios comparison', pad = 10, fontsize=20)
plt.style.use('seaborn-darkgrid')
plt.tight_layout()
plt.show()



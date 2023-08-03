# Test file: the 2D case
import numpy as np
from algorithms.Learn_EnergyFucntion import LearnEnergyFunction
from algorithms.Learn_ADS import LearnAds
import matplotlib.pyplot as plt

np.random.seed(5)

# ------------------------ Loading dataSet -----------------------------
print('Loading dataSet ...')
# changing different learning cases:
# 2D_circle, 2D_rectangle, 2D_star1, 2D_star2
type = '2D_circle'
data_set = np.loadtxt('dataSet/dataSet_' + type + '.txt')
start_index = 400
gap = 2
X = data_set[start_index::gap, 0:2]
Mean_X = np.average(X, 0)
X = X - Mean_X
Y = data_set[start_index::gap, 3:5]
length = np.shape(X)[0]
print('DataSet loading completed')

# ------------------------ Learning energy function -----------------------------
alpha = 1.0
c = 1.0
likelihood_noise = 0.02
if type is '2D_star1':
    likelihood_noise = 0.2
Ef = LearnEnergyFunction(X=X, alpha=alpha, c=c, likelihood_noise=likelihood_noise)
print('Learning (Loading) data-driven energy function ...')
energy_function_path = 'model_parameters/' + type + '_Ef_params.txt'
# Training energy function
Ef.train(path=energy_function_path)
'''
# Loading parameters
Ef_param = np.loadtxt(energy_function_path)
Ef.gp.set_param(Ef_param)
'''
print('Energy function learning (Loading) completed')

'''
print('Plotting energy function ...')
area = {'x_min': -0.20, 'x_max': 0.16, 'y_min': -0.3, 'y_max': 0.3, 'step': 0.005}
Ef.plot_V(area=area)
print('Energy function plotting completed')
'''

# --------------------------- Learning the ADS --------------------------------
b = 1
max_v = 0.1
likelihood_noise = 0.05
ads = LearnAds(X=X, Y=Y, b=b, max_v=max_v, likelihood_noise=likelihood_noise)
print('Learning (Loading) the ADS ...')
oads_function_path = 'model_parameters/' + type + '_oads_params.txt'

# Training oads
ads.train_original_ads()
ads.original_ads.save_param(oads_function_path)
'''
# Loading parameters
oads_param = np.loadtxt(oads_function_path)
ads.original_ads.set_param(oads_param)
'''
print('ADS learning (Loading) completed')
# ------------------------------ Running the ADS ----------------------------------
Maxsteps = 2000
Period = 0.01
collect_data = []
training_options = {'feastol': 1e-9, 'abstol': 1e-9, 'reltol': 1e-9, 'maxiters': 50, 'show_progress': False}

off_set = np.array([-0.18, 0.0])
x = X[0, :] + off_set
x_list = [x]
V_list = [Ef.V(x)]
v_list = []
print('ADS Reproduction ...')
for step in range(Maxsteps):
    o_x_dot, u = ads.ads_evolution(x=x, lf=Ef, training_options=training_options)
    x = x + (o_x_dot + u) * Period
    x_list.append(x)
    V_list.append(Ef.V(x))
    v_list.append(np.hstack((o_x_dot, u)))
print('ADS reproduction completed')

print('Plotting the reproduction results ...')
x_list = np.array(x_list)
V_list = np.array(V_list)
v_list = np.array(v_list)
plt.plot(x_list[:, 0], x_list[:, 1], c='red')
area = {'x_min': -0.20, 'x_max': 0.16, 'y_min': -0.3, 'y_max': 0.3, 'step': 0.005}
Ef.plot_V(area=area, handle=plt)
plt.show()
length = np.shape(V_list)[0]
plt.plot(np.arange(length), V_list)
plt.show()
print('Plotting completed, exit the program')

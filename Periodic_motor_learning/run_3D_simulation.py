import numpy as np
from algorithms.Learn_EnergyFucntion import LearnEnergyFunction
from algorithms.Learn_ADS import LearnAds
from algorithms.GPR import sgpr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
np.random.seed(5)

# ------------------------ Loading dataSet -----------------------------
print('Loading dataSet ...')
# changing different learning cases:
# 3D_tra1, 3D_tra2
type = '3D_tra2'
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
Ef = LearnEnergyFunction(X=X, alpha=alpha, c=c, likelihood_noise=likelihood_noise)
energy_function_path = 'model_parameters/' + type + '_Ef_params.txt'
print('Learning (Loading) data-driven energy function ...')
'''
# Training energy function
Ef.train(path=energy_function_path)
'''
# Loading parameters
Ef_param = np.loadtxt(energy_function_path)
Ef.gp.set_param(Ef_param)

print('Energy function learning (Loading) completed')

print('Plotting energy function ...')
area = {'x_min': -0.20, 'x_max': 0.16, 'y_min': -0.3, 'y_max': 0.3, 'step': 0.005}
Ef.plot_V(area=area)
print('Energy function plotting completed')

# --------------------------- Learning the ADS --------------------------------
b = 1
max_v = 0.1
likelihood_noise = 0.05
ads = LearnAds(X=X, Y=Y, b=b, max_v=max_v, likelihood_noise=likelihood_noise)
print('Learning (Loading) the ADS ...')
oads_function_path = 'model_parameters/' + type + '_oads_params.txt'
'''
# Training oads
ads.train_original_ads()
ads.original_ads.save_param(oads_function_path)
'''

# Loading parameters
oads_param = np.loadtxt(oads_function_path)
ads.original_ads.set_param(oads_param)


Z = data_set[start_index::gap, 2]
Mean_Z = np.average(Z)
Z = Z - Mean_Z
likelihood_noise = 0.05
z_predictor = sgpr(X=X, y=Z, likelihood_noise=likelihood_noise)
z_predictor_path = 'model_parameters/' + type + '_z_predictor_params.txt'

'''
# Training z_predictor
z_predictor.train()
z_predictor.save_param(z_predictor_path)
'''

# Loading parameters
z_predictor_param = np.loadtxt(z_predictor_path)
z_predictor.set_param(z_predictor_param)

print('ADS learning (Loading) completed')

# ---------------------------- Running the ADS ----------------------------------
Maxsteps = 2000
Period = 0.01
collect_data = []
training_options = {'feastol': 1e-9, 'abstol': 1e-9, 'reltol': 1e-9, 'maxiters': 50, 'show_progress': False}


# exp_data = np.loadtxt('exp_data_' + type + '.txt')
x = data_set[10, 0:2] - Mean_X
z = data_set[10, 2] - Mean_Z
x_list = [x]
z_list = [z]
V_list = [Ef.V(x)]
v_list = []
gain_z = 1.0
print('ADS Reproduction ...')
for step in range(Maxsteps):
    o_x_dot, u = ads.ads_evolution(x=x, lf=Ef, training_options=training_options)
    z_pre, _ = z_predictor.predict_determined_input(x.reshape(1, -1))
    z_pre = z_pre.reshape(-1)
    z_dot = -gain_z * (z - z_pre) + z_predictor.gradient2input(x).dot(o_x_dot + u)
    z = z + z_dot * Period
    x = x + (o_x_dot + u) * Period
    x_list.append(x)
    z_list.append(z)
    V_list.append(Ef.V(x))
    v_list.append(np.hstack((o_x_dot + u, z_dot)))
print('ADS reproduction completed')

print('Plotting the reproduction results ...')
x_list = np.array(x_list)
z_list = np.array(z_list)
V_list = np.array(V_list)
v_list = np.array(v_list)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x_list[:, 0], x_list[:, 1], z_list, c='red')
ax.scatter3D(X[0::3, 0], X[0::3, 1], Z[0::3], c='blue', alpha=1.0, s=10, marker='x')
plt.show()
print('Plotting completed, exit the program')



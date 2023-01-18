import matplotlib.pyplot as plt
import numpy as np

#%% OOD Detector AUPR

baseline_aupr = [(0.8432+0.8493+0.8538)/3,
                 (0.9085+0.9233+0.9152)/3,
                 (0.9074+0.9233+0.9132)/3,
                 (0.8249+0.8432+0.8508)/3,
                 (0.9409+0.9128+0.8915)/3,
                 (0.9744+0.9441+0.9682)/3,
                 (0.9310+0.9265+0.9714)/3]

odin_aupr = [(0.8528+0.8609+0.8628)/3,
             (0.9390+0.9254+0.9301)/3,
             (0.9505+0.9455+0.9472)/3,
             (0.8682+0.8855+0.8832)/3,
             (0.9831+0.9640+0.9559)/3,
             (0.9823+0.9588+0.9502)/3,
             (0.9245+0.9406+0.9416)/3]

energy_aupr = [(0.8517+0.8599+0.8617)/3,
               (0.9407+0.9184+0.9279)/3,
               (0.9516+0.9462+0.9510)/3,
               (0.8697+0.8864+0.8845)/3,
               (0.9874+0.9676+0.9616)/3,
               (0.9789+0.9568+0.9374)/3,
               (0.9188+0.9387+0.9253)/3]

x = np.arange(7)
y1 = baseline_aupr
y2 = odin_aupr
y3 = energy_aupr
width = 0.2

plt.figure(dpi=200)
plt.bar(x-0.2, y1, width)
plt.bar(x, y2, width)
plt.bar(x+0.2, y3, width)
plt.xticks(x, ['CIFAR-100', 'SVHN', 'LSUN-C', 'LSUN-R', 'MNIST', 'Uniform', 'Gaussian'])
plt.title('OOD Detector Performance (AUPR)')
plt.xlabel('OOD Datasets')
plt.ylabel('AUPR')
plt.legend(['Baseline', 'ODIN', 'Energy'], loc='lower right')
plt.show()

#%% OOD Detector FPR@95TPR

baseline_fpr95tpr = [(0.7294+0.7412+0.7431)/3,
                     (0.7279+0.6369+0.7015)/3,
                     (0.6125+0.6369+0.6248)/3,
                     (0.7598+0.739+0.7372)/3,
                     (0.4551+0.5961+0.6902)/3,
                     (0.2396+0.8075+0.4338)/3,
                     (0.7725+0.8905+0.2713)/3]

odin_fpr95tpr = [(0.6610+0.6732+0.6675)/3,
                 (0.5057+0.612+0.5859)/3,
                 (0.3337+0.3606+0.3598)/3,
                 (0.6443+0.5860+0.5841)/3,
                 (0.1294+0.2849+0.3896)/3,
                 (0.0888+0.8075+0.9991)/3,
                 (0.9869+0.9836+0.9971)/3]

energy_fpr95tpr = [(0.6538+0.6718+0.6674)/3,
                   (0.4904+0.6587+0.6065)/3,
                   (0.3041+0.3581+0.3344)/3,
                   (0.6325+0.5692+0.5673)/3,
                   (0.0763+0.2498+0.3322)/3,
                   (0.1566+0.9508+0.9999)/3,
                   (0.9980+0.9982+0.9999)/3]

x = np.arange(7)
y1 = baseline_fpr95tpr
y2 = odin_fpr95tpr
y3 = energy_fpr95tpr
width = 0.2

plt.figure(dpi=200)
plt.bar(x-0.2, y1, width)
plt.bar(x, y2, width)
plt.bar(x+0.2, y3, width)
plt.xticks(x, ['CIFAR-100', 'SVHN', 'LSUN-C', 'LSUN-R', 'MNIST', 'Uniform', 'Gaussian'])
plt.title('OOD Detector Performance (FPR@95TPR)')
plt.xlabel('OOD Datasets')
plt.ylabel('FPR@95TPR')
plt.legend(['Baseline', 'ODIN', 'Energy'], loc='lower right')
plt.show()

#%% Outlier Exposure AUPR

cifar_aupr = [0.5438, 0.6492, 0.6122, 0.7932, 0.8845, 0.8809]
svhn_aupr = [0.8457, 0.2971, 0.6281, 0.8368, 0.8855, 0.9668]
lsun_aupr = [0.8676, 0.3966, 0.5987, 0.6872, 0.8740, 0.9296]
mnist_aupr = [0.8634, 0.8015, 0.6308, 0.3763, 0.7661, 0.8332]
uniform_aupr = [0.8036, 0.7120, 0.6285, 0.8152, 0.9743, 0.9992]
gaussian_aupr = [0.7572, 0.3113, 0.6511, 0.8472, 0.9917, 0.9991]

x = ['0', '1', '5', '20', '50', '100']
y1 = cifar_aupr
y2 = svhn_aupr
y3 = lsun_aupr
y4 = mnist_aupr
y5 = uniform_aupr
y6 = gaussian_aupr

plt.figure(dpi=200)
plt.plot(x, y1, label='CIFAR-10 (6-10)')
plt.plot(x, y2, label='SVHN')
plt.plot(x, y3, label='LSUN')
plt.plot(x, y4, label='MNIST')
plt.plot(x, y5, label='Uniform-Random')
plt.plot(x, y6, label='Gaussian-Random')
plt.title('Outlier Exposure Performance (AUPR)')
plt.xlabel('Outlier Diversity (CIFAR-100)')
plt.ylabel('AUPR')
plt.legend(loc='lower right')
plt.show()

#%% Outlier Exposure FPR@95TPR

cifar_fpr95tpr = [0.9722, 0.9517, 0.9042, 0.6785, 0.6435, 0.6480]
svhn_fpr95tpr = [0.7547, 0.9092, 0.9261, 0.3919, 0.4712, 0.0797]
lsun_fpr95tpr = [0.5801, 0.8629, 0.8170, 0.6618, 0.4938, 0.2112]
mnist_fpr95tpr = [0.6403, 0.8160, 0.9005, 0.9560, 0.8072, 0.6824]
uniform_fpr95tpr = [0.9339, 1.0000, 0.9999, 0.9999, 0.0876, 0.0000]
gaussian_fpr95tpr = [0.9370, 1.0000, 0.9985, 0.9984, 0.0143, 0.0000]

x = ['0', '1', '5', '20', '50', '100']
y1 = cifar_fpr95tpr
y2 = svhn_fpr95tpr
y3 = lsun_fpr95tpr
y4 = mnist_fpr95tpr
y5 = uniform_fpr95tpr
y6 = gaussian_fpr95tpr

plt.figure(dpi=200)
plt.plot(x, y1, label='CIFAR-10 (6-10)')
plt.plot(x, y2, label='SVHN')
plt.plot(x, y3, label='LSUN')
plt.plot(x, y4, label='MNIST')
plt.plot(x, y5, label='Uniform-Random')
plt.plot(x, y6, label='Gaussian-Random')
plt.title('Outlier Exposure Performance (FPR@95TPR)')
plt.xlabel('Outlier Diversity (CIFAR-100)')
plt.ylabel('FPR@95TPR')
plt.legend(loc='lower left')
plt.show()

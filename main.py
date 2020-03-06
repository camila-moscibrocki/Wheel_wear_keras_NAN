import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.contrib.rnn import BasicRNNCell

# Lê os dados e ordena as medições de rodas por datas conforme coluna "Measurement date" da tabela
lcc_data = pd.read_csv('lcc.csv')
lcc_data.sort_values(["Measurement date"], inplace=True)

# Seleciona os dados das colunas de taxa de desgaste de aro em serviço, diferença entre aro desgastado e aro teorico
# e taxa de desgaste total
tes = lcc_data[['Tes [mm/10^5 km]']]
# trt = lcc_data[['Trt [mm/10^5 km]']]
# tdt = lcc_data[['Tdt [mm/10^5 km]']]
n_steps = 20

# cria n_steps colunas com o desgaste defasado
for time_step in range(1, n_steps + 1):
    tes['Tes [mm/10^5 km]' + str(time_step)] = tes[['Tes [mm/10^5 km]']].shift(-time_step).values

# deleta linhas com valores nulos
tes.dropna(inplace=True)

# Extrai variáveis independes - colunas de tes até tes19 e y/ dependentes - tes1 até tes20
# Adiciona dimensão a variáveis
X = tes.iloc[:, :n_steps].values
X = np.reshape(X, (X.shape[0], n_steps, 1))

y = tes.iloc[:, 1:].values
y = np.reshape(y, (y.shape[0], n_steps, 1))

# print (X.shape, y.shape)

# para que a analise seja uma aproximação assertiva do desgaste do mês de Maio, é necessario separar os ultimos
# dados da serie de tempo para servirem de set de test

n_test = 500

# Realiza a indexação
X_train, X_test = X[:-n_test, :, :], X[-n_test:, :, :]
y_train, y_test = y[:-n_test, :, :], y[:-n_test, :, :]

# Cria array de 0 a n_train e mescla x e y
shuffle_mask = np.arange(0, X_train.shape[0])
np.random.shuffle(shuffle_mask)

X_train = X_train[shuffle_mask]
y_train = y_train[shuffle_mask]

# Seleciona dados no primeiro periodo de tempo e inicia o teste oculto com X0
X0 = X[:, 0, :]
Ht = tf.elu(tf.matmul(X0, Wx_h) + b)
y = []

# itera para cada período de tempo definido
for t in range(1, n_steps):
    Xt = X[:, t, :]
    Ht = tf.elu(tf.matmul(X0, Wx_h) + tf.matmul(Ht, Wh_h) + b)
    y.append(tf.matmul(Ht, Wh_y) + b_o)
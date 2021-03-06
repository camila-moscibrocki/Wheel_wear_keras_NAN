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
# X0 = X[:, 0, :]
# Ht = tf.nn.elu(tf.matmul(X0, Wx_h) + b)
# y = []

# Faz a iteração nos respectivos periodos de tempo, pegando o periodo mais proximo e iniciando o estado oculto em X0
# for t in range(1, n_steps):
#    Xt = X[:, t, :]
#    Ht = tf.nn.elu(tf.matmul(Xt, Wx_h) + tf.matmul(Ht, Wh_h) + b)
#    y.append(tf.matmul(Ht, Wx_h) + Wx_h)

n_inputs = 1
n_neurons = 64
n_outputs = 1
learning_rate = 0.001

graph = tf.Graph()
with graph.as_default():
    # placeholders
    tf_X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='X')
    tf_y = tf.placeholder(tf.float32, [None, n_steps, n_outputs], name='y')

    with tf.name_scope('Recurent_Layer'):
        cell = BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
        outputs, last_state = tf.nn.dynamic_rnn(cell, tf_X, dtype=tf.float32)

    with tf.name_scope('outlayer'):
        stacked_outputs = tf.reshape(outputs, [-1, n_neurons])
        stacked_outputs = tf.layers.dense(stacked_outputs, n_outputs, activation=None)
        net_outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

    with tf.name_scope('train'):
        loss = tf.reduce_mean(tf.abs(net_outputs - tf_y))  # MAE
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

n_iterations = 10000
batch_size = 64

with tf.Session(graph=graph) as sess:
    init.run()

    for step in range(n_iterations + 1):
        # cria os mini-lotes
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        X_batch = X_train[offset:(offset + batch_size), :]
        y_batch = y_train[offset:(offset + batch_size)]

    # mostra o MAE de treito a cada 2000 iterações
    if step % 2000 == 0:
        train_mae = loss.eval(feed_dict={tf_X: X_train, tf_y: y_train})
    print(step, "\tTrain MAE:", train_mae)

    # mostra o MAE de teste no final do treinamento
    test_mae = loss.eval(feed_dict={tf_X: X_test, tf_y: y_test})
    print(step, "\tTest MAE:", test_mae)

    # realiza previsões
    y_pred = sess.run(net_outputs, feed_dict={tf_X: X_test})

# Inserção de variaveis preditivas
features = ['km [10^5] SW', 'DELTA At [mm]', 'Hollow (Hl)  (Left)']
tes = lcc_data[features]
n_steps = 20

for var_col in features
    for tima_step in range(1, n_steps+1):
    # Cria colunas da variável defasada
    tes[var_col+str(time_step)]
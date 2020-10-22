import sys
sys.path.append("..") # Adds higher directory to python modules path.
import importlib
import initialize
import implementations
importlib.reload(initialize)
importlib.reload(implementations)
from implementations import logistic_regression, reg_logistic_regression
import datetime
from initialize import initialize

x_tr, x_te, y_tr, y_te, w_initial = initialize(data_train_path='../data/train500.csv', split_ratio = 0.5)

max_iters = 1000
gamma = 0.05
lambda_=5

# Logistic regression
start_time = datetime.datetime.now()
w, loss = logistic_regression(y=y_tr, tx=x_tr, initial_w=w_initial, max_iters=max_iters, gamma=gamma)
end_time = datetime.datetime.now()
print('== Logistic regression ==')
print("loss: " + str(loss) + ', Time: ' + str((end_time - start_time).total_seconds()) + 's')

# Logistic regression
start_time = datetime.datetime.now()
w, loss = reg_logistic_regression(y=y_tr, tx=x_tr, lambda_=lambda_, initial_w=w_initial, max_iters=max_iters, gamma=gamma)
end_time = datetime.datetime.now()
print('== Reg logistic regression ==')
print("loss: " + str(loss) + ', Time: ' + str((end_time - start_time).total_seconds()) + 's')
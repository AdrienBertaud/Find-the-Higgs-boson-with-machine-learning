import importlib
import initialize
importlib.reload(initialize)
from implementations import logistic_regression
import datetime
from initialize import initialize

x_tr, x_te, y_tr, y_te, w_initial = initialize()

max_iters = 1000
gamma = 0.05

# Logistic regression
start_time = datetime.datetime.now()
w, loss = logistic_regression(y=y_tr, tx=x_tr, initial_w=w_initial, max_iters=max_iters, gamma=gamma)
end_time = datetime.datetime.now()
print('== Logistic regression ==')
print("loss: " + str(loss) + ', Time: ' + str((end_time - start_time).total_seconds()) + 's')
import numpy as np

def sigmold(x):
    return 1/(1 + np.exp(-x))


def loss_func(x, t):
    delta = 1e-7    #log 무한대 발산 방지

    z = np.dot(x, W) + b
    y = sigmold(z)

    #cross-entropy
    return -np.sum(t * np.log(y+delta) + (1-t) * np.log((1-y) + delta))


def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = tmp_val + delta_x
        fx1 = f(x)
        x[idx] = tmp_val - delta_x
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2 * delta_x)
        x[idx] = tmp_val
        it.iternext()
    return grad


def error_val(x, t):
    delta = 1e-7  # log 무한대 발산 방지

    z = np.dot(x, W) + b
    y = sigmold(z)

    # cross-entropy
    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log((1 - y) + delta))

def predict(x):

    z = np.dot(x, W) + b
    y = sigmold(x)

    if y > 0.5:
        result = 1    #True
    else:
        result = 0    #False

    return y, result



x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(10, 1)
t_data = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).reshape(10, 1)

W = np.random.rand(1, 1)
b = np.random.rand(1)

learning_rate = 1e-2    #발산하는 경우, 1e-3 ~ 1e-6 등으로 바꾸어서 실행

f = lambda x : loss_func(x_data, t_data)

print("Initial error value =", error_val(x_data, t_data), "Initial W =", W, "Initial b =", b)

for step in range(10001):
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)

    if step % 400 == 0:
        print("step =", step, "error value =", error_val(x_data, t_data), "W =", W, "b =", b)

while True:
    print("X = ", end='')
    N = int(input())
    if N < 0:
        break
    print(predict(N))

import numpy as np
import matplotlib.pyplot as plt

def loss_func(x,t):
    y = np.dot(x,W) + b
    return (np.sum((t-y)**2))/(len(x))

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
        grad[idx] = (fx1-fx2)/(2*delta_x)

        x[idx] = tmp_val
        it.iternext()
    return grad

#손실함수 값 계산 함수
#입력변수 x, t : numpy type
def error_val(x, t):
    y=np.dot(x, W) + b
    return (np.sum((t-y)**2))/(len(x))

#학습을 마친 후, 임의의 데이터에 대해 미래 값 예측 함수
#입력변수 x : numpy type
def predict(x):
    y = np.dot(x, W)+b

    return y


loaded_data = np.loadtxt("./linear1.csv", delimiter=',', dtype=np.float32)

x_data = loaded_data[ : , 0:-1]
t_data = loaded_data[ : , [-1]]

W = np.random.rand(1,1)
b = np.random.rand(1)
print("W = ",W,",b = ",b)

learning_rate = 1e-8   #발산하는 경우, 1e-3 ~ 1e-6 등으로 바꾸어서 실행
f = lambda x : loss_func(x_data, t_data)    #f(x) = loss_func(x_data, t_data)
print("Initial error value =", error_val(x_data, t_data), "Initial W =[", end=' ')
for i in W:
    print(i, end=' ')
print("] , b =", b)

for step in range(40000):
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)
    if step % 4000 == 0:
        print("step =", step, "error value =", error_val(x_data, t_data), "W =[", end=' ')
        for i in W:
            print(i, end=' ')
        print("] , b =", b)

plt.scatter(x_data, t_data)
yhat = W[0]*x_data + b[0]
fig=plt.plot(x_data, yhat, lw=4, c='orange', label='Regression Line')
plt.xlabel('x_data', fontsize='20')
plt.ylabel('t_data', fontsize='20')
plt.show()

while True:
    print("X : ", end='')
    N = list(map(int, input().split()))
    ret = predict(N)
    print("예측값 : ", end='')
    for i in ret:
        print(round(i,3))


import numpy as np

# and, or, nand, xor data
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_tdata = np.array([0, 0, 0, 1]).reshape(4, 1)
or_tdata = np.array([0, 1, 1, 1]).reshape(4, 1)
nand_tdata = np.array([1, 1, 1, 0]).reshape(4, 1)
xor_tdata = np.array([0, 1, 1, 0]).reshape(4, 1)

# test data
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

#define neural network architecture
input_nodes = 2     #입력노드 2개
hidden_nodes = 6    #은닉노드 6개
output_nodes = 1    #출력노드 1개

W2 = np.random.rand(input_nodes, hidden_nodes)  #입력층-은닉층 가중치
b2 = np.random.rand(hidden_nodes)               #은닉층 바이어스

W3 = np.random.rand(hidden_nodes, output_nodes) #은닉층-출력층 가중치
b3 = np.random.rand(output_nodes)               #출력층 바이어스

learning_rate = 1e-2


#sigmoid 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#수치미분 함수
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
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad


#feed forward
def feed_forward(xdata, tdata):     #feed forward 를 통하여 손실함수(cross-entropy) 값 계산
    delta = 1e-7    #log 무한대 발산 방지

    z2 = np.dot(xdata, W2) + b2     #은닉층의 선형회귀 값
    a2 = sigmoid(z2)                #은닉층의 출력

    z3 = np.dot(a2, W3) + b3        #출력층의 선형회귀 값
    y = sigmoid(z3)                #출력층의 출력

    #cross-entropy
    return -np.sum( tdata*np.log(y + delta) + (1-tdata)*np.log((1-y) + delta))


#loss val
def loss_val(xdata, tdata):         #feed forward 를 통하여 손실함수(cross-entropy) 값 계산
    delta = 1e-7

    z2 = np.dot(xdata, W2) + b2  # 은닉층의 선형회귀 값
    a2 = sigmoid(z2)  # 은닉층의 출력

    z3 = np.dot(a2, W3) + b3  # 출력층의 선형회귀 값
    y = sigmoid(z3)  # 출력층의 출력

    # cross-entropy
    return -np.sum(tdata * np.log(y + delta) + (1 - tdata) * np.log((1 - y) + delta))


#query, 즉 미래값 예측 함수
def predict(xdata):
    z2 = np.dot(xdata, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    y = sigmoid(z3)

    if y > 0.5:
        result = 1  #True
    else:
        result = 0  #False

    return y, result


'''
#AND Gate 검증
f = lambda x : feed_forward(xdata, and_tdata)

print("Initial loss value =", loss_val(xdata, and_tdata))

for step in range(10001):
    W2 -= learning_rate * numerical_derivative(f, W2)

    b2 -= learning_rate * numerical_derivative(f, b2)

    W3 -= learning_rate * numerical_derivative(f, W3)

    b3 -= learning_rate * numerical_derivative(f, b3)

    if(step % 400 == 0):
        print("step =", step, ", loss value =", loss_val(xdata, and_tdata))

for data in test_data:
    (real_val, logical_val) =  predict(data)
    print("real_val =", real_val, ", logical_val =", logical_val)
'''

'''
#OR Gate 검증
f = lambda x : feed_forward(xdata, or_tdata)

print("Initial loss value =", loss_val(xdata, or_tdata))

for step in range(10001):
    W2 -= learning_rate * numerical_derivative(f, W2)

    b2 -= learning_rate * numerical_derivative(f, b2)

    W3 -= learning_rate * numerical_derivative(f, W3)

    b3 -= learning_rate * numerical_derivative(f, b3)

    if(step % 400 == 0):
        print("step =", step, ", loss value =", loss_val(xdata, or_tdata))

for data in test_data:
    (real_val, logical_val) =  predict(data)
    print("real_val =", real_val, ", logical_val =", logical_val)
'''

'''
#NAND Gate 검증
f = lambda x : feed_forward(xdata, nand_tdata)

print("Initial loss value =", loss_val(xdata, nand_tdata))

for step in range(10001):
    W2 -= learning_rate * numerical_derivative(f, W2)

    b2 -= learning_rate * numerical_derivative(f, b2)

    W3 -= learning_rate * numerical_derivative(f, W3)

    b3 -= learning_rate * numerical_derivative(f, b3)

    if(step % 400 == 0):
        print("step =", step, ", loss value =", loss_val(xdata, nand_tdata))

for data in test_data:
    (real_val, logical_val) =  predict(data)
    print("real_val =", real_val, ", logical_val =", logical_val)
'''

#XOR Gate 검증
f = lambda x : feed_forward(xdata, xor_tdata)

print("Initial loss value =", loss_val(xdata, xor_tdata))

for step in range(20001):
    W2 -= learning_rate * numerical_derivative(f, W2)

    b2 -= learning_rate * numerical_derivative(f, b2)

    W3 -= learning_rate * numerical_derivative(f, W3)

    b3 -= learning_rate * numerical_derivative(f, b3)

    if(step % 800 == 0):
        print("step =", step, ", loss value =", loss_val(xdata, xor_tdata))

for data in test_data:
    (real_val, logical_val) =  predict(data)
    print("real_val =", real_val, ", logical_val =", logical_val)
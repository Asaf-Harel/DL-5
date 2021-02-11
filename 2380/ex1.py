from gradient_descent import train_non_adaptive, calc_J_non_adaptive
import matplotlib.pyplot as plt
import unit10.b_utils as u10

X, Y = u10.load_dataB1W4_trainN()

costs, w, b = train_non_adaptive(X, Y, 0.0001, 150000, calc_J_non_adaptive)
print(f'w1={str(w[0])}  w2={str(w[1])}  w3={str(w[2])}  w4={str(w[3])}  b={str(b)}')
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 10,000)')
plt.show()

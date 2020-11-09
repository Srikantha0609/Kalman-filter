import numpy as np
import csv


class KalmanFilter(object):
    def __init__(self, F, B, H, Q, R, P, x):
        # F=State transition matrix
        # B=Control input matrix
        # P=state covariance matrix
        # Q=Process noise covariance matrix
        # R=Measurement Noise covariance matrix
        # x=state transition matrix
        # H is a matrix used in converting matrix into desired order
        if (F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x is None else x
    # Step-1: prediction of altitude data using acceleration data
    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x
    #Step-2: Updating the altitude value using predicted value and kalman gain
    def update(self, z):
        y = z - np.dot(self.H, self.x)    #y is the difference between measured and predicted value
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T)) # S is a matrix used to store denominator part of Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)  # estimation of state matrix
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),#updating matrix P
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)



def example():
        dt = 0.2
        x = np.array([[0.74], [0]])
        F = np.array([[1, dt], [0, 1]])
        B = np.array([[(0.5) * (dt) * (dt)], [dt]])
        H = np.array([1, 0]).reshape(1, 2)
        Q = np.array([[0.4, 0.4], [0.4, 4]])
        R = np.array([[0.4]])

        data = open(r"Dataset_1.csv")
        csv_data = csv.reader(data)
        data_lines = list(csv_data)
        u = []
        measurements = []
        for line in data_lines[0:]:
            u.append(float(line[7]))
        for line in data_lines[0:]:
            measurements.append(float(line[2]))

        kf = KalmanFilter(F=F, B=B, H=H, Q=Q, R=R, P=None, x=x)
        predictions = []

        for x in range(0, len(measurements)):
            z = measurements[x]
            predictions.append(np.dot(H, kf.predict([[u[x]]])))
            kf.update(z)
        pred=[]
        for j in range(0,len(predictions)):
            pred.append((predictions[j][0][0]))



        import matplotlib.pyplot as plt
        plt.plot(range(len(measurements)), measurements, label = 'Measurements')
        plt.plot(range(len(pred)), np.array(pred), label = 'Kalman Filter Prediction')
        plt.legend()
        plt.show()




if __name__ == '__main__':
    example()





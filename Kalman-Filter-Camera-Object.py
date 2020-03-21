import numpy as np
import matplotlib.pyplot as plt

##################################### Initialization ###################################
P = 100.0*np.eye(9)
dt = 0.01 # Time Step between Filter Steps
# Dynamic Matrix
A = np.matrix([[1.0, 0.0, 0.0, dt, 0.0, 0.0, 1/2.0*dt**2, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2],
               [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

# Measurement Matrix
H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

# Measurement Noise Covariance Matrix R
rp = 1.0**2  # Noise of Position Measurement
R = np.matrix([[rp, 0.0, 0.0],
               [0.0, rp, 0.0],
               [0.0, 0.0, rp]])

# Process Noise Covariance Matrix Q
sa = 0.1
G = np.matrix([[1/2.0*dt**2],
               [1/2.0*dt**2],
               [1/2.0*dt**2],
               [dt],
               [dt],
               [dt],
               [1.0],
               [1.0],
               [1.0]])
Q = G*G.T*sa**2

# Disturbance Control Matrix B
B = np.matrix([[0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0]])

# Control Input u, Assumed constant over time
u = 0.0
# Identity Matrix
I = np.eye(9)

# Measurements
Hz = 30.0 # Hz, frequency of IMU measurements
dt = 1.0/Hz
T = 0.5 # s measuremnt time
m = int(T/dt) # number of measurements

##################################### Data Productions ###################################

# Creation of the position data for the object
px = 0.0 # x Position Start
py = -1.0 # y Position Start
pz = 1.0 # z Position Start
vx = 10.0 # m/s Velocity at the beginning
vy = 0.0 # m/s Velocity
vz = 0.0 # m/s Velocity
c = 0.1 # Drag Resistance Coefficient
d = 0.9 # Damping
Xb=[]
Yb=[]
Zb=[]
for i in range(int(m)):
    accx = -c*vx**2  # Drag Resistance
    vx += accx*dt
    px += vx*dt
    accz = -9.806 + c*vz**2 # Gravitation + Drag
    vz += accz*dt
    pz += vz*dt
    
    if pz<0.01:
        vz=-vz*d
        pz+=0.02
    if vx<0.1:
        accx=0.0
        accz=0.0
    
    Xb.append(px)
    Yb.append(py)
    Zb.append(pz)

# Creation of the position data for the camera
t0 = 0
Xc=[0, 0, 0]
Yc=[1, 1, 1]
Zc=[2, 2, 2]
t0 = 0.1
for i in range(int(m)):
    t1 = t0+dt
    px = t1
    py = 1
    pz = t1
    t0 = t1
    Xc.append(px)
    Yc.append(py)
    Zc.append(pz)

Xbc = []
Ybc = []
Zbc = []

# Relative measurements Position_object --> Position_camera
lag = 2 #sec lag of IMU(camera)
for i,j in zip(range(int(m)+1), range(lag, int(m)+lag)):
    Xbc.append(Xb[i]-Xc[j])
    Ybc.append(Yb[i]-Yc[j])
    Zbc.append(Zb[i]-Zc[j])

# Add noise to the real position
noise= 0.1 # Sigma for position noise
Xbc_ = Xbc + noise * (np.random.randn(m))
Ybc_ = Ybc + noise * (np.random.randn(m))
Zbc_ = Zbc + noise * (np.random.randn(m))

measurements_bc = np.vstack((Xbc_,Ybc_,Zbc_))
measurements_c = np.vstack((Xbc,Ybc,Zbc))# camera produce 3d positions for object when t%5!=0
for t in range(int(m)):
    if t%5==0:
        for i in range(3):
            measurements_c[i][t] = 0.

##################################### Kalman Filter ###################################
# Initial State
x = np.matrix([0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 0.0, 0.0, -9.81]).T

xt = []
yt = []
zt = []

# Kalman Filter
for filter_step in range(m):

# 1- Prediction
    # state estimation
    x = A*x + B*u
    # A = state transition model which is applied to the previous state
    # B = control-input model which is applied to the control vector u

    # Predicted covariance to describe the distribution
    # Projection of covariance
    P = A*P*A.T + Q
    # Q is expected variance

# 2- Correction
    # Kalman Gain (information gain)
    S = H*P*H.T + R # R expected variance
    K = (P*H.T) * np.linalg.pinv(S)

    # Measurement bc
    Z = measurements_bc[:,filter_step].reshape(H.shape[0],1)
    y = Z - (H*x)     # correction
    x = x + (K*y)
    
    '''# Measurement c
    Z = measurements_c[:,filter_step].reshape(H.shape[0],1)
    y = Z - (H*x)       # correction
    x = x + (K*y)'''
    
    # Covariance estimation
    P = (I - (K*H))*P

    xt.append(float(x[0]))
    yt.append(float(x[1]))
    zt.append(float(x[2]))


##################################### Plot Preditions ###################################
# Plot positions in x/z Plane
fig = plt.figure(figsize=(16,9))
plt.plot(xt,zt, label='Kalman Filter Estimate')
plt.scatter(measurements_bc[0][:],measurements_bc[2][:], label='Measurement_relateve_obj_camera', c='gray', s=30)
plt.scatter(measurements_c[0][:],measurements_c[2][:], label='Measurement_camera', c='red', s=30)
plt.plot(Xbc_, Zbc_, label='Real')
plt.title('Kalman Filter Tracking')
plt.legend(loc='best',prop={'size':22})
plt.axhline(0, color='k')
plt.axis('equal')
plt.xlabel('X ($m$)')
plt.ylabel('Z ($m$)')
plt.ylim(-2, 2);
plt.savefig('Kalman-Filter-object-StateEstimates_2mes.png', dpi=150, bbox_inches='tight')

# Error measurement
dist = np.sqrt((np.asarray(Xbc)-np.asarray(xt))**2 + (np.asarray(Ybc)-np.asarray(yt))**2 + (np.asarray(Zbc)-np.asarray(zt))**2)
print('Estimated Position is %.2fm away from object position.' % dist[-1])




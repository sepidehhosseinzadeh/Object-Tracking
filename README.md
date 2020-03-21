# Object-Tracking


## Problem Statement
Data fusion and predictive object tracking. A hand-held object X equipped with 9DOF IMU sensors must be tracked within a range of 3m. X’s IMU sensors broadcast accelerom- eter information periodically. A stereo camera C, itself moving and equipped with 9DOF IMU sensors, provides occasional 3D positional information for X in C’s co-ordinate space. This information comes sporadically (i.e. when available), and is noisy due to imperfect resolution.
Question: Find the best way to fuse this data and to reliably predict X’s position in C’s reference frame, at some time in the near future (< 0.2 sec from now). Accuracy is more important than performance. You may assume that both X and C are hand-held by two separate people. You have access to all IMU data. Details:
1. C is equipped with software to extract the 3D position of X (when visible) from the stereo image.
2. The IMU data coming from the different devices is not synchronized, but the lag has been calibrated in advance and is known.
3. The lag in the 3D positional information has also been calibrated and is known.
4. All IMU data comes at the rate of 30 Hz.

## Data
from “http://robotsforroboticists.com/kalman-filtering/

## Method
(Adaptive) Kalman Filter is used for this problem. Alternative solutions can be Exponential Moving Average, Particle filter, and etc.
Particle filtering suffers from the well-known problem of sample degeneracy. Ensemble Kalman filtering avoids this, at the expense of treating non-Gaussian features of the forecast distribution incorrectly.

## Result
![](https://github.com/sepidehhosseinzadeh/Object-Tracking/blob/master/Kalman-Filter-object-StateEstimates_2mes.png)
![](https://github.com/sepidehhosseinzadeh/Object-Tracking/blob/master/Adaptive-Kalman-Filter-object-StateEstimates_2mes.png)
![](https://github.com/sepidehhosseinzadeh/Object-Tracking/blob/master/EMA-Estimates_2mes.png)
### Dependencies
numpy, matplotlib



#Question 1 
#Part a)
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

mu = 3.986E5
a = 6871.0
e = 0.07
i = 5.2
w = 37.8
omega = 89.0
v0 = 213.4

p = a*(1-e**2)
r = p/(1+e*np.cos(np.deg2rad(v0)))
r_vec = (np.array([[r*np.cos(np.deg2rad(v0)), r*np.sin(np.deg2rad((v0))), 0]])).reshape(3,1)
v_vec = (np.sqrt(mu/p)*np.array([-np.sin(np.deg2rad(v0)), e+np.cos(np.deg2rad((v0))), 0])).reshape(3,1)

R3 = np.array([[np.cos(np.deg2rad(-omega)), np.sin(np.deg2rad(-omega)),0],
      [-np.sin(np.deg2rad(-omega)), np.cos(np.deg2rad(-omega)), 0],
      [0, 0, 1]]);

R1 = np.array([[1, 0, 0],
      [0, np.cos(np.deg2rad((-i))), np.sin(np.deg2rad(-i))], 
      [0, -np.sin(np.deg2rad(-i)), np.cos(np.deg2rad(-i))]])

R3w = np.array([[np.cos(np.deg2rad(-w)), np.sin(np.deg2rad(-w)), 0],
       [-np.sin(np.deg2rad(-w)), np.cos(np.deg2rad(-w)), 0],
       [0 ,0,1]])

R = np.matmul(np.matmul(R3,R1),R3w)
rIJK = np.matmul(R,r_vec)
vIJK = np.matmul(R,v_vec)

print(rIJK)
print(vIJK)

#Part b

r = np.array([4589.7, 4980.4, 2997.5]) 
v = np.array([1.259, 4.613, 6.208])

def convert(r_,v_,mu = 3.986E5):
    h = np.cross(r_,v_)
    n = np.cross(np.array([0,0,1]),h)
    
    e = (1/mu)*((((np.linalg.norm(v_)**2)-(mu/np.linalg.norm(r_)))*r_)-(np.dot(r_,v_))*v_)
    e_val = np.linalg.norm(e)
    p = (np.linalg.norm(h)**2)/mu
    a = p/(1-(e_val**2))
    i = np.rad2deg(np.arccos(np.dot(np.array([0,0,1]),h)/(np.linalg.norm(h))))
    omega = np.rad2deg(np.arccos(np.dot(n,np.array([1,0, 0]))/np.linalg.norm(n)))
    w = 360-np.rad2deg(np.arccos(np.dot(n,e)/(np.linalg.norm(n)*np.linalg.norm(e))))
    nu = np.rad2deg(np.arccos(np.dot(e,r_)/(np.linalg.norm(e)*np.linalg.norm(r_))))
    
    print('e: ',e_val)
    print('a: ',a)
    print('i: ',i)
    print('Omega: ', omega)
    print('w: ',w)
    print('nu: ',nu)

convert(r,v)

#Question 2
def convert(r_,v_,mu = 3.986E5):
    h = np.cross(r_,v_)
    n = np.cross(np.array([0,0,1]),h)

    e = (1/mu)*((((np.linalg.norm(v_)**2)-(mu/np.linalg.norm(r_)))*r_)-(np.dot(r_,v_))*v_);
    e_val = np.linalg.norm(e)
    p = (np.linalg.norm(h)**2)/mu
    a = p/(1-(e_val**2))
    i = np.rad2deg(np.arccos(np.dot(np.array([0,0,1]),h)/(np.linalg.norm(h))))
    omega = np.rad2deg(np.arccos(np.dot(n,np.array([1,0, 0]))/np.linalg.norm(n)))
    w = np.rad2deg(np.arccos(np.dot(n,e)/(np.linalg.norm(n)*np.linalg.norm(e))))
    nu = np.rad2deg(np.arccos(np.dot(e,r_)/(np.linalg.norm(e)*np.linalg.norm(r_))))
    
    return (e_val,a,i,omega,w,nu)

vals = pd.read_csv('trv.csv')
r = []
v = []
for i in range(len(vals)):
    r_ini = []
    v_ini = []
    
    for j in range(3):
        r_ini.append(vals.iloc[i][j+1])
        v_ini.append(vals.iloc[i][j+4])
        
    r.append(r_ini)
    v.append(v_ini)

r = np.array(r)
v = np.array(v)
e_vals = []
a_vals = []
i_vals = []
omega_vals = []
w_vals = []
nu_vals = []

time = list(vals['t (s)'])

for i in range(len(r)):
    vals = convert(r[i],v[i])
    e_vals.append(vals[0])
    a_vals.append(vals[1])
    i_vals.append(vals[2])
    omega_vals.append(vals[3])
    w_vals.append(vals[4])
    nu_vals.append(vals[5])
    
plt.plot(time,e_vals)
plt.xlabel('Time (s)')
plt.ylabel('Eccentricity')
plt.title('Eccentricity vs time')
plt.show()

plt.plot(time,a_vals)
plt.xlabel('Time (s)')
plt.ylabel('Semi-major axis (km)')
plt.title('Semi-major axis vs time')
plt.show()

plt.plot(time,i_vals)
plt.xlabel('Time (s)')
plt.ylabel('Inclination angle (deg)')
plt.title('Inclination angle vs time')
plt.show()

plt.plot(time,omega_vals)
plt.xlabel('Time (s)')
plt.ylabel('Longitude of Ascending Node (deg)')
plt.title('Longitude of Ascending Node vs time')
plt.show()

plt.plot(time,w_vals)
plt.xlabel('Time (s)')
plt.ylabel('Argument of Periapsis (deg)')
plt.title('Argument of Periapsis vs time')
plt.show()

plt.plot(time,nu_vals)
plt.xlabel('Time (s)')
plt.ylabel('True Anomoly (deg)')
plt.title('True Anomoly vs time')
plt.show()
#-----------------

#Question 3 part b
M,e = .8892,.4

def dx(f,x):
    return abs(0-f(x))

def f(x):
    return x-e*math.sin(x)-M

def df(x):
    return 1-e*math.cos(x)

def newton(f,df,x0,tol):
    delta = dx(f,x0)
    while delta > err:
        x0 = x0 - f(x0)/df(x0)
        delta = dx(f,x0)
    print('Root is at:{}'.format(x0))
    print('f(x) at root is:{} '.format(f(x0)))

newton(f,df,M,1e-12)
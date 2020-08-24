import numpy as np
import matplotlib.pyplot as plt

alt = np.linspace(200,40000,1000)
# defining constants
mu = 3.986*(10**5) 
r_earth = 6571

# F = mu*m2/(r12)^2
# here, F = m2*a
#a = mu/(r12)^2

a = []
for i in range(len(alt)):
    accel = (mu/((alt[i]+r_earth)**2))*1000 #in m/s^2
    a.append(accel)

plt.plot(alt,a)
plt.xlabel('Altitute(km)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Gravitational acceleration vs. altitude')
plt.show()

#m2*v^2/(r12) = mu*m2/(r12)^2
#v = sqrt(mu/(r12))

v = []

for i in range(len(alt)):
    vel = np.sqrt(mu/(alt[i]+r_earth))
    v.append(vel)

plt.plot(alt,v)
plt.xlabel('Altitute(km)')
plt.ylabel('Orbital velocity (km/s)')
plt.title('Orbital velocity vs. altitude')
plt.show()

#V = sqrt(mu/(r12))
#V = (2*pi*R)/T
#T = (2*pi*r12)/sqrt(mu/(r12))
#T = 2*pisqrt((r12^3)/mu)

T = []

for i in range(len(alt)):
    per = 2*np.pi*np.sqrt(((alt[i]+r_earth)**3)/mu) #in seconds
    per = per/(3600) #in hours
    T.append(per)

plt.plot(alt,T)
plt.xlabel('Altitute(km)')
plt.ylabel('Orbital period (hr)')
plt.title('Orbital period vs. altitude')
plt.show()

radius = np.linspace(6771,10**6,3000)

hyper_vel = []
for i in range(len(radius)):
    h_v = np.sqrt((1.5**2) + ((2*mu)/radius[i]))
    hyper_vel.append(h_v)

plt.plot(hyper_vel)
plt.title('Spacecraft velocity vs. radius')
plt.xlabel('Radius(km)')
plt.ylabel('Velocity(km/s)')
plt.show()
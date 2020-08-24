import numpy as np
import matplotlib.pyplot as plt

# Rho values for all 3 different altitudes 
rho = [0.0023768924,0.001495637712,0.000889268208]
alt = [0,15000,30000]

##Declaring all the given variable
W = 19815 #Gross weight
wing_span = 53.3
av_chord = 6
e = .81
Ta_sl = 2*3650
r_sl = rho[1]

# Calculating Sref, AR
Sref = wing_span*av_chord 
AR = (wing_span**2)/(Sref)
aspeed = np.linspace(100,1000,500)

#function for Power required and Power available at 3 different altitudes with 500 different velocity data points

Pr_val = [[],[],[]]

def Pr(speed, r):
    for j in range(len(r)):
        for i in range (len(speed)):
            Cl = W/(.5*r[j]*(speed[i]**2)*Sref)
            Cd = .02 + ((Cl**2)/(np.pi*e*AR))
        
            num = 2*(W**3)*(Cd**2)
            den = r[j]*Sref*(Cl**3)
            
            Pr_val[j].append(np.sqrt(num/den)) 

Pa_val =[[],[],[]]
def Pa(speed, r):
    for j in range(len(r)):
        for i in range(len(speed)):
            Ta = (r[j]/r_sl)*Ta_sl
            Pa_val[j].append(Ta*(speed[i]))

Pr(aspeed,rho)
Pa(aspeed,rho)

#Q.3 a)
# plotting Power Required and Power Available vs. Airspeed
plt.plot(aspeed,Pr_val[0], label ='Power Required at 0 ft Altitude')
plt.plot(aspeed,Pa_val[0], label = 'Power Available')
plt.legend()
plt.xlabel('Airspeed')
plt.ylabel('Power')
plt.title('Power Required and Power Available vs. Airspeed')
plt.show()

plt.plot(aspeed,Pr_val[1], label ='Power Required at 15k ft Altitude')
plt.plot(aspeed,Pa_val[1], label = 'Power Available')
plt.legend()
plt.xlabel('Airspeed')
plt.ylabel('Power')
plt.title('Power Required and Power Available vs. Airspeed')
plt.show()

plt.plot(aspeed,Pr_val[2], label ='Power Required at 30k ft Altitude')
plt.plot(aspeed,Pa_val[2], label = 'Power Available')
plt.legend()
plt.xlabel('Airspeed')
plt.ylabel('Power')
plt.title('Power Required and Power Available vs. Airspeed')
plt.show()

#Q.3 b)
#Calculating the excess power
diff = [[],[],[]]

for i in range(len(Pa_val)):
    for j in range(len(Pa_val[0])):
        diff[i].append((Pa_val[i][j]-Pr_val[i][j]))

max_r_c_0 = max(diff[0])/W
max_r_c_15 = max(diff[1])/W
max_r_c_30 = max(diff[2])/W

print('Maximum rate of climb at 0 ft is: {} ft/s'.format(max_r_c_0))
print('Maximum rate of climb at 15000 ft is: {} ft/s'.format(max_r_c_15))
print('Maximum rate of climb at 30000 ft is: {} ft/s'.format(max_r_c_30))

r_c = [max_r_c_0, max_r_c_15, max_r_c_30]
plt.plot(r_c, alt)
plt.show()

#Q.3 c)
# R_Cmax = 1.66666667 ft/s
slope, inter = np.polyfit(r_c,alt,1)
s_ceiling = slope*1.66666667+inter

print('Absolute Ceiling: {}ft'.format(inter))
print('Service Ceiling: {}ft'.format(s_ceiling ))

#Q.4 a)
# Calculating and plotting Endurance
end =[[],[],[]]
def Endurance(speed, r):
    for j in range(len(r)):
        for i in range (len(speed)):
            Cl = W/(.5*r[j]*(speed[i]**2)*Sref)
            Cd = .02 + ((Cl**2)/(np.pi*e*AR))
            
            q = (1/.6)*(Cl/Cd)*np.log((7500+W)/W)
            end[j].append(q)

Endurance(aspeed,rho)

plt.plot(aspeed,end[0], label ='Endurance at 0ft ')
plt.plot(aspeed,end[1], label ='Endurance at 15k ft')
plt.plot(aspeed,end[2], label ='Endurance 30k ft')
plt.legend(loc=0)
plt.xlabel('Airspeed')
plt.ylabel('Endurance')
plt.title('Endurance vs. Airspeed')
plt.show()

#Q.4 b)
# Calculating and plotting Range
ran =[[],[],[]]

def Range(speed, r):
    for j in range(len(r)):
        for i in range (len(speed)):
            Cl = W/(.5*r[j]*(speed[i]**2)*Sref)
            Cd = .02 + ((Cl**2)/(np.pi*e*AR))
            
            q = 2*np.sqrt(2/(r[j]*Sref))*(1/.6)*((Cl**.5)/(Cd))*(np.sqrt(7500+W)-np.sqrt(W))
            
            ran[j].append(q)

Range(aspeed, rho)

plt.plot(aspeed,ran[0], label ='Range at 0ft ')
plt.plot(aspeed,ran[1], label ='Range at 15k ft')
plt.plot(aspeed,ran[2], label ='Range at 30k ft')
plt.legend(loc=0)
plt.xlabel('Airspeed')
plt.ylabel('Range')
plt.title('Range vs. Airspeed')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv('naca2412i_1.csv')
df2 = pd.read_csv('naca2412i_2.csv')
df3 = pd.read_csv('naca2412i_3.csv')

df1.dropna(inplace=True)
df2.dropna(inplace=True)
df3.dropna(inplace=True)

x = np.linspace(0,1,num=(len(df1)))

#Quesiton 4 part a)

plt.plot(x,df1['Cp lower'].tolist(), color='g', label ='Cp upper vs. x/c upper')
plt.plot(x, df1['Cp upper'].tolist(), color='orange',label ='Cp upper vs. x/c upper')
plt.gca().invert_yaxis()
plt.xlabel('x/c')
plt.ylabel('Cp')
plt.title('Cp vs x/c at Zero  Angle of Attack')
plt.legend()
plt.show()

plt.plot(x,df2['Cp lower'].tolist(), color='g', label ='Cp lower vs. x/c lower')
plt.plot(x, df2['Cp upper'].tolist(), color='orange',label ='Cp upper vs. x/c upper')
plt.gca().invert_yaxis()
plt.xlabel('x/c')
plt.ylabel('Cp')
plt.title('Cp vs x/c at Five Degrees Angle of Attack')
plt.legend()
plt.show()

plt.plot(x,df3['Cp lower'].tolist(), color='g', label ='Cp lower vs. x/c lower')
plt.plot(x, df3['Cp upper'].tolist(), color='orange', label ='Cp upper vs. x/c upper')
plt.gca().invert_yaxis()
plt.xlabel('x/c')
plt.ylabel('Cp')
plt.title('Cp vs x/c at Negative Five Degrees Angle of Attack')
plt.legend()
plt.show()

#Quesiton 4 part b)
l_0 = np.trapz(x = df1['x/c lower'], y = df1['Cp lower'])
u_0 = np.trapz(x = df1['x/c upper'], y= df1['Cp upper'])

cl_0 = l_0-u_0
print('cl_0: {}'.format(cl_0))

l_5 = np.trapz(x = df2['x/c lower'], y = df2['Cp lower'])
u_5 = np.trapz(x = df2['x/c upper'], y= df2['Cp upper'])

cl_5 = l_5-u_5
print('cl_5: {}'.format(cl_5))

l_n5 = np.trapz(x = df3['x/c lower'], y = df3['Cp lower'])
u_n5 = np.trapz(x = df3['x/c upper'], y= df3['Cp upper'])

cl_n5 = l_n5 - u_n5
print('cl_n5: {}'.format(cl_n5))

#Quesiton 4 part c)
cl = [cl_0 ,cl_5, cl_n5]
angles = [0,5,-5]
d = pd.Series(data = cl, index = angles)

d.plot()
plt.axhline(y = 0, color= 'red', ls = '--')
plt.axvline(x = 0, color= 'red', ls = '--')
plt.xlabel('Angle of Attack')
plt.ylabel('Cl')
plt.title('Coefficient of life vs. Angle of Attack')
plt.show()

slope, inter = np.polyfit(angles,cl,1)
xinter = -inter/slope
print('x_intercept: {}'.format(xinter))


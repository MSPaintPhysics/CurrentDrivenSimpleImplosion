"""
Created on Thu Sep 14 12:29:54 2017

@author: Paul
"""
import numpy as np
from scipy import integrate
import matplotlib.pylab as plt

#Liner Parameters
mu = 4*np.pi*10**-7 #magnetic constant (m kg s^-2 A^-2)
rho_Al = 2700 #liner density, in this case Aluminium in (kg/m^3)
t_Al = 1000E-9 #liner thickness, in meters
R0 = 0.003175 #liner starting radius, in meters
h = 1E-2 #liner height, in meters
l = 2.5E-2 #liner length, in meters

#Calculate some useful values from the paramters
V_Al = l*t_Al*h
SA_Al = 2*np.pi*h
m_Al = V_Al*rho_Al
I_0 = 1000E3 #peak current in amps
t_rise = 100E-9 #rise time of current in seconds
B = mu*I_0/(2*np.pi)
Pmag = B**2/(2*mu)
C = (Pmag*SA_Al)/(m_Al)
w = (2*np.pi)/(4*t_rise)


#Defining the 2nd order ODE as 2 1st order ODEs
def Implosion(t, y): 
    #Output from ODE function must be a COLUMN vector, with n rows
    n = 2      #2:for two ODEs
    dydt = np.zeros((n,1))
    dydt[0] = y[1]
    dydt[1] = -C*(np.sin(w*t))**4/y[0]
    return[ dydt[0], dydt[1] ]

#Choose the integrator
sol = integrate.ode(Implosion).set_integrator('vode', method='bdf',order=5)
 
#Set the time range
t_start = 0
t_final = 3*t_rise
delta_t = 1E-10

#Number of time steps plus 1 extra for initial condition
num_steps = int(np.floor((t_final - t_start)/delta_t) + 1)
 
#Set initial condition(s): for integrating variable and time
r0 = R0 #starting radius
v0 = 0 #starting velocity
sol.set_initial_value([r0, v0])

# Create vectors to store trajectories
t = []
r = []
v = []
t.append(t_start)
r.append(r0)
v.append(v0)
 
# Integrate the ODE across each delta_t timestep
k = 1
while sol.successful() and k < num_steps:
    sol.integrate(sol.t + delta_t)
 
    # Store the results to plot later
    t.append(sol.t)
    r.append(sol.y[0])
    v.append(sol.y[1])
    k += 1

#Scale Answers
ts = [x*1E9 for x in t]
rs = [x/R0 for x in r]

#Define time steps for current trace
xs2 = np.linspace(0, 2*t_rise, num_steps)
fig, ax1 = plt.subplots()

#Plotting solution and current pulse
ax1.plot(ts, rs, 'k', linewidth=4)
plt.ylim([0,1.01])
ax1.set_xlabel('Time (ns)', fontsize=24)
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Normalized Liner Radius', color='k', fontsize=24)
ax1.tick_params('y', colors='k', labelsize=24)
ax1.tick_params('x', colors='k', labelsize=24)

ax2 = ax1.twinx()
s2 = I_0*np.sin(w*xs2)/1E6
ax2.plot(xs2*1E9,s2, 'b', linewidth=4)
ax2.set_ylabel('Current (MA)', color='b', fontsize=24)
ax2.tick_params('y', colors='b', labelsize=24)
plt.xlim([0,3*t_rise*1E9])
plt.ylim([0,1.1*I_0/1e6])
fig.set_figheight(12)
fig.set_figwidth(16)
fig.tight_layout()
plt.show()
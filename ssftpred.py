#Lucas Logue 08/2/2025
#super low fidelity TDSE solver to kickstart project
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
import time

#domain parameters
Nx, Ny = 256, 256
Lx, Ly = 10.0, 10.0
dx, dy = Lx/Nx, Ly/Ny #space between points on grid

x = np.linspace(-Lx/2, Lx/2 - dx, Nx) #from -5 to 5
y = np.linspace(-Ly/2, Ly/2 - dy, Ny) #from -5 to 5
X, Y = np.meshgrid(x, y) #make grid from point vectors

#Step parameters
dt = 0.005 #time between steps
Nt = 1000 #number of steps

#Fourier Transformation Wave Numbers
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2
#precomputes momentum factors for wave number's across the steps

#Initial Wave-Packet
k0 = 5.0 #momentum in x
sigma = 0.5 #gaussian width
x0, y0 = -2.0, 0.0 #starting point of wavepacket

psi0 = (
    1/(sigma * np.sqrt(np.pi)) #ensure normalization across width, so probability = 1 over all space
    * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) #gaussian envelope centered at our starting point
    * np.exp(1j * k0 * X) #plane wave factor so it's all good for the fourier frequency approach
)
psi = psi0.copy()

#propagator values 
halfkprop = np.exp(-.05j * K2 * dt) #using automic units h=m=1 
V = np.zeros_like(X) #placeholder values, remember to set V(X, Y) to 0 for slit
#CREATE THE BARRIER
V0 = 1e3 #super high energy for the barrier region, so the wave bounces off the fucker
barrier_width = .2
slitheight = .25
barmask = (np.abs(X) < barrier_width/2) #for the points X in the - and + direction that are in the bars interval
slitmask = (np.abs(Y - 1.0) < slitheight) | (np.abs(Y + 1.0) < slitheight) #in the range +-1 for Y's, where the Y < then the height
V[barmask & ~slitmask] = V0 #set the barrier 
fullvprop = np.exp(-1j * V * dt)

#Pulse params
dt_vec = np.arange(Nt)*dt
control_amp = 100.0
control_freq = 2*np.pi*2.0
controlph = 0
envelope = np.exp(-((dt_vec - 0.2)/.05)**2)

drivefield = barmask & ~slitmask
drive = control_amp*envelope*np.sin(control_freq* dt_vec + controlph)
control_shape = X * drivefield

drivepos = np.maximum(drive, 0.0)

#Time stepping
start = time.time()
frames = []  #frames array for gif
for n in range(Nt): #loop all steps
    #kinetic half step
    psi_hat = fft2(psi)
    psi_hat *= halfkprop
    psi = ifft2(psi_hat)

    #potential for the full step
    Vpulsed = V.copy()
    Vpulsed[drivefield] = V[drivefield] - drivepos[n]
    #psi *= fullvprop
    psi *= np.exp(-1j * Vpulsed * dt)

    #kinetic half step again
    psi_hat = fft2(psi)
    psi_hat *= halfkprop
    psi = ifft2(psi_hat)

    #capture frame for gif every 50th step
    if n % (Nt // 50) == 0:
        intensity = np.abs(psi)**2
        norm = intensity / intensity.max()
        frames.append((plt.cm.inferno(norm)[:,:,:3] * 255).astype(np.uint8))
end = time.time()
print(f"Simulation duration {end - start:.2f} s")
from imageio import mimsave
fps = 10
mimsave("small_double_slit.gif", frames, fps=fps)
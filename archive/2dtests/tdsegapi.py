import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
import time
from imageio import mimsave


def init_domain(Nx=256, Ny=256, Lx=10.0, Ly=10.0, dt=0.005):
    dx, dy = Lx/Nx, Ly/Ny #space between points on grid

    x = np.linspace(-Lx/2, Lx/2 - dx, Nx) 
    y = np.linspace(-Ly/2, Ly/2 - dy, Ny) 
    X, Y = np.meshgrid(x, y) #make grid from point vectors

    #Fourier Transformation Wave Numbers
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2

    halfkprop = np.exp(-.25j * K2 * dt) #using automic units h=m=1 
    V = np.zeros_like(X) #placeholder values, remember to set V(X, Y) to 0 for slit
    V0, bwidth, sheight = 1e3, 0.2, 0.25

    barmask = (np.abs(X) < bwidth/2) #for the points X in the - and + direction that are in the bars interval
    slitmask = (np.abs(Y - 1.0) < sheight) | (np.abs(Y + 1.0) < sheight) #in the range +-1 for Y's, where the Y < then the height
    bmask = barmask & ~slitmask
    V[bmask] = V0

    return X, Y, dx, dy, V, halfkprop, bmask

def propogate(psi, V, halfkprop, dt, drive_val, control_shape):
    psi_hat = fft2(psi)
    psi_hat *= halfkprop
    psi = ifft2(psi_hat)

    #potential for the full step
    Vinter = -control_shape * drive_val
    Vpulsed = V + Vinter
    psi *= np.exp(-1j * Vpulsed * dt)

    psi_hat = fft2(psi)
    psi_hat *= halfkprop
    psi = ifft2(psi_hat)
    return psi

def reward(psi, X, dx, dy):
    intensity = np.abs(psi)**2
    return np.sum(intensity[X>0])*dx*dy

def main():
    #parameters (will be inputted in future)
    dt = 0.005
    Nt = 1000

    X, Y, dx, dy, V, halfkprop, bmask = init_domain(dt=dt)

    # initial wavepacket
    k0, sigma = 5.0, 0.5
    x0, y0 = -2.0, 0.0
    psi0 = (
        1/(sigma * np.sqrt(np.pi))
        * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
        * np.exp(1j * k0 * X)
    )
    psi = psi0.copy()

    dt_vec = np.arange(Nt) * dt
    control_amp = 100.0
    control_freq = 2*np.pi*2.0
    control_phase = 0.0

    # Define the shape of the pulse
    pulse_center_t = 1.0 # Time (in simulation units) when the pulse peaks
    pulse_width = 0.2    # Duration of the pulse
    
    # Create the Gaussian envelope
    envelope = np.exp(-((dt_vec - pulse_center_t) / pulse_width)**2)
    drive = control_amp *envelope* np.sin(control_freq * dt_vec + control_phase)


    frames = []
    start = time.time()
    for n in range(Nt):
        psi = propogate(psi=psi, V=V, halfkprop=halfkprop, dt=dt, drive_val=drive[n], control_shape=X*bmask)

        if n % (Nt//50) == 0:
            intensity = np.abs(psi)**2
            norm = intensity / intensity.max()
            frames.append((plt.cm.inferno(norm)[:,:,:3] * 255).astype(np.uint8))
    duration = time.time() - start
    trans = reward(psi, X, dx, dy)
    print(trans)
    print('dur ', duration)
    mimsave("small_double_slit.gif", frames, fps=10)

if __name__ == '__main__':
    main()
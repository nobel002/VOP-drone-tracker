import numpy as np

mic1 = np.array([0.1,np.pi/3,np.pi/3])
mic2 = np.array([0.1,np.pi/3,np.pi])
mic3 = np.array([0.1,np.pi/3,5*np.pi/3])
mic4 = np.array([0.1,np.pi,0])

def x(r,theta,phi):
    return r*np.sin(theta)*np.cos(phi)
def y(r,theta,phi):
    return r*np.sin(theta)*np.sin(phi)
def z(r,theta,phi):
    return r*np.cos(theta)

def detalax(m1,m2):
    return x(m1[0],m1[1],m1[2]) - x(m2[0],m2[1],m2[2])

def delatay(m1,m2):
    return y(m1[0],m1[1],m1[2]) - y(m2[0],m2[1],m2[2])

def deltaz(m1,m2):
    return z(m1[0],m1[1],m1[2]) - z(m2[0],m2[1],m2[2])

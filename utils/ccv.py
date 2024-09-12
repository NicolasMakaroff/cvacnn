import jax
from jax import grad, jit, lax
import jax.numpy as jnp
from scipy.ndimage import distance_transform_edt as distance
from matplotlib import pyplot as plt
import numpy as np

@jit
def _im2double(img):
    #img = img.astype(jnp.float)
    img /= jnp.abs(img).max()
    return img

def _compute_phi(mask: jnp.ndarray):
    phi = distance(mask == 0) - distance((1 - mask) == 0) + _im2double(mask) - 0.5
    return phi

@jit
def CDT(I, k):
    l = 0.35
    n,h,w,c = I.shape
    [X,Y] = jnp.meshgrid(jnp.linspace(-jnp.floor(k/2),jnp.floor(k/2),num=7), jnp.linspace(-jnp.floor(k/2),jnp.floor(k/2),num=7))
    dis_0 = jnp.exp(-jnp.sqrt(X**2 + Y**2)/l)
    dis = jnp.reshape(jnp.dstack([dis_0]*n),(n,c,7,7)) 
    return -l * jnp.log(lax.conv_general_dilated(I,dis,window_strides=(1,1),padding='SAME',dimension_numbers=('NHWC', 'OIHW', 'NHWC')))
    
@jit
def CCDT_2(I, s, k):
    
    n,h,w,c = I.shape

    D_star = jnp.reshape(jnp.nan_to_num(CDT(I,k),posinf=0.0)[:,:,:,0],(n,h,w,c))

    flat = D_star > 0
    D = D_star 
    I = I + flat
    pad = jnp.floor(k/2)
    def for_loop(idx, input):
        conv = CDT(input[1],7)
        D_star = jnp.reshape(jnp.clip(jnp.nan_to_num(conv ,posinf=0.0),0)[:,:,:,0],(n,h,w,c))
        flat = D_star > 0
        I = input[1] + flat
        D = input[0] +  (idx * pad) * flat +  D_star
        return D, I

    D, I = lax.fori_loop(lower=0, upper=50, body_fun=for_loop, init_val=(D,I))
    return D

@jit
def conv_phi(mask: jnp.ndarray):
    return CCDT_2((mask == 0)*1.0,1,7) - CCDT_2(((1-mask)==0)*1.0,1,7)+_im2double(mask) - 0.5

@jit
def _heavyside(x: jnp.array, eps=1.):
    """Returns the result of a regularised heavyside function of the
    input value(s).
    """
    return 0.5 * (1. + (2./jnp.pi) * jnp.arctan(x/eps))

@jit
def _region_mean(image: jnp.array, Hphi: jnp.array):
    Hinv = 1. - Hphi
    Hsum = jnp.sum(Hphi)
    Hinvsum = jnp.sum(Hinv)
    avg_inside = jnp.sum(image * Hphi)
    avg_oustide = jnp.sum(image * Hinv)
    #if Hsum != 0:
    avg_inside /= Hsum
    #if Hinvsum != 0:
    avg_oustide /= Hinvsum

    return (avg_inside, avg_oustide)  #

@jit
def _energy_contrib(image: jnp.array, 
                    Hphi: jnp.array, 
                    lambda1: float, 
                    lambda2: float):
    (c1, c2) = _region_mean(image, Hphi)
    Hinv = 1. - Hphi
    return (lambda1 * (image-c1.reshape(-1,1,1))**2 * Hphi +
            lambda2 * (image-c2.reshape(-1,1,1))**2 * Hinv)

@jit
def _length_contrib(phi: jnp.array, mu: float = 0.2):
    padded_phi = jnp.pad(phi, ((0,0),(1,1),(1,1),(0,0)), mode='edge')
    fy = (padded_phi[:,2:, 1:-1,:] - padded_phi[:,:-2, 1:-1,:]) / 2.0
    fx = (padded_phi[:,1:-1, 2:,:] - padded_phi[:,1:-1, :-2,:]) / 2.0
    fyy = padded_phi[:,2:, 1:-1,:] + padded_phi[:,:-2, 1:-1,:] - 2*phi
    fxx = padded_phi[:,1:-1, 2:,:] + padded_phi[:,1:-1, :-2,:] - 2*phi
    fxy = .25 * (padded_phi[:,2:, 2:,:] + padded_phi[:,:-2, :-2,:] - 
        padded_phi[:,:-2, 2:,:] - padded_phi[:,2:, :-2,:])
    grad2 = fx**2 + fy**2
    K = ((fxx*fy**2 - 2*fxy*fx*fy + fyy*fx**2) /
         (grad2*jnp.sqrt(grad2) + 1e-8))
    
    return mu * K

def _ballon_force(phi: jnp.array):
    levelset = jnp.array((phi > 0)*1)
    grad = jnp.gradient(levelset)
    normal = jnp.sqrt(grad[0]**2 + grad[1]**2)
    return normal

@jit
def _energy(image: jnp.array, 
            phi: jnp.array, 
            mu: float = 0.2, 
            lambda1: float = 1.0, 
            lambda2: float = 1.0,
            balloon:float = 1.0):
    H = _heavyside(phi)
    avgenergy = _energy_contrib(image, H, lambda1, lambda2)
    lnenergy = _length_contrib(phi, mu)
    #balenergy = _ballon_force(phi)
    return jnp.sum(avgenergy) + jnp.sum(lnenergy) #+ balloon * jnp.sum(balenergy)
@jit
def _dirac(x, eps=1.):
    """Returns the result of a regularised dirac function of the
    input value(s).
    """
    return eps / (eps**2 + x**2)

@jit
def _levelset_variation(image: jnp.array, 
            phi: jnp.array, 
            mu: float, 
            lambda1: float, 
            lambda2: float,
            dt: float):
    eta = 1e-16
    P = jnp.pad(phi, ((0,0),(1,1),(1,1),(0,0)), mode='edge')
    phixp = P[:,1:-1, 2:,:] - P[:,1:-1, 1:-1,:]
    phixn = P[:,1:-1, 1:-1,:] - P[:,1:-1, :-2,:]
    phix0 = (P[:,1:-1, 2:,:] - P[:,1:-1, :-2,:]) / 2.0

    phiyp = P[:,2:, 1:-1,:] - P[:,1:-1, 1:-1,:]
    phiyn = P[:,1:-1, 1:-1,:] - P[:,:-2, 1:-1,:]
    phiy0 = (P[:,2:, 1:-1,:] - P[:,:-2, 1:-1,:]) / 2.0

    C1 = 1. / jnp.sqrt(eta + phixp**2 + phiy0**2)
    C2 = 1. / jnp.sqrt(eta + phixn**2 + phiy0**2)
    C3 = 1. / jnp.sqrt(eta + phix0**2 + phiyp**2)
    C4 = 1. / jnp.sqrt(eta + phix0**2 + phiyn**2)

    K = (P[:,1:-1, 2:,:] * C1 + P[:,1:-1, :-2,:] * C2 +
         P[:,2:, 1:-1,:] * C3 + P[:,:-2, 1:-1,:] * C4)

    Hphi = 1 * (phi > 0)
    (c1, c2) = _region_mean(image, Hphi)

    difference_from_average_term = (- lambda1 * (image-c1.reshape(-1,1,1))**2 +
                                    lambda2 * (image-c2.reshape(-1,1,1))**2)
    new_phi = (phi + (dt*_dirac(phi)) *
               (mu*K + difference_from_average_term))

    ### Add balloon force ? 

    return new_phi / (1 + mu * dt * _dirac(phi) * (C1+C2+C3+C4)) 

def test_edt(img):

   w, h = img.shape
   dt = np.zeros((w,h), np.uint32)
   # Forward pass
   x = 0
   y = 0
   if img[x,y] == 0:
      dt[x,y] = 65535 # some large value
   for x in range(1, w):
      if img[x,y] == 0:
         dt[x,y] = 3 + dt[x-1,y]
   for y in range(1, h):
      x = 0
      if img[x,y] == 0:
         dt[x,y] = min(3 + dt[x,y-1], 4 + dt[x+1,y-1])
      for x in range(1, w-1):
         if img[x,y] == 0:
            dt[x,y] = min(4 + dt[x-1,y-1], 3 + dt[x,y-1], 4 + dt[x+1,y-1], 3 + dt[x-1,y])
      x = w-1
      if img[x,y] == 0:
         dt[x,y] = min(4 + dt[x-1,y-1], 3 + dt[x,y-1], 3 + dt[x-1,y])
   # Backward pass
   for x in range(w-2, -1, -1):
      y = h-1
      if img[x,y] == 0:
         dt[x,y] = min(dt[x,y], 3 + dt[x+1,y])
   for y in range(h-2, -1, -1):
      x = w-1
      if img[x,y] == 0:
         dt[x,y] = min(dt[x,y], 3 + dt[x,y+1], 4 + dt[x-1,y+1])
      for x in range(1, w-1):
         if img[x,y] == 0:
            dt[x,y] = min(dt[x,y], 4 + dt[x+1,y+1], 3 + dt[x,y+1], 4 + dt[x-1,y+1], 3 + dt[x+1,y])
      x = 0
      if img[x,y] == 0:
         dt[x,y] = min(dt[x,y], 4 + dt[x+1,y+1], 3 + dt[x,y+1], 3 + dt[x+1,y])
   return dt

@jit
def chan_vese(image: jnp.ndarray, 
            phi: jnp.ndarray, 
            max_iter: int = 200, 
            mu: float = 0.2,
            lambda1: float = 1.0,
            lambda2: float = 2.0,
            balloon: float = 1.0, 
            tol: float = 1e-3,
            dt: float = 0.5, 
            color: str = 'r'):

        #phi = conv_phi(init_mask)#jnp.reshape(test_edt(init_mask.squeeze()==0) - test_edt((1 - init_mask.squeeze()) == 0)  - 0.5, (1,1,112,112))
        #print(phi)
        #energy = _energy(image, phi, mu, lambda1, lambda2, balloon)
        #i=0
        #energies = []
        #phivar = tol + 1
        #oldphi = phi
        #phi = _levelset_variation(image, phi, mu, lambda1, lambda2, dt)
        #phivar = jnp.sqrt(((oldphi - phi)**2).mean())

        #segmentation = phi > 0
        #energy = _energy(image, phi, mu, lambda1, lambda2, balloon)
        #energies.append(energy)
        #energy = new_energy
        def while_loop(i,phi):
            return _levelset_variation(image, phi, mu, lambda1, lambda2, dt)
        #list = []
        phi = lax.fori_loop(lower=0, upper=50, body_fun=while_loop, init_val=phi)
        #for i in range(500): 
        #    phi = _levelset_variation(image, phi, mu, lambda1, lambda2, dt)
        #    list.append(phi)
        '''while(phivar > tol and i < max_iter):
                oldphi = phi
                phi = _levelset_variation(image, phi, mu, lambda1, lambda2, dt)
                phivar = jnp.sqrt(((oldphi - phi)**2).mean())

                segmentation = phi > 0
                energies.append(energy)
                new_energy = _energy(image, phi, mu, lambda1, lambda2, balloon)
                energy = new_energy

                i+=1'''
        #segmentation = phi > 0
        return phi#, energies)

        
'''if __name__ == '__main__':
    import time
    import tensorflow as tf
    from skimage import segmentation

    img = imread('media_images_attention_5711_231626d72c6588cad226.png')[:,:,0] /255
    
    #img2 = imread('TCGA_DU_7019_19940908_17.tif')
    #img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    #img2 = np.dot(img2[...,:3], [0.2989, 0.5870, 0.1140])
    #mask = jnp.zeros(img.shape)
    updated_mask = np.array(imread('media_images_attention_15681_86747732cdcc9c88964c.png')[:,:,0] /255) > 0.7 * 1.0

    #mask.at[80:120,90:135].set(1) # for brain IRM
    #updated_mask2 = mask.at[100:190,60:120].set(1) # for brain IRM
    #imgs = jnp.reshape(jnp.stack([img]),(1,1,112,112))
    #masks = jnp.reshape(jnp.stack([updated_mask]),(1,1,112,112))
    #imgs_3 = imgs * masks

    print('Method running...')
    start = time.time()
    print(img.shape)
    print(type(updated_mask))
    seg, phi, energies = segmentation.chan_vese(img, init_level_set=updated_mask,  max_num_iter=1000,mu=.1,lambda1=1, lambda2=2,extended_output=True)

    end = time.time()
    print(f'Running time: {end-start}')
    f, axs = plt.subplots(2,2, sharey=True, sharex=True)
    axs[0,1].imshow(img)
    #axs[0,0].contour(updated_mask)
    #xs[1,0].imshow(energies)
    #axs[1,0].imshow(phi[0,0,:,:])
    axs[0,0].imshow(updated_mask)
    axs[1,1].imshow(seg.squeeze())
    plt.show()
    print('... Done')'''

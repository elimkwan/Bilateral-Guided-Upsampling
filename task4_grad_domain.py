#Task 4: Gradient domain reconstruction

import os.path as path
import skimage.io as io
import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import skimage.filters as filters
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sksparse.cholmod import cholesky
from scipy import signal
import skimage
import time

def img2grad_field(img):
    """Return a gradient field for a greyscale image
    The function returns image [height,width,2], where the last dimension selects partial derivates along x or y
    """
    # img must be a greyscale image
    sz = img.shape
    G = np.zeros([sz[0], sz[1], 2])
    # Gradients along x-axis
    G[:,:,0] = signal.convolve2d( img, np.array([1, -1, 0]).reshape(1,3), 'same', boundary='symm' )
    # Gradients along y-axis
    G[:,:,1] = signal.convolve2d( img,  np.array([1, -1, 0]).reshape(3,1), 'same', boundary='symm' )
    return G

def reconstruct_grad_field( G, w, v_00, img, solver):
    """Reconstruct a (greyscale) image from a gradcient field
    G - gradient field, for example created with img2grad_field
    w - weight assigned to each gradient
    v_00 - the value of the first pixel 
    solver: can be specified as "cholesky" or "spsolve"
    """
    sz = G.shape[:2] 
    N = sz[0]*sz[1]

    # Gradient operators as sparse matrices
    o1 =  np.ones((N,1))
    B = np.concatenate( (-o1, np.concatenate( (np.zeros((sz[0],1)), o1[:N-sz[0]]), 0 ) ), 1)
    B[N-sz[0]:N,0] = 0
    Ogx = sparse.spdiags(B.transpose(), [0 ,sz[0]], N, N, format = 'csr') # Forward difference operator along x

    B = np.concatenate( (-o1 ,np.concatenate((np.array([[0]]), o1[0:N-1]) ,0)), 1)
    B[sz[0]-1::sz[0], 0] = 0
    B[sz[0]::sz[0],1] = 0
    Ogy = sparse.spdiags( B.transpose(), [0, 1], N, N, format = 'csr') # Forward difference operator along y
    
    #TODO: Implement the gradient domain reconstruction 
    Ogxt = Ogx.transpose()
    Ogyt = Ogy.transpose()
    w_diag = sparse.spdiags(w.flatten(order='F'),0, Ogxt.shape[0], Ogxt.shape[1], format = 'csr')

    Gmx = G[:,:,0].flatten(order='F').reshape(-1,1) #column vector
    Gmx_flat = sp.sparse.csr_matrix(Gmx)
    Gmy = G[:,:,1].flatten(order='F').reshape(-1,1) #column vector
    Gmy_flat = sp.sparse.csr_matrix(Gmy)

    C = np.concatenate([[1], np.zeros(G.shape[0]*G.shape[1]-1)], 0)
    Cs = sp.sparse.csr_matrix(C)
    Cst = Cs.transpose()
    
    # Compute matrix A and b
    A = Ogxt @ w_diag @ Ogx + Ogyt @ w_diag @ Ogy + Cst @ Cs
    b = Ogxt @ w_diag @ Gmx_flat + Ogyt @ w_diag @ Gmy_flat + v_00 * Cst

    start_time = time.time()
    if (solver == "cholesky"):
        factor = cholesky(A)
        x = factor(b)
        x = x.reshape(img.shape, order='F').toarray()
    else:
        x = sparse.linalg.spsolve(A,b)
        x = x.reshape(img.shape, order='F')
        
    time_taken = time.time() -start_time
    print(solver, " time:", time_taken)
    return x

if __name__ == "__main__":
    #TODO: Replace with your own image
    im = io.imread(path.join('images','task4.jpg'), as_gray=True)
    im = skimage.img_as_float(im)

    G = img2grad_field(im)
    Gm = np.sqrt(np.sum(G*G, axis=2))
    w = 1/(Gm + 0.0001)# To avoid pinching artefacts

    # imr = reconstruct_grad_field(G,w,im[0,0], im, "spsolve").clip(0,1)
    imr = reconstruct_grad_field(G,w,im[0,0], im, "cholesky").clip(0,1)

    plt.figure(figsize=(9, 3))

    plt.subplot(131)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(im,cmap='gray')

    plt.subplot(132)
    plt.title('Reconstructed')
    plt.axis('off')
    plt.imshow(imr,cmap='gray')

    plt.subplot(133)
    plt.title('Difference')
    plt.axis('off')
    plt.imshow(imr-im)
    plt.show()

    # plt.savefig('./results/task4.png')

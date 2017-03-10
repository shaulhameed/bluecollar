from skimage.measure import structural_similarity as sim
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util.dtype import dtype_range
from skimage.util.shape import view_as_windows
import glob

def ssim(X, Y, win_size=7,
                          gradient=False, dynamic_range=None):
    if not X.dtype == Y.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if dynamic_range is None:
        dmin, dmax = dtype_range[X.dtype.type]
        dynamic_range = dmax - dmin

    XW = view_as_windows(X, (win_size, win_size))
    YW = view_as_windows(Y, (win_size, win_size))

    NS = len(XW)
    NP = win_size * win_size

    ux = np.mean(np.mean(XW, axis=2), axis=2)
    uy = np.mean(np.mean(YW, axis=2), axis=2)

    # Compute variances var(X), var(Y) and var(X, Y)
    cov_norm = 1 / (win_size ** 2 - 1)
    XWM = XW - ux[..., None, None]
    YWM = YW - uy[..., None, None]
    vx = cov_norm * np.sum(np.sum(XWM ** 2, axis=2), axis=2)
    vy = cov_norm * np.sum(np.sum(YWM ** 2, axis=2), axis=2)
    vxy = cov_norm * np.sum(np.sum(XWM * YWM, axis=2), axis=2)

    R = dynamic_range
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = (v[..., None, None] for v in
                      (2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))

    S = np.mean((A1 * A2) / (B1 * B2))

    if gradient:
        local_grad = 2 / (NP * B1 ** 2 * B2 ** 2) * \
                     (
                         A1 * B1 * (B2 * XW - A2 * YW) - \
                         B1 * B2 * (A2 - A1) * ux[..., None, None] + \
                         A1 * A2 * (B1 - B2) * uy[..., None, None]
                     )

        grad = np.zeros_like(X, dtype=float)
        OW = view_as_windows(grad, (win_size, win_size))

        OW += local_grad
        grad /= NS

        return S, grad

    else:
        return S
images = []
for img in glob.glob("/home/prema/Images4pro/*.png"):
    n= cv2.imread(img)
    images.append(n)
    img1 = cv2.imread(img)
    img2 = cv2.imread("/home/prema/Images4pro/user.2.png")

    img11 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img12 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    imageA = cv2.resize(img11, (100, 100))
    imageB = cv2.resize(img12, (100, 100))

    s = ssim(imageA, imageB,win_size=7,gradient=False,dynamic_range=None)
    title = "Comparing"
    fig = plt.figure(title)
    if s < 0:
        s = 0
        plt.suptitle("Percentage : %.2f " % (s * 100))
    a=int(s*100)
    if a>50:
        print(a)
        print("matched image,")
        print(img)


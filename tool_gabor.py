import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')
import numpy as np
import cv2
import scipy.stats as stats


def grating(X, Y, params):
    """
    GRATING --  Sinusoidal grating (grayscale image).
    %
    %  G = grating(X,Y,params)
    %
    %  X and Y are matrices produced by MESHGRID (use integers=pixels).
    %  PARAMS is a struct w/ fields 'amplit', 'freq', 'orient', and 'phase'.
    %  The function GABOR_PARAMS supplies default parameters.
    %  G is a matrix of luminance values.  size(G)==size(X)==size(Y)
    %
    %  Example:
    %    x=[-100:+100] ; y=[-120:+120] ; [X,Y] = meshgrid(x,y) ;
    %    params = gabor_params ; params.orient = 15*pi/180 ;
    %    G = grating(X,Y,params) ;
    %    imagesc1(x,y,G) ;

    """
    A = params["amplit"]
    omega = 2 * np.pi * params["freq"]
    theta = params["orient"]
    phi = params["phase"]
    slant = X * (omega * np.cos(theta)) + Y * (
        omega * np.sin(theta))  # cf. function SLANT
    G = A * np.cos(slant + phi)
    return G


def gabor(X, Y, params):
    """
    %GABOR  --  Sinusoidal grating under a Gaussian envelope.
    %
    %  G = gabor(X,Y,params)
    %
    %  X and Y are matrices produced by MESHGRID (use integers=pixels).
    %  PARAMS is a struct with fields 'amplit', 'freq', 'orient', 'phase',
    %  and 'sigma'. The function GABOR_PARAMS supplies default parameters.
    %  G is a matrix of luminance values.  size(G)==size(X)==size(Y)
    %
    %  Example:
    %    x=[-100:+100] ; y=[-120:+120] ; [X,Y] = meshgrid(x,y) ;
    %    params = gabor_params ; params.orient = 60*pi/180 ;
    %    G = gabor(X,Y,params) ;
    %    imagesc1(x,y,G) ;

    """

    sigmasq = params["sigma"] ** 2
    Gaussian = np.exp(-(X ** 2 + Y ** 2) / (2 * sigmasq))
    Grating = grating(X, Y, params)
    G = Gaussian * Grating
    return G


class Gabor:
    def __init__(self, sigma=25, freq=0.2):
        self.params = {
            # amplitude [luminance units], min=-A,max=+A  # contrast
            "amplit": 0.5,
            "freq": freq,  # spatial frequency [cycles/pixel]
            "orient": 0,  # orientation [radians]
            "phase": 1.5708,  # phase [radians]
            "sigma": sigma,  # std.dev. of Gaussian envelope [pixels]
        }

    def genGabor(self, size, orient):
        """
        :param size:
        :param orient: 0~360
        :return: G
        """
        x = np.arange(-size[1] // 2, size[1] // 2)
        y = np.arange(-size[0] // 2, size[0] // 2)
        X, Y = np.meshgrid(x, y)
        self.params["orient"] = orient * np.pi / 180
        self.G = gabor(X, Y, self.params)
        return self.G

    def showGabor(self):
        plt.figure()
        plt.imshow(self.G)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()


class Vernier:
    def __init__(self, sigma=25, freq=0.02, var_noise=0.5, phase=0, contrast=0.5):
        self.params = {
            # amplitude [luminance units], min=-A,max=+A    # contrast
            "amplit": contrast,
            "freq": freq,  # spatial frequency [cycles/pixel]
            "orient": 0,  # orientation [radians]
            "phase": phase,  # phase [radians]
            "sigma": sigma,  # std.dev. of Gaussian envelope [pixels]
            # different of two gabor for vernier
            "diff_level": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            # var_noise: variation of diff level --> noise in this version only can be 0.5
            "var_n": var_noise
        }

    def genVernier(self, size, orient, diff, label, var_noise):
        """
        :param size:
        :param orient: "V, H" or 0-180
        :param diff: 0,1,2 (easy medium hard)
        :param label: +1 or -1
        :param var_noise variation of diff level --> var:0.5
        :return: G
        """
        if orient == 'V':  
            _orient = 45
        elif orient == 'H':
            _orient = 135
        else:
            _orient = orient
        # generate Gabor
        x = np.arange(-size[1] // 4, size[1] // 4)
        y = np.arange(-size[0] // 4, size[0] // 4)
        X, Y = np.meshgrid(x, y)
        # ori of gabor is 0  
        self.params["orient"] = 0 * np.pi / 180
        self.params["var_n"] = var_noise
        G = gabor(X, Y, self.params)

        # arrange Gabor into Vernier
        diff_noi_func = stats.truncnorm(-1/self.params["var_n"], 1/self.params["var_n"],
                                        loc=0, scale=self.params["var_n"])  
        # diff_noi = np.abs(np.around(np.random.normal(0,self.params["var_n"])))
        diff_noi = np.around(diff_noi_func.rvs(1))
        jitter = int((self.params["diff_level"][diff] + diff_noi) * label)

        self.V = np.zeros(size)
        self.V[0:size[0] // 2, size[1] // 4-jitter:size[1]
               * 3 // 4-jitter] = G    
        self.V[size[0] // 2:, size[1] // 4 +
               jitter:size[1] * 3 // 4 + jitter] = G
        # pdb.set_trace()
        M = cv2.getRotationMatrix2D((size[0]//2, size[1]//2), _orient, 1)
        self.V = cv2.warpAffine(self.V, M, (size[0], size[1]))
        return self.V

    def show(self):
        plt.figure()
        plt.imshow(self.V)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

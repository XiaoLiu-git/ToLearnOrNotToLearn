import matplotlib.pyplot as plt
import numpy as np
from tool_gabor import Vernier

plt.style.use("fivethirtyeight")


class GenImg:
    def __init__(self, size=None, orient='H', loc="L", noise_cutout=0.7, diff=2,
                 var_noise=0.5, freq=0.02, phase=0, contrast=0.6, label_reverse=False):
        """
        :param size: Defaults to [400, 200]
        :param orient: H(horizontal) or V(vertical)
        :param loc: "L" or "R"
        :param noise_cutout: 0 ~ 19
        :param diff: 0-2 Hard-->Easy
        :param var_noise: variation of diff level --> var_noise: 0.5 in this version only can be 0.5
        :param freq:0.02
        :param phase:0
        """

        if size is None:
            size = [400, 200]
        self.w, self.h = size
        self.orient = orient
        self.loc = loc
        self.noise_cutout = noise_cutout
        self.diff = diff
        self.var_n = var_noise
        self.vn = Vernier(freq=freq, phase=phase, contrast=contrast)
        self.contrast = contrast
        self.label_reverse = label_reverse

    def gen_reference(self, diff=None):
        # self.label_reverse 相关未改
        if diff is not None:
            self.diff = diff

        self.label = 0
        vernier = self.vn.genVernier([200, self.h], self.orient,
                                     self.diff, self.label, self.var_n)
        # self.tg = (np.random.randn(self.w,
        #                           self.h) - 0.5) * 2 * self.noise_cutout + 1
        self.tg = (np.random.randn(self.w,
                                   self.h)) * 2 * self.noise_cutout + 1
        if self.loc == "L" :
            self.tg[:200, :] += vernier
        elif self.loc == "R" :
            self.tg[200:200*2, :] += vernier
        else:
            self.tg[200*(self.loc-1):200*self.loc, :] += vernier

        return self.tg

    def gen_train(self, exposure=False, diff=None):
        """generate 1 complete input stimulus image, whose corresponding label is randomly 1/-1. 
        Returns:
            self.label: one label
            self.tg: the correspond image
        """
        if diff is not None:
            self.diff = diff

        self.label = np.sign(np.random.rand(1) - 0.5)
        vernier = self.vn.genVernier([200, self.h], self.orient,
                                     self.diff, self.label, self.var_n)
        # self.tg = (np.random.randn(self.w,
        #                           self.h) - 0.5) * 2 * self.noise_cutout + 1
        self.tg = (np.random.randn(self.w,
                                   self.h)) * 2 * self.noise_cutout + 1
                                   
        if self.loc == "L" :
            self.tg[:200, :] += vernier
        elif self.loc == "R" :
            self.tg[200:200*2, :] += vernier
        else:
            self.tg[200*(self.loc-1):200*self.loc, :] += vernier

        if exposure:
            exposure_label = np.sign(np.random.rand(1) - 0.5)
            exposure_vernier = self.vn.genVernier([self.w // 2, self.h], 'H',
                                                  self.diff, exposure_label, self.var_n)
            self.tg[self.w // 2:, :] += exposure_vernier

        if self.label_reverse:
            self.label = -self.label
        return self.label, self.tg

    def gen_test(self, diff=None):
        """generate 1 pair of complete input stimulus image.
        Returns:
            self.tg_p: one positive image
            self.tg_n: one negative image
        """
        if diff is not None:
            self.diff = diff

        self.label = 1
        vernier_p = self.vn.genVernier([200, self.h], self.orient,  # self.w // 2 因为vernier只占一半地方
                                       self.diff, self.label, self.var_n)
        # self.tg_p = (np.random.randn(self.w,
        #                           self.h) - 0.5) * 2 * self.noise_cutout + 1    #first add noise
        self.tg_p = (np.random.randn(self.w,
                                     self.h)) * 2 * self.noise_cutout + 1  # first add noise
        self.label = -1
        vernier_n = self.vn.genVernier([200, self.h], self.orient,
                                       self.diff, self.label, self.var_n)
        # self.tg_n = (np.random.randn(self.w,
        #                             self.h) - 0.5) * 2 * self.noise_cutout + 1
        self.tg_n = (np.random.randn(self.w,
                                     self.h)) * 2 * self.noise_cutout + 1

        if self.loc == "L" :
            self.tg_p[:200, :] += vernier_p
            self.tg_n[:200, :] += vernier_n
        elif self.loc == "R" :
            self.tg_p[200:200*2, :] += vernier_p
            self.tg_n[200:200*2, :] += vernier_n
        else:
            self.tg_p[200*(self.loc-1):200*self.loc, :] += vernier_p
            self.tg_n[200*(self.loc-1):200*self.loc, :] += vernier_n

        if self.label_reverse:
            return (-1, self.tg_p), (1, self.tg_n)
        else:
            return (1, self.tg_p), (-1, self.tg_n)
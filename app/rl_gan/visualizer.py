import numpy as np
import os
import ntpath
import time
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt 
matplotlib.use('Agg')

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    image_pil.save(image_path)
    # plt.imshow(image_numpy)
    # plt.savefig(image_path)
    # plt.clf()
    # plt.close()

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.port_id)


        
    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, iter=0):
        if self.display_id > 0: # show images in the browser   
            idx = 1         
            cmap = get_cmap(len(visuals))
            for i, (label, item) in enumerate(visuals.items()):
                clr = cmap(i)
                if 'pc' in label:
                    color_arr = np.repeat([[int(each*255) for each in clr[:3]]], len(item), axis=0)
                    self.vis.scatter(item,
                                     Y=None,
                                     opts=dict(title=label + str(iter), markersize=0.5,  markercolor=color_arr),
                                     win=self.display_id + iter)
                elif 'img' in label:
                    # the transpose: HxWxC -> CxHxW
                    self.vis.image(np.transpose(item, (2,0,1)), opts=dict(title=label),
                                   win=self.display_id + idx + iter)


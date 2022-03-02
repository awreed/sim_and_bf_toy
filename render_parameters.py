import torch
import numpy as np
from waveform_processing import hilbert_torch
import scipy.signal
import math
from utils import save_sas_plot
import matplotlib.pyplot as plt
from transducer import Transducer

class RenderParameters:
    def __init__(self, **kwargs):
        self.Fs = kwargs.get('Fs', 100000) * 1.0
        self.num_samples = None # Computed under generateTransmitSignal()
        self.data = {}
        self.num_samples_transmit = None

        self.dev = kwargs.get('device', None)

        # Will be used to create torch constant, not differentiable at this time
        self.f_start = kwargs.get('f_start', 30000)  # Chirp start frequency
        self.f_stop = kwargs.get('f_stop', 10000)  # Chirp stop frequency
        self.t_start = kwargs.get('t_start', 0)  # Chirp start time
        self.t_stop = kwargs.get('t_stop', .001)  # Chirp stop time
        self.win_ratio = kwargs.get('win_ratio', 0.1)  # Tukey window ratio for chirp
        self.c = kwargs.get('c', 339.5)

        self.transmit_signal = None  # Transmitted signal (Set to torch type)
        self.pulse = None  # Hilbert transform of the transmitted signal
        self.fft_pulse = None  # FFT of the hilbert transform of transmitted signal
        self.scene = None  # Stores the processed .obj file
        self.theta_start = None  # Projector start angle in degrees
        self.theta_stop = None  # Projector stop angle in degrees
        self.theta_step = None  # Projector step angle in degrees
        self.num_thetas = None  # Projector number of theta positions
        self.num_proj = None  # Number of projectors
        self.projectors = None  # Array of projector 3D coordinates in meters (Set to torch type)
        self.thetas = None  # Array of theta values in degrees

        self.scene_dim_x = None # Ensonified scene dimensions [-x, x]
        self.scene_dim_y = None # Ensonified scene dimensions [-y, y]
        self.scene_dim_z = None # Ensonified scene dimensions [-z, z]
        self.pix_dim = None # N pixels in beamformed image in format [x, y, z]
        self.x_vect = None # Vector of scene pixels x positions
        self.y_vect = None # Vector of scene pixel y positions
        self.z_vect = None # Vector of scene pixel z positions
        self.z_TX = None
        self.Z_RX = None
        self.num_pix_3D = None
        self.scene_center = None # Center of the ensonified scene
        self.pix_pos = None
        self.hydros = None
        self.nf = 128
        self.pixels_3D_bf = None
        self.pixels_3D_sim = None
        self.perturb = False
        self.min_dist = 0.00

    def define_scene_dimensions(self, **kwargs):
        self.scene_dim_x = kwargs.get('scene_dim_x', None)
        self.scene_dim_y = kwargs.get('scene_dim_y', None)
        self.scene_dim_z = kwargs.get('scene_dim_z', None)

        # Beamform dimensions
        self.pix_dim_bf = kwargs.get('pix_dim_bf', None)

        self.circle = kwargs.get('circle', False)

        """

        clamp = lambda x : max(min(x, 100), -100)

        self.x_shift = clamp(kwargs.get('x_shift', 0))
        self.y_shift = clamp(kwargs.get('y_shift', 0))

        # Percentages, somewhere in -1 and 1
        self.x_shift = self.x_shift / 100
        self.y_shift = self.y_shift / 100

        print(self.pix_dim_bf[0])

        psize_x = (np.abs(self.scene_dim_x[0])
            + np.abs(self.scene_dim_x[1]))/(self.pix_dim_bf[0] + 1)

        psize_y = (np.abs(self.scene_dim_y[0])
            + np.abs(self.scene_dim_y[1]))/(self.pix_dim_bf[1] + 1)

        psize_z = (np.abs(self.scene_dim_z[0])
            + np.abs(self.scene_dim_z[1]))/(self.pix_dim_bf[2] + 1)


        start = self.scene_dim_x[0] + psize_x/2
        stop = self.scene_dim_x[1] - psize_x/2

        self.x_vect = np.arange(start, stop, psize_x)

        start = self.scene_dim_y[0] + psize_y/2
        stop = self.scene_dim_y[1] - psize_y/2

        self.y_vect = np.arange(start, stop, psize_y)

        start = self.scene_dim_z[0] + psize_z/2
        stop = self.scene_dim_z[1] - psize_z/2

        print(start, stop)
        if start == 0 and stop == 0:
          self.z_vect = np.array([0.00])
        else:
          self.z_vect = np.arange(start, stop, psize_z)

        # move left or right in voxel
        x_vect_bf = self.x_vect + self.x_shift*(psize_x/2)

        plt.figure()
        plt.scatter(self.x_vect[0:5], np.zeros_like(self.x_vect[0:5]))
        plt.scatter(x_vect_bf[0:5], np.zeros_like(self.x_vect[0:5]))
        plt.savefig('check_vals.png')
        # move up and down in voxel 
        y_vect_bf = self.y_vect + self.y_shift*(psize_y/2)

        """
        assert abs(self.scene_dim_x[0]) == abs(self.scene_dim_x[1])
        assert abs(self.scene_dim_y[0]) == abs(self.scene_dim_y[1])

        x_vect_bf = np.linspace(self.scene_dim_x[0], self.scene_dim_x[1], 
            self.pix_dim_bf[0], endpoint=True)
        y_vect_bf = np.linspace(self.scene_dim_y[0], self.scene_dim_y[1],
             self.pix_dim_bf[1], endpoint=True)
        z_vect_bf = np.linspace(self.scene_dim_z[0], self.scene_dim_z[1],
             self.pix_dim_bf[2], endpoint=True)

        #print(self.x_vect)

        self.num_pix_3D_bf = np.size(x_vect_bf) * np.size(y_vect_bf) *\
             np.size(self.z_vect)

        #self.scene_center = np.array([np.median(x_vect_bf), 
        #    np.median(y_vect_bf), np.median(z_vect_bf)])
        self.scene_center = np.array([0., 0., 0.])

        (x, y, z) = np.meshgrid(x_vect_bf, y_vect_bf, z_vect_bf)

        pixel_grid = np.hstack((np.reshape(x, 
            (np.size(x), 1)), np.reshape(y, (np.size(y), 1)),
            np.reshape(z, (np.size(z), 1))))

        self.circle_indeces = np.where(pixel_grid[..., 0]**2 +\
            pixel_grid[...,1]**2 <= self.scene_dim_x[0]**2)

        self.mask = np.zeros((self.num_pix_3D_bf))
        self.mask[self.circle_indeces] = 1
        self.mask = self.mask.reshape(self.pix_dim_bf[0], self.pix_dim_bf[0])
        print(self.mask.shape)

        #save_sas_plot(mask, 'mask.png')

        pixel_circle = pixel_grid[self.circle_indeces]

        if self.circle:
          self.pixels_3D_bf = torch.from_numpy(pixel_circle)
          self.pixels_3D_sim = self.pixels_3D_bf
        else:
          self.pixels_3D_bf = torch.from_numpy(pixel_grid)
          self.pixels_3D_sim = self.pixels_3D_bf


        #self.pixels_3D_bf = torch.from_numpy(np.hstack((np.reshape(x, 
        #        (np.size(x), 1)), np.reshape(y, (np.size(y), 1)),
        #                           np.reshape(z, (np.size(z), 1)))))
    

        # Simulation dimensions
        #self.pix_dim_sim = kwargs.get('pix_dim_sim', None)
        #self.perturb = kwargs.get('perturb', True)

        
        #self.x_vect = np.linspace(self.scene_dim_x[0], self.scene_dim_x[1], 
        #    self.pix_dim_sim[0])
        #if len(self.x_vect) > 1:
        #    x_dist = np.abs(self.x_vect[0] - self.x_vect[1])
        #else:
        #    x_dist = 0
        #x_noise = np.random.uniform(low=-x_dist/2, high=x_dist/2, 
        #    size=self.x_vect.shape) 
        
        #self.y_vect = np.linspace(self.scene_dim_y[0], self.scene_dim_y[1],
        #     self.pix_dim_sim[1])

        #if len(self.y_vect) > 1:
        #    y_dist = np.abs(self.y_vect[0] - self.y_vect[1])
        #else:
        #    y_dist = 0
        #y_noise = np.random.uniform(low=-y_dist/2, high=y_dist/2, 
        #    size=self.y_vect.shape) 

        #self.z_vect = np.linspace(self.scene_dim_z[0], self.scene_dim_z[1],
        #     self.pix_dim_sim[2])

        #if len(self.z_vect) > 1:
        #    z_dist = np.abs(self.z_vect[0] - self.z_vect[1])
        #else:
        #    z_dist = 0
        #z_noise = np.random.uniform(low=-z_dist/2, high=y_dist/2, 
        #    size=self.z_vect.shape)

        #self.num_pix_3D_sim = np.size(self.x_vect) * np.size(self.y_vect) *\
        #     np.size(self.z_vect)

        #(x, y, z) = np.meshgrid(self.x_vect, self.y_vect, self.z_vect)
        #self.pixels_3D_sim = np.hstack((np.reshape(x, 
        #        (np.size(x), 1)), np.reshape(y, (np.size(y), 1)),
        #                           np.reshape(z, (np.size(z), 1))))

        #if self.perturb:
        #  self.pixels_3D_sim[:, 0:2] = self.pixels_3D_sim[:, 0:2]
        #  + np.random.uniform(low=-y_dist/2, high=y_dist/2,
        #      size=self.pixels_3D_sim[:, 0:2].shape)

        #self.pixels_3D_sim = torch.from_numpy(self.pixels_3D_sim)
        #self.pixels_3D_sim = self.pixels_3D_bf


    def generate_transmit_signal(self,wfm=None, **kwargs):

      crop_wfm = kwargs.get('crop_wfm', False)
      pixels_3D_sim = self.pixels_3D_sim.detach().cpu().numpy()

      # Find min and max time of flight to edges of scene
      if crop_wfm:
        pixels_3D_sim = self.pixels_3D_sim.detach().cpu().numpy()
        
        edge_indeces = np.where(
          (pixels_3D_sim[:, 0] == min(pixels_3D_sim[:, 0])) |
          (pixels_3D_sim[:, 0] == max(pixels_3D_sim[:, 0])) |
          (pixels_3D_sim[:, 1] == min(pixels_3D_sim[:, 1])) |
          (pixels_3D_sim[:, 1] == max(pixels_3D_sim[:, 1])) 
        )

        edges = pixels_3D_sim[edge_indeces]

        min_dist = []
        max_dist = []

        for trans in self.trans:
          tx = trans.tx_pos.detach().cpu().numpy()[None, ...]
          rx = trans.rx_pos.detach().cpu().numpy()[None, ...]

          dist1 = np.sqrt(np.sum((edges - tx)**2, 1))
          dist2 = np.sqrt(np.sum((edges - rx)**2, 1))

          dist = dist1 + dist2

          min_dist.append(np.min(dist))
          max_dist.append(np.max(dist))
        
        # pad min and max distance by a bit
        self.min_dist = min(min_dist) - .05 # (m)
        self.max_dist = max(max_dist) + .05 # (m)

        assert self.max_dist > self.min_dist

        t_dur = (self.max_dist - self.min_dist) / self.c

        self.num_samples = math.ceil(t_dur * self.Fs)

        #print("Min dist", self.min_dist)
        #print("Max dist", self.max_dist)

      else:
        self.num_samples = 1000
        self.min_dist = 0.0

      if not wfm:
          #print("using the analytic transmit waveform")
          times = np.linspace(self.t_start, self.t_stop - 1 / self.Fs, num=int((self.t_stop - self.t_start) * self.Fs))
          LFM = scipy.signal.chirp(times, self.f_start, self.t_stop, self.f_stop)  # Generate LFM chirp
          ind1 = 0  # Not supporting staring time other than zero atm
          ind2 = ind1 + len(LFM)
            
          # If scene smaller than transmit signal length
          self.num_samples = self.num_samples + 2*len(LFM)
          #print("Num samples", self.num_samples)
          sig = np.full(int(self.num_samples), 1e-8)

          sig[ind1:ind2] = LFM  # Insert chirp into receive signal
      else:
          #print("using the measured transmit waveform")
          #print("Num samples", self.num_samples)
          sig = np.full(int(self.num_samples), 1e-8)

          sig[:len(wfm)] = np.array(wfm).squeeze()
          LFM = np.array(wfm).squeeze()

      sig = torch.from_numpy(sig)

      self.pulse_fft_kernel = torch.fft.fft(hilbert_torch(sig))

      # Used to build received waveform
      LFM = torch.from_numpy(LFM)
      self.transmit_signal = LFM
      self.num_samples_transmit = len(LFM)


    def define_transducer_pos(self, **kwargs):
        self.theta_start = kwargs.get('theta_start', 0)
        self.theta_stop = kwargs.get('theta_stop', 359)
        self.theta_step = kwargs.get('theta_step', 1)
        self.r = kwargs.get('r', None)
        self.z_TX = kwargs.get('z_TX', None)
        self.z_RX = kwargs.get('z_RX', None)

        self.thetas = range(self.theta_start, self.theta_stop, self.theta_step)

        self.num_thetas = len(self.thetas)

        trans = []

        # Pack every projector position into an array
        for i in range(0, self.num_thetas):
            tx_pos = torch.tensor(
                [self.r * math.cos(np.deg2rad(self.thetas[i])), self.r * math.sin(np.deg2rad(self.thetas[i])),
                 self.z_TX])
            rx_pos = torch.tensor(
                [self.r * math.cos(np.deg2rad(self.thetas[i])), self.r * math.sin(np.deg2rad(self.thetas[i])),
                 self.z_RX])
            trans.append(Transducer(tx_pos=tx_pos, rx_pos=rx_pos))

        self.num_proj = self.num_thetas
        self.trans = trans

    

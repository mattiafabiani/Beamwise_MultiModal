
from PIL import Image
import random
import pandas as pd
from pykalman import KalmanFilter
import numpy as np
import torch 
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
import torchvision.transforms.functional as F
from scipy import stats
import utm
import cv2

random.seed(100)

class MATTIA_Data(Dataset):
    def __init__(self, root, root_csv, config, test=False, augment=False, flip=False):

        self.root = root
        self.dataframe = pd.read_csv(self.root + root_csv)
        self.seq_len = config.seq_len
        self.test = test
        self.flip = flip
        self.augment = augment
        self.segment = config.segment
        self.gps_features = config.gps_features
        self.n_gps = config.n_gps
        self.crop = config.crop

    def __len__(self):
        """Returns the length of the dataset. """
        return self.dataframe.index.stop

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['front_images'] = []
        data['back_images'] = []
        data['scenario'] = []
        data['loss_weight'] = []
        data['gps'] = []

        add_front_images = []
        add_back_images = []
        instanceidx = ['1','2', '3','4','5']
        
        gps = np.zeros((self.seq_len,self.n_gps,2)) 
        for time_idx in range(self.seq_len):
            gps[time_idx,0,:] = np.loadtxt(self.root + self.dataframe['x'+str(time_idx+1)+'_unit1_gps1'][index])
            gps[time_idx,1,:] = np.loadtxt(self.root + self.dataframe['x'+str(time_idx+1)+'_unit2_gps1'][index])
        
        if self.gps_features:
            data['gps'] = extract_gps_features(gps=gps,seq_len=self.seq_len,flip=self.flip)
        else:
            for i in range(self.n_gps):
                data['gps'].append(torch.from_numpy(gps[:,i,:]))
        
        for stri in instanceidx:
            add_front_images.append(self.root + self.dataframe['x'+stri+'_unit1_rgb5'][index])
            add_back_images.append(self.root + self.dataframe['x'+stri+'_unit1_rgb6'][index])

        # check which scenario is the data sample associated 
        scenarios = ['scenario36', 'scenario37', 'scenario38', 'scenario39']

        for i in range(len(scenarios)): 
            s = scenarios[i]
            if s in self.dataframe['x1_unit1_rgb5'][index]:
                data['scenario'] = s
                break
        
        if self.augment:
            transform = random.randint(1, 7)
            augment_params = {
                'brightness_factor': random.uniform(0.5, 2),
                'contrast_factor': random.uniform(0.5, 4),
                'gamma_factor': random.uniform(0.5, 3),
                'hue_factor': random.uniform(-0.5, 0.5),
                'saturation_factor': random.uniform(0, 4),
                'sharpness_factor': random.uniform(0, 10),
                'apply_blur': random.uniform(0,1),
                'kernel_size_factor': (9, 7),
                'sigma_factor': (3, 5)
            }

        for i in range(self.seq_len):
            front_imgs = np.array(Image.open(add_front_images[i]).resize((self.crop,self.crop)))
            back_imgs = np.array(Image.open(add_back_images[i]).resize((self.crop,self.crop)))
            
            if self.flip:
                front_imgs = np.ascontiguousarray(np.flip(front_imgs,1))
                back_imgs = np.ascontiguousarray(np.flip(back_imgs,1))
            # apply a random image transformation for all the image sequences
            if self.augment:
                front_imgs = Image.open(add_front_images[i]).resize((self.crop,self.crop))
                back_imgs = Image.open(add_back_images[i]).resize((self.crop,self.crop))
                front_imgs = np.array(apply_random_transform(front_imgs,transform,augment_params))
                back_imgs = np.array(apply_random_transform(back_imgs,transform,augment_params))
            if self.segment:
                # load segmented dataset on scenario 37 & 39
                if 'scenario37' in add_front_images[i] or 'scenario39' in add_front_images[i]:
                    seg_front = np.array(
                        Image.open(add_front_images[i][:43] + '_segment/seg_' + add_front_images[i][44:]).resize(
                            (self.crop,self.crop)))
                    seg_back = np.array(
                        Image.open(add_back_images[i][:43] + '_segment/seg_' + add_back_images[i][44:]).resize(
                            (self.crop,self.crop)))
                    
                    # use segmented front images
                    a = seg_front[..., 0]
                    a = a[:, :, np.newaxis]
                    a = np.concatenate([a, a, a], axis=2)
                    seg_car = cv2.bitwise_and(front_imgs, a)
                    front_imgs = cv2.addWeighted(front_imgs, 0.5, seg_car, 0.8, 0)
                    # use segmented back images
                    a = seg_back[..., 0]
                    a = a[:, :, np.newaxis]
                    a = np.concatenate([a, a, a], axis=2)
                    seg_car = cv2.bitwise_and(back_imgs, a)
                    back_imgs = cv2.addWeighted(back_imgs, 0.5, seg_car, 0.8, 0)
            
            front_imgs = normalize_imagenet(front_imgs)
            back_imgs = normalize_imagenet(back_imgs)
            data['front_images'].append(torch.from_numpy(np.transpose(front_imgs, (2, 0, 1))))
            data['back_images'].append(torch.from_numpy(np.transpose(back_imgs, (2, 0, 1))))

        # training labels
        if not self.test:
            data['beam'] = []
            data['beamidx'] = []
            data['beam_pwr'] = []
            data['all_true_beams'] = []
            # gaussian distributed target instead of one-hot
            beamidx = self.dataframe['y1_unit1_overall-beam'][index] - 1
            # beamidx = self.dataframe['y1_unit1_overall-beam'][index]
            # if beamidx == 256:
            #     beamidx = 255
            _start = np.mod(beamidx - 5,256)
            _end = np.mod(beamidx + 5,256)
            x_data = list(range(_start, 256)) + list(range(0, _end)) if _end < _start else list(range(_start,_end))
            y_data = stats.norm.pdf(x_data, beamidx, 0.5)
            data_beam = np.zeros((256))
            data_beam[np.mod(x_data,256)] = y_data * 0.9858202 # after truncation this ensures unitary sum of the new distribution
            if self.flip:
                beamidx = 256-beamidx
                data_beam = np.ascontiguousarray(np.flip(data_beam,0))
            data['beam'].append(data_beam)
            data['beamidx'].append(beamidx)
            
            # load beam power
            y_pwrs = np.zeros((4,64))
            for arr_idx in range(4): # 4 antenna arrays
                y_pwrs[arr_idx,:] = np.loadtxt(self.root + self.dataframe[f'y1_unit1_pwr{arr_idx+1}'][index])
            y_pwrs = y_pwrs.reshape((256)) # N_ARR*N_BEAMS
            data['beam_pwr'].append(torch.from_numpy(y_pwrs))
            all_true_beams = np.flip(np.argsort(y_pwrs, axis=0), axis=0)
            data['all_true_beams'].append(torch.from_numpy(all_true_beams.copy().astype(int)))
            
        return data
    
def apply_random_transform(image_sample,transform_choice,dict_params):
    """
    image_sample shape: (H,W,C)
    Takes as input an image and returns the same image with one of seven
    transformations applied with equal probability.
    """
    if transform_choice == 1:
        img_aug = F.adjust_brightness(image_sample, dict_params['brightness_factor'])
    elif transform_choice == 2:
        img_aug = F.adjust_contrast(image_sample, dict_params['contrast_factor'])
    elif transform_choice == 3:
        img_aug = F.adjust_gamma(image_sample, dict_params['gamma_factor'])
    elif transform_choice == 4:
        img_aug = F.adjust_hue(image_sample, dict_params['hue_factor'])
    elif transform_choice == 5:
        img_aug = F.adjust_saturation(image_sample, dict_params['saturation_factor'])
    elif transform_choice == 6:
        img_aug = F.adjust_sharpness(image_sample, dict_params['sharpness_factor'])
    else:
        img_aug = F.gaussian_blur(image_sample, kernel_size=dict_params['kernel_size_factor'], sigma=dict_params['sigma_factor'])

    return img_aug

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images shape -> (H,W,C)
    """
    x = x.copy().astype(float)
    x[:,:, 0] = (x[:,:, 0]/255.0 - 0.485) / 0.229
    x[:,:, 1] = (x[:,:, 1]/255.0 - 0.456) / 0.224
    x[:,:, 2] = (x[:,:, 2]/255.0 - 0.406) / 0.225
    return x

def xy_from_latlong(lat_long):
    """
    lat_long shape: (time_samples,n_gps,2)
    Requires lat and long, in decimal degrees, in the 1st and 2nd columns.
    Returns same row vec/matrix on cartesian (XY) coords.
    """
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat_long[:,:,0], lat_long[:,:,1])
    return np.stack((x,y), axis=2)

def estimate_positions(input_positions, delta_input, delta_output):
    """
    input positions shape => (5,2)
    out shape => (5,2)
    """
    out_pos = np.zeros((2,))
    x_size = input_positions.shape[0]
    x = delta_input * np.arange(x_size)

    f_lat = interp1d(x, input_positions[:, 0], kind='cubic', fill_value='extrapolate')
    f_lon = interp1d(x, input_positions[:, 1], kind='cubic', fill_value='extrapolate')

    out_pos[0] = f_lat(x[-1] + delta_output)
    out_pos[1] = f_lon(x[-1] + delta_output)

    return out_pos

def calculate_derivative(x,dt):
    return np.diff(x,axis=0) / dt

def extra_seq(x,dt):
    '''
    dt [seconds]
    x shape -> (n_samples,2)
    returns -> (n_samples+1,2)
    After calculating the derivative one sequence sample is lost.
    This function exploits interpolation to derive the next sample, assuming
    that dt is the time (in seconds) between two adjacent GPS measurements.
    '''
    extra_x = estimate_positions(x,dt,dt)
    x_new = np.zeros((x.shape[0]+1,2))
    x_new[:x.shape[0]], x_new[x.shape[0]:] = x, extra_x
    return x_new

def kalman_filter(data):
    '''
    data shape -> ()
    '''    
    initial_state_mean = [data[0,0],
                        0,
                        data[0, 1],
                        0]
    transition_matrix = [[1, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                        [0, 0, 1, 0]]
    kf1 = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean)

    kf1 = kf1.em(data, n_iter=5)
    kf2 = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean,
                    observation_covariance = 10*kf1.observation_covariance,
                    em_vars=['transition_covariance', 'initial_state_covariance'])
    kf2 = kf2.em(data, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(data)
    return smoothed_state_means[:,(0,2)]

def extract_gps_features(gps,seq_len,flip=False):
    """    
    gps shape: (seq_len,n_gps,2)
    Returns features derived from with gps: curvature, relative coordinates, relative velocity and relative acceleration.
    """
    dt = 0.2  # time spacing between two adjacent GPS samples (assumed to be constand due to the absence of info in test set)
    
    ## mean and std of features 
    vel_unit1_mean, vel_unit1_std = np.array([-0.26015284, 0.07839386]), np.array([8.842594165, 13.110609225])
    vel_unit2_mean, vel_unit2_std = np.array([-0.26399123, -0.174639675]), np.array([9.22977422, 13.766921845])
    rel_vel_mean, rel_vel_std = np.array([-0.006897985, 0.07949663]), np.array([1.7108489375, 1.98936632])
    acc_unit1_mean, acc_unit1_std = np.array([0.0053756025, -0.00156347]), np.array([1.76213068, 1.678330725])
    acc_unit2_mean, acc_unit2_std = np.array([0.0038153675, -0.0016251625]), np.array([1.4004151, 1.7065172175])
    rel_acc_mean, rel_acc_std = np.array([-0.00339077, -0.001546065]), np.array([1.574713075, 1.58025691])
    curvature_unit1_mean, curvature_unit1_std = 0.009349174538764625, 0.033239195710574785
    curvature_unit2_mean, curvature_unit2_std = 0.009349174538764625, 0.033239195710574785
    rel_coords_mean, rel_coords_std = np.array([0.24961407, 0.34429007]), np.array([38.34547093, 40.45922917])
    
    ## feature engineering
    # convert raw GPS into UTM coordinates
    gps_xy = xy_from_latlong(gps)
    
    if flip:
        gps_xy[:,1] = -gps_xy[:,1]
    
    # add an extra sample to maintain 5 velocity samples
    _gps1 = extra_seq(gps_xy[:,0,:],dt)
    _gps2 = extra_seq(gps_xy[:,1,:],dt)
    
    # smooth out velocity
    vel_unit1 = kalman_filter( calculate_derivative(_gps1,dt) )
    vel_unit2 = kalman_filter( calculate_derivative(_gps2,dt) )

    _vel1 = extra_seq(vel_unit1,dt)
    _vel2 = extra_seq(vel_unit2,dt)
    
    # compute acceleration for both vehicles
    acc_unit1 = calculate_derivative(_vel1,dt)
    acc_unit2 = calculate_derivative(_vel2,dt)
    
    # set zero curvature if there is at least one sample with zero velocity
    if np.sum(vel_unit1[:,0]**2 + vel_unit1[:,1]**2 == 0) == 0:
        curvature_unit1 = np.abs( vel_unit1[:,0]*acc_unit1[:,1] - vel_unit1[:,1]*acc_unit1[:,0]  ) / (vel_unit1[:,0]**2 + vel_unit1[:,1]**2)**1.5
    else:
        curvature_unit1 = np.zeros(5,)
    if np.sum(vel_unit2[:,0]**2 + vel_unit2[:,1]**2 == 0) == 0:
        curvature_unit2 = np.abs( vel_unit2[:,0]*acc_unit2[:,1] - vel_unit2[:,1]*acc_unit2[:,0]  ) / (vel_unit2[:,0]**2 + vel_unit2[:,1]**2)**1.5
    else:
        curvature_unit2 = np.zeros(5,)
        
    curvature_unit1[curvature_unit1 > 2] = 0
    curvature_unit2[curvature_unit2 > 2] = 0
    curvature = kalman_filter( np.stack((curvature_unit1,curvature_unit2),axis=1) )
    rel_coords = np.diff(gps_xy,axis=1).reshape((seq_len,2))
    
    vel_unit1 = (vel_unit1 - vel_unit1_mean) / vel_unit1_std
    vel_unit2 = (vel_unit2 - vel_unit2_mean) / vel_unit2_std
    # rel_vel = (relative_vel - rel_vel_mean) / rel_vel_std

    acc_unit1 = (acc_unit1 - acc_unit1_mean) / acc_unit1_std
    acc_unit2 = (acc_unit2 - acc_unit2_mean) / acc_unit2_std
    # rel_acc = (relative_acc - rel_acc_mean) / rel_acc_std

    curvature_unit1 = (curvature[:,0] - curvature_unit1_mean) / curvature_unit1_std
    curvature_unit2 = (curvature[:,1] - curvature_unit2_mean) / curvature_unit2_std

    rel_coords = (rel_coords - rel_coords_mean) / rel_coords_std

    gps_list = []
    gps_list.append(torch.from_numpy(rel_coords))
    gps_list.append(torch.from_numpy(np.stack((curvature_unit1,curvature_unit2),axis=1)))
        
    return gps_list
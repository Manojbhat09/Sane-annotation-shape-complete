# Written by Muhammad Sarmad
# Date : 23 August 2018


import sys 
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(ROOT_DIR))

from RL_params_car import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from misc import util
matplotlib.use('Agg')
import glob
# from open3d import *
# from open3d import io
# import open3d as o3d

img_path = os.path.join(ROOT_DIR, "images/")
vis = None
Kitti = False
Argo = True


class Object(object):
    pass

def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.array(pcd.points)

def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def plot_bbox(ax, bbox):
    # plot vertices
    ax.scatter(bbox[:, 0], bbox[:, 1], bbox[:, 2], c='k')

    # list of sides' polygons of figure
    verts = [[bbox[0], bbox[1], bbox[2], bbox[3]],
             [bbox[4], bbox[5], bbox[6], bbox[7]],
             [bbox[0], bbox[1], bbox[5], bbox[4]],
             [bbox[2], bbox[3], bbox[7], bbox[6]],
             [bbox[1], bbox[2], bbox[6], bbox[5]],
             [bbox[4], bbox[7], bbox[3], bbox[0]],
             [bbox[2], bbox[3], bbox[7], bbox[6]]]

    # plot sides
    bbox = Poly3DCollection(verts, linewidths=1, edgecolors='r', alpha=.1)
    bbox.set_facecolor('cyan')
    ax.add_collection3d(bbox)


def within_bbox(point, bbox):
    """Determine whether the given point is inside the given bounding box."""
    bbox = np.array(bbox)
    x = bbox[3, :] - bbox[0, :]
    y = bbox[0, :] - bbox[1, :]
    z = bbox[4, :] - bbox[0, :]
    return ((np.dot(x, bbox[0, :]) <= np.dot(x, point)) and (np.dot(x, point) <= np.dot(x, bbox[3, :])) and
            (np.dot(y, bbox[1, :]) <= np.dot(y, point)) and (np.dot(y, point) <= np.dot(y, bbox[0, :])) and
            (np.dot(z, bbox[0, :]) <= np.dot(z, point)) and (np.dot(z, point) <= np.dot(z, bbox[4, :])))

def getCorners(height,width,length,x,y,z,θ,rotation=True):
    
    corners = np.array([[-length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2],
                        [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2],
                        [0, 0, 0, 0, height , height, height, height]])
    θ = np.pi/2 - θ
    rotMat = np.array([[np.cos(θ) , -np.sin(θ) , 0],
                       [np.sin(θ) ,  np.cos(θ) , 0],
                       [    0     ,     0      , 1]])
    if rotation:
        cornersPos = (np.dot(rotMat,corners)+np.tile([x,y,z],(8,1)).T).transpose()
        corner1,corner2,corner3,corner4,corner5,corner6,corner7,corner8 = cornersPos[0],cornersPos[1],cornersPos[2],cornersPos[3],cornersPos[4],cornersPos[5],cornersPos[6],cornersPos[7]
    else:
        cornersPos = (corners + np.tile([x,y,z],(8,1)).T).transpose()
        corner1,corner2,corner3,corner4,corner5,corner6,corner7,corner8 = cornersPos[0],cornersPos[1],cornersPos[2],cornersPos[3],cornersPos[4],cornersPos[5],cornersPos[6],cornersPos[7]
    
    return list(corner1),list(corner2),list(corner3),list(corner4),list(corner5),list(corner6),list(corner7),list(corner8)


def test_sample(folder_name, frame_no, list_of_boxes):
    global ROOT_DIR
    global vis

    if not vis:
        from visualizer import Visualizer
        settings = Object()
        settings.display_winsize = 8
        settings.display_id = 1
        settings.print_freq = 200
        settings.port_id = 8097
        settings.name = 'pcn'
        vis = Visualizer(settings)

    
    args = get_parameters()
    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")  # for selecting device for chamfer loss
    args.checkpoints_dir = args.save_path
    torch.cuda.set_device(args.gpu_id)

    #os.makedirs(os.path.join('plots'), exist_ok=True)
    #os.makedirs(os.path.join('completions'), exist_ok=True)
    
    if(Kitti):
        data_dir = os.path.join(ROOT_DIR, "..", "test_dataset/{}/bin_data/{:010d}.bin".format(folder_name, int(frame_no)))
    elif(Argo):
        data_dir = os.path.join(ROOT_DIR, "..", "test_dataset/{}/sample/argoverse/lidar/{:03d}.bin".format(folder_name, int(frame_no)))
    #pcd_dir = os.path.join(ROOT_DIR, "..", "test_dataset", folder_name, "cars")
    #bbox_dir = os.path.join(ROOT_DIR, "..", "test_dataset", folder_name, "bboxes")
    results_dir = os.path.join(ROOT_DIR, "..", "test_dataset", "results")
    print("data_dir ", data_dir)
    print(args)

    total_time = 0
    total_points = 0
    plot_id = 0

    c = np.cos
    s = np.sin
    rot_mat = lambda x:[[c(x), -s(x), 0], [s(x), c(x), 0], [0,0,1.0]]
    # avg_ = np.array([-5.5450159e-03, 3.1248237e-05, -1.7545074e-02])
    # var_ = np.array([0.06499612, 0.01349955, 0.00575584])   
    # min_ = np.array([-0.4482678174972534, -0.17956072092056274, -0.12795628607273102])
    # max_ = np.array([0.4482678174972534, 0.17956072092056274, 0.12795628607273102])
    points = np.fromfile(data_dir,dtype=np.float32).reshape((-1,4))[:,:3]
    print("Points look like: ")
    print(points[:10,:])
    #for i, car_id in enumerate(car_ids):
    concat_complete = list()
    for j in range(len(list_of_boxes)):
        car_id = folder_name+ '_frame_%d_car_%d' % (int(frame_no), j)
        cur_box = list_of_boxes[j]
        bbox = list(getCorners(0,cur_box['width']+0.3*cur_box['width'],cur_box['length']+0.3*cur_box['length'],cur_box['center']['x'],cur_box['center']['y'],cur_box['height']/2,-cur_box['angle'],rotation=True))
        bbox = np.array(bbox)

        # original pointcloud
        partial_xyz = np.array([point for point in points if within_bbox(point, bbox)])
        total_points += partial_xyz.shape[0]
        plot_path = os.path.join(results_dir, 'plots', '%s.png' % car_id)
        # plot_pcd_three_views(plot_path, [partial_xyz], ['input'],'%d input points' % partial_xyz.shape[0], [5])

        # Calculate center, rotation and scale
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        # scale = bbox[3, 0] - bbox[0, 0]
        # bbox /= scale
        
        # Scope 0
        # Scale and rotate to make cannonical, Rotate from y axis to x axis
        partial_xyz_center = np.dot(partial_xyz - center, rotation) #/ scale
        partial_xyz_center = np.dot(partial_xyz_center, np.array(rot_mat(90*3.14/180)))

        # Scope 1
        # Ground removal
        partial_xyz_center_gr = partial_xyz_center[partial_xyz_center[:, 2] > min(partial_xyz_center[:, 2])+0.2]

        # Scope 2
        # Height based point removal 
        base_pt = np.min(partial_xyz_center_gr[:, 2])
        ht_pt = base_pt+1.6
        partial_xyz_center_ht = partial_xyz_center_gr[partial_xyz_center_gr[:, 2] < ht_pt]

        max_x, max_y, max_z = np.max(partial_xyz_center_ht, axis=0)
        min_x, min_y, min_z = np.min(partial_xyz_center_ht, axis=0)
        mheight = min_z - max_z
        mheight = mheight if mheight > 1.3 else 1.6
        mcenter_height = max_z + mheight/2
        
        # Scope 3
        # Scale input
        min_ = np.array([-0.4482678174972534, -0.17956072092056274, -0.12795628607273102]) # shapenet specific change according to the traning dataset 
        max_ = np.array([0.4482678174972534, 0.17956072092056274, 0.12795628607273102])
        maxx = np.max(partial_xyz_center_ht, axis=0)
        minn = np.min(partial_xyz_center_ht, axis=0)
        avg_ = np.array([-5.5450159e-03, 3.1248237e-05, -1.7545074e-02])
        var_ = np.array([0.06499612, 0.01349955, 0.00575584])

        partial_xyz_center_ht[:, 2] += 1.6+(1)/2
        # print(mcenter_height)
        # partial_xyz_center_ht[:, 2] -=mcenter_height

        act_mean = np.mean(partial_xyz_center_ht)
        act_var = np.var(partial_xyz_center_ht, axis=0)
        # partial_xyz_center_gr_ht_scaled = avg_ + (partial_xyz_center_ht - act_mean)* var_/act_var

        partial_xyz_center_gr_ht_scaled = ((partial_xyz_center_ht)/(maxx[2] - minn[2]))*(0.38)
        

        # Get output
        start = time.time()
        completion_xyz_center_out = main(args, input_data=partial_xyz_center_gr_ht_scaled)
        if completion_xyz_center_out.shape[0] == 3:
            completion_xyz_center_out = completion_xyz_center_out.transpose()
        total_time += time.time() - start

        # completion_xyz = np.dot(completion_xyz_center * scale, rotation.T) + center
        # Scale output back 
        completion_xyz_center = ((completion_xyz_center_out )/0.295)*(maxx[2] - minn[2])
        # completion_xyz_center = act_mean + (completion_xyz_center_out - avg_)* act_var/var_
        completion_xyz_center[:, 2] -= 1.3+(1)/2
        # completion_xyz_center[:, 2] +=mcenter_height

        # Rotate output back
        completion_xyz_center = np.dot(completion_xyz_center, np.array(rot_mat(-90*3.14/180)))
        # Rotate translate output back
        completion_xyz = np.dot(completion_xyz_center, rotation.T) + center
        pcd_path = os.path.join(results_dir, 'completions_rl', '%s.pcd' % car_id)
        
        if vis:
            print("plotting")
            
            # Scope 0
            # Check axis x and y. Check if at (0,0)
            visuals = OrderedDict([('Partial_pc partial_xyz_center ', partial_xyz_center)]) # ('Complete Predicted_pc ', completion)
            vis.display_current_results(visuals, 0, plot_id)
            plot_id+=1

            # Scope 1
            # Check ground removed points
            visuals = OrderedDict([('Partial_pc partial_xyz_center_gr ', partial_xyz_center_gr)]) # ('Complete Predicted_pc ', completion)
            vis.display_current_results(visuals, 0, plot_id)
            plot_id+=1
            
            # Scope 2
            # Check height removed points
            visuals = OrderedDict([('Partial_pc partial_xyz_center_ht ', partial_xyz_center_ht)]) # ('Complete Predicted_pc ', completion)
            vis.display_current_results(visuals, 0, plot_id)
            plot_id+=1
            
            # Scope 3
            # Check height scaled points
            visuals = OrderedDict([('Partial_pc partial_xyz_center_gr_ht_scaled ', partial_xyz_center_gr_ht_scaled)]) # ('Complete Predicted_pc ', completion)
            vis.display_current_results(visuals, 0, plot_id)
            plot_id+=1

            # Check direct output 
            visuals = OrderedDict([('Complete Predicted_pc ', completion_xyz_center_out)]) 
            vis.display_current_results(visuals, 0, plot_id)
            plot_id+=1
            
            # Check output scaled 
            visuals = OrderedDict([('Complete Predicted_pc ', completion_xyz_center)]) 
            vis.display_current_results(visuals, 0, plot_id)
            plot_id+=1

            # Check translated and rotaed to original poistion
            visuals = OrderedDict([('Complete Predicted_pc ', completion_xyz)]) 
            vis.display_current_results(visuals, 0, plot_id)
            plot_id+=1

            both_scaled = np.concatenate((partial_xyz_center_gr_ht_scaled, completion_xyz_center_out))
            both_final = np.concatenate((partial_xyz_center, completion_xyz))

            visuals = OrderedDict([('Both pcs scaled', both_scaled)]) 
            vis.display_current_results(visuals, 0, plot_id)
            plot_id+=1

            visuals = OrderedDict([('Both pcs original', both_final)]) 
            vis.display_current_results(visuals, 0, plot_id)
            plot_id+=1

        concat_complete.append(completion_xyz)
    
    print('Average # input points:', total_points / len(list_of_boxes))
    print('Average time:', total_time / len(list_of_boxes))
    return concat_complete


def main(args, input_data=None):
    # co_transforms = pc_transforms.Compose([])
    
    # input_transforms = transforms.Compose([
    #     pc_transforms.ArrayToTensor(),
    #     #   transforms.Normalize(mean=[0.5,0.5],std=[1,1])
    # ])

    # target_transforms = transforms.Compose([
    #     pc_transforms.ArrayToTensor(),
    #     #  transforms.Normalize(mean=[0.5, 0.5], std=[1, 1])
    # ])

    """-----------------------------------------------Data Loader----------------------------------------------------"""

    # if (args.net_name == 'auto_encoder' and input_data!=None):

    #     [test_dataset,_] = Datasets.__dict__[args.dataName](input_root=args.dataIncompLarge,
    #                                                                       target_root=None,
    #                                                                       split=1.0,
    #                                                                       net_name=args.net_name,
    #                                                                       input_transforms=input_transforms,
    #                                                                       target_transforms=target_transforms,
    #                                                                       co_transforms=co_transforms,
    #                                                                       give_name = True)



    # if(input_data!=None):
    #     test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                             batch_size=1,
    #                                             num_workers=args.workers,
    #                                             shuffle=True,
    #                                             pin_memory=True)
    test_loader = None

    print('Encoder Model: {0}, Decoder Model : {1}'.format(args.model_encoder,args.model_decoder))
    print('GAN Model Generator:{0} & Discriminator : {1} '.format(args.model_generator,args.model_discriminator))


    network_data_AE = torch.load( args.pretrained_enc_dec )

    network_data_G = torch.load(args.pretrained_G)

    network_data_D = torch.load(args.pretrained_D)

    network_data_Actor = torch.load(args.pretrained_Actor)

    network_data_Critic = torch.load(args.pretrained_Critic)

    model_encoder = models_rl.__dict__[args.model_encoder](args, num_points=2048, global_feat=True,
                                                        data=network_data_AE, calc_loss=False).cuda()
    model_decoder = models_rl.__dict__[args.model_decoder](args, data=network_data_AE).cuda()

    model_G = models_rl.__dict__[args.model_generator](args, data=network_data_G).cuda()

    model_D = models_rl.__dict__[args.model_discriminator](args, data=network_data_D).cuda()

    model_actor = models_rl.__dict__['actor_net'](args, data=network_data_Actor).cuda()

    model_critic = models_rl.__dict__['critic_net'](args, data=network_data_Critic).cuda()

    params = get_n_params(model_encoder)
    print('| Number of Encoder parameters [' + str(params) + ']...')

    params = get_n_params(model_decoder)
    print('| Number of Decoder parameters [' + str(params) + ']...')

    chamfer = ChamferLoss(args)
    nll = NLL()
    mse = MSE(reduction = 'elementwise_mean')
    norm = Norm(dims=args.z_dim)
    
    epoch = 0
    
    test_loss, out_pcd = testRL(input_data, model_encoder, model_decoder, model_G, model_D, model_actor,model_critic,epoch, args, chamfer,nll, mse, norm)
    # if input_data:
    #     test_loss, out_pcd = testRL(input_data, model_encoder, model_decoder, model_G, model_D, model_actor,model_critic,epoch, args, chamfer,nll, mse, norm)
    # else:
    #     test_loss, out_pcd = testRL(test_loader, model_encoder, model_decoder, model_G,model_D, model_actor,model_critic,epoch, args, chamfer,nll, mse, norm)
    print('Average Loss :{}'.format(test_loss))
    return out_pcd


def testRL(test_loader,model_encoder,model_decoder, model_G,model_D,model_actor,model_critic,epoch,args, chamfer,nll, mse,norm):

    model_encoder.eval()
    model_decoder.eval()
    model_G.eval()
    model_D.eval()
    model_actor.eval()
    model_critic.eval()
    epoch_size = 1
    
    env = envs(args, model_G, model_D, model_encoder, model_decoder, epoch_size)

    if type(test_loader) == type(np.array([0])):
        input = torch.Tensor(test_loader).cuda().unsqueeze(0).unsqueeze(0)
        obs = env.agent_input(input)  # env(input, action_rand)
        done = False
        while not done:
            # Action By Agent and collect reward
            action = model_actor(np.array(obs))#policy.select_action(np.array(obs))
            action = torch.tensor(action).cuda().unsqueeze(dim=0)
            new_state, _, reward, done, _, out_pcd = env(input, action, render=True, disp=True)
        print("done")
        print(out_pcd.shape)
        return reward, out_pcd
    else:
        for i, (input,fname) in enumerate(test_loader):
            inp = pruned_pcd_list[i+10]
            inp_pcd = np.fromfile(inp, dtype=np.float64)
            inp_pcd = inp_pcd.reshape((-1, 3))
            input = torch.Tensor(inp_pcd).cuda().unsqueeze(0).unsqueeze(0)
            obs = env.agent_input(input)  # env(input, action_rand)
            done = False
            while not done:
                # Action By Agent and collect reward
                action = model_actor(np.array(obs))#policy.select_action(np.array(obs))
                action = torch.tensor(action).cuda().unsqueeze(dim=0)
                new_state, _, reward, done, _, out_pcd = env(input, action, render=True, disp=True,fname=fname, filenum = str(i))
            print("done")
            print(out_pcd.shape)
        return reward, out_pcd

def save_plt(pcd, label):
    global img_path
    img_path_save = os.path.join(img_path, "test_plt.png")
    Xs = pcd[0,:]
    Ys = pcd[1,:]
    Zs = pcd[2,:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.scatter(Xs, Ys, Zs, cmap=cm.jet)
    ax.set_title(label)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    fig.tight_layout()
    fig.savefig(img_path_save)


class envs(nn.Module):
    def __init__(self,args,model_G,model_D,model_encoder,model_decoder,epoch_size):
        super(envs,self).__init__()
        self.nll = NLL()
        self.mse = MSE(reduction='elementwise_mean')
        self.norm = Norm(dims=args.z_dim)
        self.chamfer = ChamferLoss(args)
        self.epoch = 0
        self.epoch_size =epoch_size
        self.model_G = model_G
        self.model_D = model_D
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.j = 1
        self.i = 0;
        self.figures = 25
        self.attempts = args.attempts
        self.end = time.time()
        self.batch_time = AverageMeter()
        self.lossess = AverageMeter()
        self.attempt_id =0
        self.state_prev = np.zeros([4,])
        self.iter = 0
        self.args = args
        
    def reset(self,epoch_size,figures =3):
        self.j = 1;
        self.i = 0;
        self.figures = figures;
        self.epoch_size= epoch_size
        
    def agent_input(self,input):
        with torch.no_grad():
            input = input.cuda(async=True)
            input_var = Variable(input, requires_grad=True)
            encoder_out = self.model_encoder(input_var, )
            out = encoder_out.detach().cpu().numpy().squeeze()
        return out
    
    def forward(self , input , action , render=False, disp=False,fname=None,filenum = None):
        with torch.no_grad():

            input = input.cuda(async=True)
            input_var = Variable(input, requires_grad=True)
            
            visuals = OrderedDict(
            [('Input_pc',  np.transpose(torch.squeeze(input_var, dim=1).detach().cpu().numpy()[0]) ),
             ('AE Predicted_pc', np.transpose(torch.squeeze(input_var, dim=1).detach().cpu().numpy()[0]) ),
             ('GAN Generated_pc', np.transpose(torch.squeeze(input_var, dim=1).detach().cpu().numpy()[0]) )])
            
            encoder_out = self.model_encoder(input_var, )
            pc_1 = self.model_decoder(encoder_out)
            # Generator Input
            z = Variable(action, requires_grad=True).cuda()

            # Generator Output
            out_GD, _ = self.model_G(z)
            out_G = torch.squeeze(out_GD, dim=1)
            out_G = out_G.contiguous().view(-1, self.args.state_dim)

            # Discriminator Output
            out_D, _ = self.model_D(out_GD)

            # H Decoder Output
            pc_1_G = self.model_decoder(out_G)

            # Preprocesing of Input PC and Predicted PC for Visdom
            trans_input = torch.squeeze(input_var, dim=1)
            trans_input = torch.transpose(trans_input, 1, 2)
            trans_input_temp = trans_input[0, :, :]
            pc_1_temp = pc_1[0, :, :] # D Decoder PC
            pc_1_G_temp = pc_1_G[0, :, :] # H Decoder PC

        # Discriminator Loss
        loss_D = 0.01*self.nll(out_D)

        # Loss Between Noisy GFV and Clean GFV
        loss_GFV = 10*self.mse(out_G, encoder_out)

        # Norm Loss
        loss_norm = 0.1*self.norm(z)

        # Chamfer loss
        loss_chamfer = 100*self.chamfer(pc_1_G, trans_input) #self.chamfer(pc_1_G, pc_1)  # instantaneous loss of batch items

        # States Formulation
        state_curr = np.array([loss_D.cpu().data.numpy(), loss_GFV.cpu().data.numpy()
                                  , loss_chamfer.cpu().data.numpy(), loss_norm.cpu().data.numpy()])

        reward_D = state_curr[0] #state_curr[0] - self.state_prev[0]
        reward_GFV =-state_curr[1] # -state_curr[1] + self.state_prev[1]
        reward_chamfer = -state_curr[2] #-state_curr[2] + self.state_prev[2]
        reward_norm =-state_curr[3] # - state_curr[3] + self.state_prev[3]
        # Reward Formulation
        reward = reward_D + reward_GFV + reward_chamfer 

        # measured elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()
        
        test1 = trans_input_temp.detach().cpu().numpy()
        test2 = pc_1_temp.detach().cpu().numpy()
        test3 = pc_1_G_temp.detach().cpu().numpy()
        # fname = fname[0]
        # if  not os.path.exists('test_car/'+fname[:8]):
        #     os.makedirs('test_car/'+fname[:8])

        Gan_generated_out = pc_1_G_temp.detach().cpu().numpy()
        
        save_plt(Gan_generated_out, "GAN generated")
        print('[{4}][{0}/{1}]\t Reward: {2}\t States: {3}'.format(self.i, self.epoch_size,reward,state_curr,self.iter))
        self.i += 1

        done = True
        state = out_G.detach().cpu().data.numpy().squeeze()
        return state, _, reward, done, self.lossess.avg, Gan_generated_out



if __name__ == '__main__':
    args = get_parameters()
    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")  # for selecting device for chamfer loss
    args.checkpoints_dir = args.save_path
    torch.cuda.set_device(args.gpu_id)
    print(args)
    #main(args)
    drivename = "0_drive_0064_sync"
    fname = "0000000500"
    bbox_dir = os.path.join(ROOT_DIR, "..", "test_dataset", drivename)
    files_path = os.path.join(bbox_dir, "bboxes", drivename+"_frame_{}_car_*".format(int(fname)))
    listt = glob.glob( files_path )
    import pdb; pdb.set_trace()
    bbox = np.loadtxt(listt[0])
    test_sample(drivename, fname, bbox)








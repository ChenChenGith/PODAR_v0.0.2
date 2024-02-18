# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/11/12
# @Author  : Chen Chen (Tsinghua Univ.)
# @FileName: PODAR.py
# =====================================

from dataclasses import dataclass, field
import numpy as np
from shapely.geometry import Polygon
from math import cos, sin, pi, atan2, sqrt
from typing import List, Tuple, Dict


@dataclass
class Veh_obj:
    """Basic class of a object
    risk will be assessed and saved to each instance of this class
    """
    # basic information
    name: str = None
    id: int = 0
    type: str = 'car'  # vehicle type, determine shapes and sensitive 
    x0: float = 0.  # position X, unit: m
    y0: float = 0.  # position Y, unit: m
    speed: float = 0.  # speed, unit: m/s
    phi0: float = 0.  # heading angle, to right is zero. unit: rad
    a0: float = 0.  # acceleration, unit: m/s^2
    phi_a0: float = 0.  # yaw rate, nuit: rad/s
    mass: float = -1.  # mass, unit: ton
    sensitive: float = -1.  # how vulnerable the obj is
    length: float = -1.  # shape L
    width: float = -1.  # shape W
    max_dece: float = 7.5  # maximum deceleration values, unit: m/s^2
    
    # intermediate variables from trajectory prediction
    x_pred: np.ndarray = np.ndarray(0, )  # predicted position X
    y_pred: np.ndarray = np.ndarray(0, )  # predicted position Y
    v_pred: np.ndarray = np.ndarray(0, )  # predicted speed
    phi_pred: np.ndarray = np.ndarray(0, )    # predicted heading angle
    virtual_center_x: np.ndarray = np.ndarray(0, )  # surrogate point, for surr. veh., it is the center of rear bumper; for ego veh., it is the centers of rear/front bumper
    virtual_center_y: np.ndarray = np.ndarray(0, )
    
    # results of risk assessment
    risk: float = 0.  # final determined risk values, scalar
    collided: bool = False  # if will be a collision in predicted horizon
    rc: bool = False  # if a collision occurs at current (real collision)
    risk2obj: Dict[str, list] = field(default_factory=dict)
    
    # intermediate variables from risk assessment
    future_position: List[Polygon] = field(default_factory=list)  # list of Shapely Polygon obj. to calculate the minimum distance between ego and surr. veh.
    risk_curve: np.ndarray = np.ndarray(0, )  # predicted risk values in predicted horizon
    dis_t: np.ndarray = np.ndarray(0, )  # distance between ego and surr. veh. in predicted horizon
    damage: np.ndarray = np.ndarray(0, )  # predicted collision damage in predicted horizon
    dis_de_curve: np.ndarray = np.ndarray(0, )  # spatial attenuation curve in predicted horizon
    weight_t: np.ndarray = np.ndarray(0, )  # temporal attenuation curve in predicted horizon
    delta_v: np.ndarray = np.ndarray(0, )  # speed differences in predicted horizon
    abs_v: np.ndarray = np.ndarray(0, )  # speed amplitude of the two vehicle in predicted horizon
    attention: float = 1  # useless
    max_risk_step: int = None  # the index of maximum risk value in risk_curve
    max_risk_source: int = None
    
    # used for render: draw a rectangle to represent a vehicle (use matplotlib)
    rect_x: float = None  # left-bottom point X
    rect_y: float = None  # left-bottom point Y
    rect_phi: float = None  

    def __post_init__(self):
        """set default parameters
        """
        assert self.type in ['car', 'tru', 'bic', 'ped', "toBeDetermined"], "vehicle type should be one of {'car', 'tru', 'bic', 'ped'}"
        if self.type == "toBeDetermined":
            if self.length < 1 and self.width < 1:  # ped
                if self.mass == -1.: self.mass = 0.07
                if self.sensitive == -1.: self.sensitive = 50
                if self.length == -1.: self.length = 0.6
                if self.width == -1.: self.width = 0.6
            elif self.length < 2.5:  # bic
                if self.mass == -1.: self.mass = 0.09
                if self.sensitive == -1.: self.sensitive = 50
                if self.length == -1.: self.length = 1.65
                if self.width == -1.: self.width = 0.7
            elif self.length < 6.5:  # car
                if self.mass == -1.: self.mass = 1.8
                if self.sensitive == -1.: self.sensitive = 1
                if self.length == -1.: self.length = 4.5
                if self.width == -1.: self.width = 1.8
            else:  # truck
                if self.mass == -1.: self.mass = 4.5
                if self.sensitive == -1.: self.sensitive = 1
                if self.length == -1.: self.length = 6
                if self.width == -1.: self.width = 1.9
        elif self.type == 'car':
            if self.mass == -1.: self.mass = 1.8
            if self.sensitive == -1.: self.sensitive = 1
            if self.length == -1.: self.length = 4.5
            if self.width == -1.: self.width = 1.8
        elif self.type == 'tru':
            if self.mass == -1.: self.mass = 4.5
            if self.sensitive == -1.: self.sensitive = 1
            if self.length == -1.: self.length = 6
            if self.width == -1.: self.width = 1.9
        elif self.type == 'bic':
            if self.mass == -1.: self.mass = 0.09
            if self.sensitive == -1.: self.sensitive = 50
            if self.length == -1.: self.length = 1.65
            if self.width == -1.: self.width = 0.7
        elif self.type == 'ped':
            if self.mass == -1.: self.mass = 0.07
            if self.sensitive == -1.: self.sensitive = 50
            if self.length == -1.: self.length = 0.6
            if self.width == -1.: self.width = 0.6

    def update(self, **kwargs):
        """update related parameters during calculation
        """
        for key, val in kwargs.items():
            assert key in vars(self), '{} is not a class attr'.format(key)
            exec("self.{0}=val".format(key), {'self': self, 'val': val})

    def init_rect(self):
        '''计算绘图用初始位置矩形参数'''
        veh_beta = atan2(self.width / 2, self.length / 2)
        veh_r = np.sqrt(self.width ** 2 + self.length ** 2) / 2
        self.rect_x = self.x0 - veh_r * cos(self.phi0 + veh_beta)
        self.rect_y = self.y0 - veh_r * sin(self.phi0 + veh_beta)
        self.rect_phi = self.phi0 / pi * 180.

@dataclass
class Vehicles:
    """used to deal risk assessment for ego vehicle
    """
    ego: Veh_obj = field(default_factory=Veh_obj)
    obj: List[Veh_obj] = field(default_factory=list)
    step_interval=0.1

    def set_ego(self, type, **kwargs):
        """set ego vehicle information
        """
        _o = Veh_obj(type=type)
        for key, val in kwargs.items():
            assert key in vars(_o), '{} is not a class attr'.format(key)
            exec("_o.{0}={1}".format(key, val))
        traj_predition(_o, step_interval=self.step_interval)
        get_future_position_shapely(_o, ego_flag=True)
        _o.init_rect()
        self.ego = _o        

    def add_obj(self, type, **kwargs):
        """add surr. vehicle
        """
        _o = Veh_obj(type=type, id=len(self.obj))
        for key, val in kwargs.items():
            assert key in vars(_o), '{0} is not a class attr'.format(key)
            exec("_o.{0}={1}".format(key, val))
        traj_predition(_o, step_interval=self.step_interval)
        get_future_position_shapely(_o)
        assert self.ego.type != None, 'Please add a ego vehicle first'
        get_risk_to_obj(self.ego, _o)
        _o.init_rect()
        self.obj.append(_o)
        
    def update_obj(self, objID, **kwargs):  
        """update surr. vehicle
        """
        _o = self.obj[objID]
        for key, val in kwargs.items():
            assert key in vars(_o), '{0} is not a class attr'.format(key)
            exec("_o.{0}={1}".format(key, val))
        traj_predition(_o, step_interval=self.step_interval)
        get_future_position_shapely(_o)
        assert self.ego.type != None, 'Please add a ego vehicle first'
        get_risk_to_obj(self.ego, _o)

    def reset(self):
        self.ego: Veh_obj = field(default_factory=Veh_obj)
        self.obj.clear()
    
    def estimate_risk(self):
        """run the risk evaluation

        Parameters
        ----------
        vehs : Vehicles

        Returns
        -------
        Tuple[risk, if_pred_colli, if_true_colli]
        """
        if len(self.obj) == 0: return (0, 0, 0)  # no surrounding vehicle

        risk_ = [r.risk for r in self.obj]
        collided_ = [r.collided for r in self.obj]
        rc_ = [r.rc for r in self.obj]
        risk = np.max(risk_)  # the max risk value among all vehicles is regarded as final risk
        collided = True if np.sum(collided_) > 0 else False
        rc = True if np.sum(rc_) > 0 else False

        source = np.argmax(risk_)
        self.ego.update(risk=risk, collided=collided, rc=rc, max_risk_source=source)

        return (risk, collided, rc)

    def get_risk_in_stru(self):
        _ = []
        for ov in self.obj:
            _.append(
                [self.ego.speed, sqrt(ov.x0 ** 2 + ov.y0 ** 2), ov.type, ov.phi0, ov.phi0 / pi * 180,
                 ov.x0, ov.y0, ov.speed, ov.speed * 3.6, ov.risk, ov.risk_curve, ov.collided, ov.rc])
            # columns=['ego_speed', 'r', 'type', 'phi', 'phi_de', 'x', 'y', 'ov_speed', 'ov_speed_km', 'risk', 'risk_curve', 'collided', 'rc']
        return _



def traj_predition(veh: Veh_obj, step_interval=0.1):
    """predict the future position and heading angle of an object
    prediciton horzion is 4 second

    Parameters
    ----------
    veh : Veh_obj
        ego or surr. veh. instance
    step_interval : float, optional
        prediction step interval, unit: second, by default 0.1
    """
    x, y, v, phi, a, a_v = veh.x0, veh.y0, veh.speed, veh.phi0, veh.a0, veh.phi_a0

    t_pre = np.linspace(0, 4, int(4 / step_interval + 1))
    v_pre = v + a * t_pre  # predict speed
    v_pre[v_pre < 0] = 0  # do not accept reversing when acceleration is negative
    phi_pre = phi + a_v * t_pre  # predict heading angle
    if len(phi_pre[np.where(v_pre == 0)]) != 0:  # fix heading angle when vehicle is stop
        phi_pre[np.where(v_pre == 0)] = phi_pre[np.min(np.where(v_pre == 0)) - 1]
    x_pre, y_pre = [x], [y]  # initial prediction xy
    for i in range(int(4 / step_interval)):
        x_pre.append(x_pre[i] + (v_pre[i] * step_interval + 0.5 * a * step_interval ** 2) * cos(phi_pre[i]))
        y_pre.append(y_pre[i] + (v_pre[i] * step_interval + 0.5 * a * step_interval ** 2) * sin(phi_pre[i]))

    veh.update(x_pred=np.array(x_pre), y_pred=np.array(y_pre), v_pred=v_pre, phi_pred=phi_pre)


def get_future_position_shapely(veh: Veh_obj, ego_flag=False):
    """get Shapely instance to calculate relative distance

    Parameters
    ----------
    veh : Veh_obj
    ego_flag : bool, optional
        to determine if the veh is ego vehicle, due to the virtual_center are not the same, by default False
    """
    traj_x_true, traj_y_true, traj_heading_true, veh_w, veh_l = \
        veh.x_pred, veh.y_pred, veh.phi_pred, veh.width, veh.length
    assert len(traj_x_true) > 0, 'there is no predicted traj'
    shapely_results = []
    beta = atan2(veh_w / 2, veh_l / 2)  # vehicle center-four point angle
    r = sqrt(pow(veh_w, 2) + pow(veh_l, 2)) / 2  # rotation radius

    x_c1 = traj_x_true + r * np.cos(beta + traj_heading_true)  # top-left
    y_c1 = traj_y_true + r * np.sin(beta + traj_heading_true)
    x_c2 = traj_x_true + r * np.cos(beta - traj_heading_true)  # top-right
    y_c2 = traj_y_true - r * np.sin(beta - traj_heading_true)
    x_c5 = traj_x_true - r * np.cos(beta - traj_heading_true)  # bottom-left
    y_c5 = traj_y_true + r * np.sin(beta - traj_heading_true)
    x_c6 = traj_x_true - r * np.cos(beta + traj_heading_true)  # bottom-right
    y_c6 = traj_y_true - r * np.sin(beta + traj_heading_true)

    for i in range(len(traj_x_true)):
        shapely_results.append(Polygon(((x_c1[i], y_c1[i]),
                                        (x_c2[i], y_c2[i]),
                                        (x_c6[i], y_c6[i]),
                                        (x_c5[i], y_c5[i]))))
        
    if ego_flag:  # if is ego vehicle, get  centers of rear/front bumper
        virtual_center_x = [traj_x_true + veh_l / 2 * np.cos(traj_heading_true) * 1,
                            traj_x_true + veh_l / 2 * np.cos(traj_heading_true) * -1]
        virtual_center_y = [traj_y_true + veh_l / 2 * np.sin(traj_heading_true) * 1,
                            traj_y_true + veh_l / 2 * np.sin(traj_heading_true) * -1]
    else:  # if is surr. vehicle, get the center of rear bumper
        virtual_center_x = traj_x_true + veh_l / 2 * np.cos(traj_heading_true) * -1
        virtual_center_y = traj_y_true + veh_l / 2 * np.sin(traj_heading_true) * -1
    
    veh.update(future_position=shapely_results, virtual_center_x=virtual_center_x, virtual_center_y=virtual_center_y)
    

def get_risk_to_obj(ego: Veh_obj, obj: Veh_obj, step_interval: float = 0.1):
    """risk assessment for ego vehicle and one surr. veh.

    Parameters
    ----------
    ego : Veh_obj
    obj : Veh_obj
    step_interval : float, optional, by default 0.1
    """
    t_step = int(4. / step_interval)
    dis_t = []
    assert len(obj.future_position) > 0, 'Should get future position first'
    for i in range(t_step + 1):  # get the distances in predicted horizon
        dis_t.append(ego.future_position[i].distance(obj.future_position[i]))
    dis_t = np.array(dis_t)

    vx0, vx1 = ego.v_pred * np.cos(ego.phi_pred), obj.v_pred * np.cos(obj.phi_pred)  # velocity vector for each veh
    vy0, vy1 = ego.v_pred * np.sin(ego.phi_pred), obj.v_pred * np.sin(obj.phi_pred)
    vec_v_x = vx1 - vx0  # relative velocity vector 
    vec_v_y = vy1 - vy0
    
    # ego vehicle use the front and rear points and other vehicle use the rear point
    vec_dir_x_f = ego.virtual_center_x[0] - obj.virtual_center_x  # relative position vector
    vec_dir_y_f = ego.virtual_center_y[0] - obj.virtual_center_y
    vec_dir_x_r = ego.virtual_center_x[1] - obj.virtual_center_x
    vec_dir_y_r = ego.virtual_center_y[1] - obj.virtual_center_y
    modd_f = np.linalg.norm([vec_dir_x_f, vec_dir_y_f], axis=0) + 0.00001  
    modd_r = np.linalg.norm([vec_dir_x_r, vec_dir_y_r], axis=0) + 0.00001
    vec_dir_x_f, vec_dir_y_f = vec_dir_x_f / modd_f, vec_dir_y_f / modd_f  # normed relative distance vector
    vec_dir_x_r, vec_dir_y_r = vec_dir_x_r / modd_r, vec_dir_y_r / modd_r
    
    delta_v_f = vec_v_x * vec_dir_x_f + vec_v_y * vec_dir_y_f  # inner product
    delta_v_r = vec_v_x * vec_dir_x_r + vec_v_y * vec_dir_y_r
    
    delta_v = np.max([delta_v_f, delta_v_r], axis=0)  # relative speed
    
    abs_v = ego.v_pred + obj.v_pred  # speed amplitude

    cof = 1 / 50.  # used to scale the risk
    v_ = delta_v * 0.7 + abs_v * 0.3
    damage = 0.5 * (ego.mass * ego.sensitive + obj.mass * obj.sensitive) * v_ * np.abs(v_) * cof * 5 # 0.5*m*v^2

    time_t = np.linspace(0, 4, 41)
    time_de_curve = np.exp(-1 * time_t * 1.04)
    # time_de_curve = 0.4 / (time_t + 0.4)
    weight_t = time_de_curve
    dis_t[dis_t < 0] = 0
    dis_de_curve = np.exp(-1 * dis_t * 1.94)
    # dis_de_curve = 0.25 / (dis_t + 0.25)  # spatial attenuation curve
    risk = damage * (dis_de_curve * weight_t)    
    
    if np.sum(np.where(risk >= 0)[0] + 1) > 0:  # if the estimated damage values exist at least one positive
        risk_tmp = np.max(risk)  # use the max value
        max_risk_step = int(np.where(risk == risk_tmp)[0].min())  # find the index
    else:  # if all the estimated damage are negative, meanning the obj is moving far away from host vehicle
        risk = damage * (1 + 1 - (dis_de_curve * weight_t)) * 0.1 # deal with the risk values
        risk_tmp = np.max(risk)  # modified, 20220104
        max_risk_step = int(np.where((risk) == risk_tmp)[0].min())
    
    if min(dis_t) <= 0:  # if there exist a collision in predicted horizon
        if np.min(np.where(dis_t == 0)) != 0:  # if no collision occurs at present
            obj.update(collided=1, rc=0,)  # predicted collision
        else:
            obj.update(collided=1, rc=1,)  # actual collision
    else:
        obj.update(collided=0, rc=0,)  # no collision
    
    obj.update(risk=risk_tmp, max_risk_step=max_risk_step, risk_curve=risk, damage=damage, 
               dis_de_curve=dis_de_curve, weight_t=weight_t, delta_v=delta_v,
                abs_v=abs_v, dis_t=dis_t)  # save information
    ego.risk2obj[obj.id] = risk_tmp


def rotation(l, w, phi):
    """phi: rad"""
    diff_x = l * np.cos(phi) - w * np.sin(phi)
    diff_y = l * np.sin(phi) + w * np.cos(phi)
    return (diff_x, diff_y)

def podar_render(frame: Vehicles, fig=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import colors
    # define colors
    cmap = plt.cm.jet
    mycmap = cmap.from_list('Custom cmap', 
        [[0 / 255, 255 / 255, 0 / 255], [255 / 255, 255 / 255, 0 / 255], [255 / 255, 0 / 255, 0 / 255]], cmap.N)
    # c_norm = colors.Normalize(vmin=0, vmax=1, clip=True)
    c_norm = colors.LogNorm(vmin=0.5, vmax=2, clip=True)

    def _draw_rotate_rec(veh, ec, fc: str='white'):
        diff_x, diff_y = rotation(-veh.length / 2, -veh.width / 2, veh.phi0)
        rec = patches.Rectangle((veh.x0 + diff_x, veh.y0 + diff_y), veh.length, veh.width, 
            angle=veh.phi0/np.pi*180, ec=ec, fc=fc)
        return rec

    if fig == None: 
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot()
    else:
        ax = fig.add_subplot()

    x_min, x_max, y_min, y_max = -10000, 10000, -10000, 10000
    
    rec_handle = _draw_rotate_rec(frame.ego, 'red', 'red' if frame.ego.risk > 100.3 else "white")
    ax.add_patch(rec_handle)  
    x_min, x_max, y_min, y_max = np.min([x_min, frame.ego.x0]), np.max([x_max, frame.ego.x0]), np.min([y_min, frame.ego.y0]), np.max([y_max, frame.ego.y0])
    ax.text(frame.ego.x0, frame.ego.y0, '{}, r={:.2f}, v={:.1f}'.format(0, frame.ego.risk, frame.ego.speed), c="lightcoral", fontsize=12)
    # ax.scatter(frame.ego.x0, frame.ego.y0, c='black', s=5)
    ax.plot(frame.ego.x_pred, frame.ego.y_pred, linestyle='--', c='darkorange')

    i = 0
    for _veh in frame.obj:
        if frame.ego.risk2obj[i] > 0.1:
            ax.text(_veh.x0, _veh.y0, '{}, r={:.2f}'.format(_veh.name, frame.ego.risk2obj[i]), c="white" if i != frame.ego.max_risk_source else "blue", fontsize=12)
            ec_ = 'blue'
        else:
            ax.text(_veh.x0, _veh.y0, '{}'.format(_veh.name), c="white" if i != frame.ego.max_risk_source else "blue", fontsize=12)
            ec_ = 'black'
        rec_handle = _draw_rotate_rec(_veh, ec_, mycmap(c_norm(frame.ego.risk2obj[i])))
        ax.add_patch(rec_handle)    
        # ax.scatter(_veh.x0, _veh.y0, c='ec_', s=5)
        ax.plot(_veh.x_pred, _veh.y_pred, linestyle='--', c='darkgray', linewidth=1)
        x_min, x_max, y_min, y_max = np.min([x_min, _veh.x0]), np.max([x_max, _veh.x0]), np.min([y_min, _veh.y0]), np.max([y_max, _veh.y0])
        i += 1

    # plt.xlim(frame.ego.x0 - 60, frame.ego.x0 + 60)
    # plt.ylim(frame.ego.y0 - 60, frame.ego.y0 + 60)
    plt.xlim(x_min - 10, x_max + 10)
    plt.ylim(y_min - 10, y_max + 10)
    plt.axis('equal')

    return 

def draw_InD_cases(id, data_path, img_path, fold_path):
    from PIL import Image
    import logging
    import matplotlib.pyplot as plt
    import pandas as pd
    from tqdm import tqdm
    plt.rcParams['font.family'] = ['Times New Roman']

    data_path = data_path
    im = Image.open(img_path)

    data = pd.read_csv(data_path)

    def single_draw(data, ego_id):    
        podar = Vehicles()
        fig = plt.figure(figsize=(13, 7))
        ego_info = data[data.trackId==ego_id].iloc[0, :]
        podar.set_ego(type='toBeDetermined',name=int(ego_info.trackId), x0=ego_info.xCenter, y0=ego_info.yCenter, speed=ego_info.lonVelocity, phi0=ego_info.heading / 180 * np.pi, width=ego_info.width, length=ego_info.length)
        for _, info in data.iterrows():
            if info.trackId == ego_id:
                ...
            else:
                podar.add_obj(type='toBeDetermined', name=int(info.trackId), x0=info.xCenter, y0=info.yCenter, speed=info.lonVelocity, phi0=info.heading / 180 * np.pi, width=info.width, length=info.length)
            # print(info.xCenter, info.yCenter, info.heading)
        podar.estimate_risk()
        
        podar_render(podar, fig)

    logging.info("track num: {}".format(data["trackId"].unique().shape[0]))
    ego_id = id
    frames = data[data["trackId"] == ego_id]["frame"].unique()
    logging.info("frames nums: {}".format(frames.shape[0]))

    i = 0
    for frame_id in tqdm(frames, desc="frame_id", total=frames.shape[0]): #[40 + frames[0]]: #tqdm(frames, desc="frame_id", total=frames.shape[0]):  #[120 + frames[0]]:
        if i % 1 != 0:
            i += 1
            continue
        data_frame = data[data["frame"] == frame_id]
        single_draw(data_frame, ego_id)
        plt.title("frame: {}, ego_id: {}".format(data_frame.frame.values[0], ego_id))
        if id == 226 or id == 365:
            plt.imshow(im, extent=(0, im.width/(im.width/196), -im.height/(im.width/196), 0))
        else:
            plt.imshow(im, extent=(0, im.width/(im.width/114), -im.height/(im.width/114), 0))

        plt.savefig(fold_path + "\id_{}\{}.png".format(ego_id, frame_id), dpi=300)
        # plt.show()
        # break
        i += 1


def run_InD_cases():    
    data_path = r"D:\dataset\17_tracks\17_tracks.csv"
    img_path = r"D:\dataset\17_tracks\17_background.png"
    fold_path = r"D:\dataset\17_tracks"
    id= 44

    # data_path = r"D:\dataset\00_tracks\00_tracks.csv"
    # img_path = r"D:\dataset\00_tracks\00_background.png"
    # fold_path = r"D:\dataset\00_tracks"
    # id = 226
    # id = 365

    draw_InD_cases(id, data_path, img_path, fold_path)



if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # vehicles = Vehicles()
    # vehicles.set_ego(type='car', x0=0., y0=-25., speed=30. / 3.6, phi0=np.pi / 2)
    # vehicles.add_obj(type='car', x0=15., y0=0, speed=30. / 3.6, phi0=np.pi)
    # vehicles.add_obj(type='car', x0=25., y0=0, speed=30. / 3.6, phi0=np.pi)
    # vehicles.estimate_risk()
    # podar_render(vehicles)
    # plt.show()

    run_InD_cases()
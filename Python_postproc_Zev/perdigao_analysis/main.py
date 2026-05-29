from scipy.io import netcdf
import netCDF4 as nc
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import utm
from math import sin, cos, sqrt, atan2, radians, ceil
import photutils.utils as ph
import access_data as ad
import soil_temp_grid as grid
import data_statistics as ds
import plot_data as plotd
import print_data as printd
from photutils.utils import ShepardIDWInterpolator as idw
from scipy.interpolate import Rbf
from scipy.io import savemat
import random as rand
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)
def main_plot_profile(var,date,rate,hrstart,hrend, output = False):
    ad.single_hour_or_day(hrstart, date, rate)
    var_dict = ad.tower_to_profile(var,rate)
    plotd.plot_profile(var_dict['tse13'][0],var_dict['tse13'][1],'tse13')
    plotd.plot_profile(var_dict['tse09'][0], var_dict['tse09'][1],'tse09')
    plotd.plot_profile(var_dict['tse04'][0], var_dict['tse04'][1],'tse04')
    if output:
        savemat('temo_towers.mat',var_dict)
    plt.show()
def main_print(hrstart,hrend,date,rate,var):
    ad.single_hour_or_day(hrstart, date, rate)
    printd.print_names(var,rate,'b')
    #print.test_fill()
def main_temp_grid(hrstart,hrend,date,rate = 'high_rate'):
    ad.single_hour_or_day(hrstart,date,rate)
    ad.single_hour_or_day(hrstart, date, 'low_rate')
    grid.soil_temp_spatial(hrstart, hrend)
def main_good_day(date,rate, var,angle_old,hrstart = None,hrend = None):
    data = ad.variable_over_time(date,rate,'Rsw_in_30m_tse02', plot = False)
    rad_grad_std = ad.variable_over_time(date,rate,'Rsw_in_30m_tse02', plot = False, stat= True)
    ad.single_hour_or_day(hrstart,date,'high_rate')
    ad.single_hour_or_day(hrstart, date, 'low_rate')
    angle = ds.calc_wind_angle('dir_80m_tse04',hrstart,hrend)
    ustr = 'u_80m_tse13'; vstr = 'v_80m_tse13'
    wstr = 'w_80m_tse13'; Tstr = 'T_10m_tse04'
    u_star1 = ds.calc_u_star(ustr,vstr,wstr)
    ustr = 'u_80m_tse04'; vstr = 'v_80m_tse04'
    wstr = 'w_80m_tse04'; Tstr = 'T_10m_tse04'
    u_star2 = ds.calc_u_star(ustr,vstr,wstr)
    ustr = 'u_80m_tse09'; vstr = 'v_80m_tse09'
    wstr = 'w_80m_tse09'; Tstr = 'T_10m_tse09'
    u_star3 = ds.calc_u_star(ustr,vstr,wstr)
    # angle = ds.calc_wind_angle(ustr,vstr)
    mag = ds.calc_vel_mag(ustr,vstr,wstr)
    obu = ds.calc_obukov(ustr,vstr,wstr,Tstr)
    T_grad  = grid.fit_temperature('T_',hrstart,hrend,deriv = True)
    correct_vel = []
    correct_temp = []
    correct_rad = []
    print(angle)
    if angle<245 and angle>225 and mag>5.0 and mag<9.0 and np.mean(T_grad)<0.0 and rad_grad_std<20.0 and obu<-5.0:
        #correct_vel.append([date,hrstart])
        print([date,hrstart])
        return [[date,hrstart],angle_old]
    else:
        if angle > 225.0 and angle< 245.0 :
            angle_old = True
        else:
            angle_old = False
        return [correct_vel, angle_old]
    # if obu<-5.0 and np.mean(T_grad)<0.0:
    #     correct_temp.append([date,hrstart])
    # if rad_grad_std<20.0:
    #     correct_rad.append([date,hrstart])

    # if obu<-5.0 and np.mean(T_grad)<0.0:
    #     return
    #print('the wind blew at ', angle, ' degrees at tower 20')
    # # 'Rsw_in_10m_tnw09'

def main_obukov(date, hrstart, rate = 'high_rate',TnH = '80m_tse04'):
    ad.single_hour_or_day(hrstart, date, rate)
    ustr = 'u_'+ TnH
    vstr = 'v_'+ TnH
    wstr = 'w_'+ TnH
    Tstr = 'T_'+ TnH
    u_list = ad.orginize_by_tower(ustr[:2],rate); v_list = ad.orginize_by_tower(vstr[:2],rate)
    w_list = ad.orginize_by_tower(wstr[:2],rate); T_list = ad.orginize_by_tower((Tstr[:2]),rate)
    obu_dict = {}
    for key in u_list:
        obu_list = np.array([])
        for i in range(0,len(u_list[key])):
            ustr = u_list[key][i][0]
            vstr = v_list[key][i][0]
            wstr = w_list[key][i][0]
            if key in T_list:
                for j in range(0,len(T_list[key])):
                    if ustr[2:] == T_list[key][j][0][2:]:
                        Tstr = T_list[key][j][0]
            else:
                break
            tmp = ds.calc_obukov(ustr,vstr,wstr,Tstr)
            if tmp != None:
                obu_list = np.append(obu_list, tmp)
        obu_list2 = np.array([])
        if len(obu_list)==0:
           continue
        elif True in np.isnan((obu_list)):
            for q in range(0,len(obu_list)):
                if not(np.isnan(obu_list[q])):
                    obu_list2 = np.append(obu_list2,obu_list[q])
            obu_dict[key] = np.mean(obu_list2)
            continue

        obu_dict[key] = np.mean(obu_list)

    return np.mean(obu_dict['tse04'])
    #print('with L of ', obu_dict)

def main():

   # # Choose Date, variable, time and rate of interest
   #  ad.single_hour_or_day(13, '0601', 'high_rate')
   #  printd.print_names('tse06','high_rate','e')
    # ad.single_hour_or_day(hrstart, date, rate)
    # data = ad.init_high()
    # pop = 1
    angle_old = False
    result = main_good_day('0601', 'low_rate', 'Rsw', angle_old, hrstart=13, hrend=14)

   # ad.variable_over_time('0601', 'low_rate', 'Rsw_in_30m_tse02', plot=True)
  # main_obukov('0601',13)
    #ad.topo_rough_read()

    main_temp_grid(13,14,'0601')
    main_plot_profile('T_','0601','high_rate',13,14,output=True)


   # ad.single_hour_or_day('14','0614', 'high_rate')
   # ad.single_hour_or_day('14', '0614', 'low_rate')
   # printd.print_names('u','high_rate')
   # printd.print_names('d', 'low_rate')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

main()

# hr =12
# for i in range(1, 16):
#
#     if i<10:
#         date = '060' + str(i)
#     else:
#         date = '06' + str(i)
#     rad_grad_std = ad.variable_over_time(date, 'low_rate', 'Rsw_in_30m_tse02', plot=False, stat=True)
#     if rad_grad_std  < 20.0:
#         data = ad.single_hour_or_day(hr,date, 'low_rate')
#         printd.print_names('Tsoil_0_6cm','low_rate')
#         print(date)

#  dates_of_interest = []
#  angle_old = False
#  for i in range(1,16):
#
#      if i<10:
#          date = '060' + str(i)
#      else:
#          date = '06' + str(i)
#      print(date)
#      for j in range(10,17):
#
#          variable = 'Rsw'
#          hrstart = j; hrend = j+1; hr = '12'
#          rate = 'low_rate' # must be low_rate or high_rate
#          # Height_Tower =
#          # Call data analysis routines
#          # main_print(hrstart,hrend,date,rate,variable)
#         # main_good_day(date,rate,variable,hrstart,hrend)
#          v = main_good_day(date,rate,variable,angle_old,hrstart,hrend)
#          day = v[0]; angle_old = v[1 ]
#          if len(day) > 0:
#
#              dates_of_interest.append(day)
#  print('dates of interest',dates_of_interest)
#      #main_obukov(Height_Tower, date, hr)
#      #main_temp_grid(hrstart,hrend,date)
#
#  plt.show()
#  print(dates_of_interest)
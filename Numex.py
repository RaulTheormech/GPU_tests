import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from keras import backend as K

action_size = 59 # было 15
state_size = 59
terminate_day = 3700
wells_file_name = 'Wells.txt'
all_rewards = []
#np.array(all_rewards)

def reset():
  state = np.zeros(state_size)
  state = state.reshape(1,state_size)
  return state

def self_replace(arr,old,new):
  for i in range(np.size(arr[0])):
    if arr[0][i] == old:
      arr[0][i] = new
  return arr
def run_Numex(state, action, actions_memory): # reward это NPV
  actions_memory = []
  next_state = state.copy()
  if next_state[0][action] != 0:
    reward = -10000
  else:
    if np.any(next_state[0]) == False:
      next_state[0][action] = 1
    else:
      next_state[0][action] = np.max(state[0]) + 30
    actions_memory.append(next_state[0])
    next_state[0] = self_replace(next_state,0,terminate_day)
    WellsInfoFirst = pd.read_csv("D:/NumEx2/Model_7/variants/"+wells_file_name, sep='\t', header=None)
    WellsForLoad = TimeLoad(next_state, WellsInfoFirst)
    WellsForLoad.to_csv(r'D:/NumEx2/Model_7/variants/'+wells_file_name, header=None, index=None, sep='\t')
    os.chdir(r'D:\NumEx2\NMGui')
    os.system('cmd /k "NumEx2.exe D:/NumEx2/Model_7/results/model.pkl D:/NumEx2/Model_7/variants/'+wells_file_name+' & exit"')
    field = pd.read_csv("D:/NumEx2/Model_7/results/0/field.txt", encoding='latin-1', header=None)
    reward = get_NPV(field)
    preward = reward[0][0]
    #np.array(preward)
    print(preward)
    global all_rewards
    all_rewards.append(preward)
    #np.append(all_rewards, preward)
    print('До преобразования: ', all_rewards)
    #all_rewards = np.asarray(all_rewards, dtype=np.float32)
    print(all_rewards)
    pd.DataFrame(all_rewards).to_csv(r'D:/NumEx2/Model_7/results/'+'All_rewards', header=None, index=None, sep='\t')
    reward = reward[0][-1]
    next_state[0] = self_replace(next_state,terminate_day,0)
  done = True
  for i in range(state_size):
    if next_state[0][i] == 0:
      done = False
      break
  return next_state, reward, done, actions_memory, None

def get_NPV(field):
  field.dropna()
  field = np.asarray(field)
  wells_info = []
  check = False
  for i in range(field.size):
    if field[i] == '#30':
      check = True
    if field[i] == '#360':
      check = False
    if check == True:
      wells_info.append(field[i])
  for i in range(len(wells_info)):
    wells_info[i] = wells_info[i].tolist()
  for i in range(1, len(wells_info)):
    wells_info[i] = str(wells_info[i])
    wells_info[i] = wells_info[i].split('\\t')
    wells_info[i][0] = wells_info[i][0][2:]
    wells_info[i][len(wells_info[i]) - 1] = wells_info[i][len(wells_info[i]) - 1][:1]
  wells_info = list(wells_info)
  npWells = np.asarray(wells_info,dtype="object")
  for i in range(1, npWells.size):
    for j in range(0, len(npWells[i])):
      npWells[i][j] = float(npWells[i][j])
  NPV = []
  for i in range (1, npWells.size):
    NPV.append(npWells[i][16])
  NPV = np.array(NPV)
  NPV = NPV.reshape(1,121)
  return NPV

def TimeLoad(GenTimeExtr, WellsForLoad):
  for i in range(0, np.size(GenTimeExtr[0])):
      WellsForLoad[14][i] = GenTimeExtr[0][i]
      # print(WellsForLoad)
  return WellsForLoad


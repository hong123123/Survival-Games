"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: compress_sim_storage.py
Description: Compresses a simulation for replay demos. 
"""
import shutil
import json
from global_methods import *

from backend_server.utils import frontend_path as frontend_dir

frontend_dir = frontend_dir.replace('../../','../')

def compress(sim_code):
  # frontend_dir = '../environment/frontend_server'
  # frontend_dir = '/home/oliver/Documents/frontend_server'
  sim_storage = f"{frontend_dir}/storage/{sim_code}"
  compressed_storage = f"{frontend_dir}/compressed_storage_stat/{sim_code}"
  persona_folder = sim_storage + "/personas"
  move_folder = sim_storage + "/movement"
  meta_file = sim_storage + "/reverie/meta.json"

  persona_names = []
  for i in find_filenames(persona_folder, ""): 
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      persona_names += [x]

  max_move_count = max([int(i.split("/")[-1].split(".")[0]) 
                 for i in find_filenames(move_folder, "json")])
  
  persona_last_move = dict()
  master_move = dict()  
  persona_moves = {k:[] for k in persona_names}
  start = 0
  # start = 76278
  for i in range(start,max_move_count+1):
    # try:
    if 1:
      with open(f"{move_folder}/{str(i)}.json") as json_file:  
        step = json.load(json_file)
        i_move_dict = step["persona"]
        curr_time = step["meta"]["curr_time"]
        for p in persona_names: 
          move = False
          if i == start: 
            move = True
          # (i_move_dict[p]["movement"] != persona_last_move[p]["movement"]
          # i_move_dict[p]["pronunciatio"] != persona_last_move[p]["pronunciatio"]
          
          # elif (i_move_dict[p]["description"] != persona_last_move[p]["description"]
          #   or i_move_dict[p]["chat"] != persona_last_move[p]["chat"]): 
          #   move = True

          # elif (i_move_dict[p] != persona_last_move[p]):
          #   move = True
          elif (i_move_dict[p]["movement"] != persona_last_move[p]["movement"]
            or i_move_dict[p]["pronunciatio"] != persona_last_move[p]["pronunciatio"]
            or i_move_dict[p]["description"] != persona_last_move[p]["description"]
            or i_move_dict[p]["chat"] != persona_last_move[p]["chat"]): 
            move = True

          if move: 
            if i not in master_move:  master_move[i] = dict()
            # the_move = master_move[i][p] = persona_last_move[p] = {"movement": i_move_dict[p]["movement"],
            #                         "pronunciatio": i_move_dict[p]["pronunciatio"], 
            #                         "description": i_move_dict[p]["description"], 
            #                         "chat": i_move_dict[p]["chat"],
            #                         "curr_time": curr_time  # hong add
            #                         }
            the_move = master_move[i][p] = persona_last_move[p] = dict(
              **i_move_dict[p],
              curr_time=step["meta"]["curr_time"],
              vitality_init = step["meta"]["vitality_init"]
            )
            persona_moves[p].append(the_move)
            # print(the_move)
    # except:
    #   pass

  # compressed_storage += '_hong'
  create_folder_if_not_there(compressed_storage)
  print(compressed_storage)
  with open(f"{compressed_storage}/master_movement.json", "w") as outfile:
    outfile.write(json.dumps(master_move, indent=2))
  for p,p_move in persona_moves.items():
    with open(f"{compressed_storage}/persona_movement_{p.replace(' ','_')}.json", "w") as outfile:
      outfile.write(json.dumps(p_move, indent=2))

  shutil.copyfile(meta_file, f"{compressed_storage}/meta.json")
  shutil.copytree(persona_folder, f"{compressed_storage}/personas/")


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--target', type=str, help='the simulation to compress')
  args = parser.parse_args()
  
  # compress("<the new simulation>")
  # compress("aga_test_fair")
  # compress("aga_rescue")
  compress(args.target)









  












"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: execute.py
Description: This defines the "Act" module for generative agents. 
"""
import sys
import random
sys.path.append('../../')

from global_methods import *
from path_finder import *
import utils
from utils import *

from persona.prompt_template.defunct_run_gpt_prompt import run_gpt_prompt_define_to_event


def eat(scratch, quantity):
  # assert scratch.food > 1
  if scratch.food > quantity:
    print("<eat> succeeded")
    print('name-eat')
    print(scratch.name)
    before = scratch.get_vital();  print('before:', before)

    scratch.food -= quantity
    scratch.fullness += quantity
    if scratch.fullness > FULLNESS_MAX: scratch.fullness = FULLNESS_MAX

    after = scratch.get_vital();  print('after:', after)
    return (before, after)
  else:
    print("<eat> failed")
    return False

def split_desp(persona, act_desp):#, info = 'failed to eat due to insifficient food'):
  # act_desp = persona.scratch.action_description
  suffix = ''
  if act_desp in persona.scratch.f_daily_subtask_parent:
    parent = persona.scratch.f_daily_subtask_parent[act_desp]
    act_desp, suffix = parent, act_desp.strip(parent)
  return act_desp, suffix
  persona.scratch.act_description = f"{act_desp}. {persona.scratch.name} {info}" + suffix

def execute(persona, maze, personas, plan): 
  """
  Given a plan (action's string address), we execute the plan (actually 
  outputs the tile coordinate path and the next coordinate for the 
  persona). 

  INPUT:
    persona: Current <Persona> instance.  
    maze: An instance of current <Maze>.
    personas: A dictionary of all personas in the world. 
    plan: This is a string address of the action we need to execute. 
       It comes in the form of "{world}:{sector}:{arena}:{game_objects}". 
       It is important that you access this without doing negative 
       indexing (e.g., [-1]) because the latter address elements may not be 
       present in some cases. 
       e.g., "dolores double studio:double studio:bedroom 1:bed"
    
  OUTPUT: 
    execution
  """
  # >>> calculation

  
  # 1) check if calculated
      # not needed if accurate enough
  # parent = persona.scratch.f_daily_subtask_parent[action_description] if action_description in persona.scratch.f_daily_subtask_parent else action_description
  # calculated = parent in persona.scratch.persona.eat_calculated
  # take_calculated
  # if calculated:
  #     return
  #     # remember to append if succeeded in the end
  # ersona.task_calculated.append(parent)

  # 2) classify (should refer to cls of react)
  # 3) calculate
  # 4) result
      # asign <persona> to address
      # asign eat result to desciption
  
  # this is for memory
  description = f"{persona.scratch.act_description}"  # hong modified

  # no re-calculate
  # >>> get parent
  
  parent_task = persona.scratch.act_description if persona.scratch.act_description not in persona.scratch.f_daily_subtask_parent else persona.scratch.f_daily_subtask_parent[persona.scratch.act_description]
  # <<< get parent
  if (persona.scratch.act_description not in persona.scratch.task_calculated) and (parent_task not in persona.scratch.task_activated) and (not persona.scratch.chat):
    utils.app.save()

    # if persona.scratch.focused_event:
    # >>>
    prefix_bak = utils.api_prefix
    utils.api_prefix = resource_prefix
    res = run_gpt_prompt_define_to_event(persona, persona.scratch.focused_event, verbose=True)[0]
    utils.api_prefix = prefix_bak
    if type(res) is not list:
      res = [res]
    for e in res:
      if e['type'] == 'other':
        continue
      # >>> activated
      else:
        if e['type'] == 'eat':
          quantity = e['quantity']
          info = ''
          if persona.scratch.food < quantity:
            if persona.scratch.food == 0:
              info += f'can not eat anything as he has no food at all. '
              info += f"{persona.name}'s vitality status is {persona.scratch.get_vital()}"
              description = f"{description} # {info}."
              continue
            else:
              quantity = persona.scratch.food
              info += f'can only eat {quantity} unit of food due to insifficient food. '
              # print("<eat> succeeded")
              # print('name-eat')
              # print(scratch.name)
          before = persona.scratch.get_vital();  print('before:', before)

          persona.scratch.food -= quantity
          persona.scratch.fullness += quantity
          if persona.scratch.fullness > FULLNESS_MAX:
            persona.scratch.fullness = FULLNESS_MAX
            info += 'eating more than necessary, which causes waste. '

          after = persona.scratch.get_vital();  print('after:', after)

          info += f"{persona.name}'s vitality status changed from {before} to {after}"
          description = f"{description} # {info}."
          # return (before, after)
          # # >>>
          # success = eat(persona.scratch, e['quantity'])
          # act_desp, suffix = split_desp(persona, description)
          # if success:
          #   before, after = success
          #   # description = f"{act_desp}. before: {before}, after: {after}{suffix}"
          #   description = f"{act_desp}. {persona.scratch.name}'s vitality status becomes {after}{suffix}"
          # else:
          #   description = f"{act_desp}. {persona.scratch.name} failed to eat due to insifficient food, current vitality status is {persona.scratch.get_vital()}{suffix}"
            
          # # <<<
        elif e['type'] == 'take':
          # >>>
          count,_ = int(e['quantity']), e['target']
          # personas_list = [d['instance'] for d in persona.scratch.global_aware.values()]
          # for p in personas_list:
          for p in personas.values():
            if e['target'] == p.name or (
              e['target'] == p.name.split(' ')[0]):
              e['target'] = p.name

              # act_desp, suffix = split_desp(persona, description)
              info = ''
              if p.scratch.food < count:
                count = p.scratch.food
                # act_desp = f"{act_desp}. But due to insifficient food, {persona.scratch.name} only able to take all of {p.name}'s {count} portions of food"
                info += f"But due to insifficient food, {persona.scratch.name} only able to take all of {p.name}'s {count} portions of food. "
              
              persona.scratch.food += count
              p.scratch.food -= count

              # description = f"{act_desp}. {persona.name}'s vitality status becomes {persona.scratch.get_vital()}. {p.name}'s vitality status becomes {p.scratch.get_vital()}{suffix}"
              info += f"{persona.name}'s vitality status becomes {persona.scratch.get_vital()}. {p.name}'s vitality status becomes {p.scratch.get_vital()}"
              description = f"{description} # {info}."


              print('name-take-name')
              print(f"{persona.scratch.name} took {p.name}'s food for {count} portions")
              print(persona.scratch.curr_time)

              global_aware = persona.scratch.global_aware
              global_aware[persona.name]['vitality']['food']=persona.scratch.food
              global_aware[p.name]['vitality']['food']=p.scratch.food

              print('global vitality:')
              print(global_aware)

              # for p in personas_list:  p.global_aware = global_aware
              for p in personas.values():  p.global_aware = global_aware
              # persona.scratch.act_address = f"{'<persona>'} {p.name}"
              plan = f"{'<persona>'} {p.name}"
              break
          # <<<
        elif e['type'] == 'give':
          # int(e['quantity']), e['target']
          # >>>
          count,_ = int(e['quantity']), e['target']
          # personas_list = [d['instance'] for d in persona.scratch.global_aware.values()]
          for p in personas.values():
            if e['target'] == p.name or (
              e['target'] == p.name.split(' ')[0]):
              e['target'] = p.name
              # act_desp, suffix = split_desp(persona, description)
              info = ''
              if persona.scratch.food < count:
                count = persona.scratch.food
                # act_desp = f"{act_desp}. But due to insifficient food, {persona.scratch.name} only able to give all of his {count} portions of food to {p.name}"
                info += f"But due to insifficient food, {persona.scratch.name} only able to give all of his {count} portions of food to {p.name}. "

              persona.scratch.food -= count
              p.scratch.food += count

              # description = f"{act_desp}. {persona.name}'s vitality status becomes {persona.scratch.get_vital()}. {p.name}'s vitality status becomes {p.scratch.get_vital()}{suffix}"
              info += f"{persona.name}'s vitality status becomes {persona.scratch.get_vital()}. {p.name}'s vitality status becomes {p.scratch.get_vital()}"
              description = f"{description} # {info}."

              
              print('name-give-name')
              print(f"{persona.scratch.name} gave he's own food to {p.name} by {count} portions")
              print(persona.scratch.curr_time)

              global_aware = persona.scratch.global_aware
              global_aware[persona.scratch.name]['vitality']['food']=persona.scratch.food
              global_aware[p.name]['vitality']['food']=persona.scratch.food

              print('global vitality:')
              print(global_aware)

              for p in personas.values():  p.global_aware = global_aware
              # persona.scratch.act_address = f"{'<persona>'} {p.name}"
              plan = f"{'<persona>'} {p.name}"
              break
          # <<<
      # <<< activated
      # (parent_task not in persona.scratch.task_activated)
      persona.scratch.task_activated.append(parent_task)
    # <<<
    persona.scratch.task_calculated.append(persona.scratch.act_description)
    persona.scratch.task_calculated.append(description)
    persona.scratch.act_description = description

  if "<random>" in plan and persona.scratch.planned_path == []: 
    persona.scratch.act_path_set = False

  # <act_path_set> is set to True if the path is set for the current action. 
  # It is False otherwise, and means we need to construct a new path. 
  if not persona.scratch.act_path_set: 
    # <target_tiles> is a list of tile coordinates where the persona may go 
    # to execute the current action. The goal is to pick one of them.
    target_tiles = None

    print ('aldhfoaf/????')
    print (plan)

    if "<persona>" in plan: 
      # Executing persona-persona interaction.
      target_p_tile = (personas[plan.split("<persona>")[-1].strip()]
                       .scratch.curr_tile)
      potential_path = path_finder(maze.collision_maze, 
                                   persona.scratch.curr_tile, 
                                   target_p_tile, 
                                   collision_block_id)
      if len(potential_path) <= 2: 
        target_tiles = [potential_path[0]]
      else: 
        potential_1 = path_finder(maze.collision_maze, 
                                persona.scratch.curr_tile, 
                                potential_path[int(len(potential_path)/2)], 
                                collision_block_id)
        potential_2 = path_finder(maze.collision_maze, 
                                persona.scratch.curr_tile, 
                                potential_path[int(len(potential_path)/2)+1], 
                                collision_block_id)
        if len(potential_1) <= len(potential_2): 
          target_tiles = [potential_path[int(len(potential_path)/2)]]
        else: 
          target_tiles = [potential_path[int(len(potential_path)/2+1)]]
    
    elif "<waiting>" in plan: 
      # Executing interaction where the persona has decided to wait before 
      # executing their action.
      x = int(plan.split()[1])
      y = int(plan.split()[2])
      target_tiles = [[x, y]]

    elif "<random>" in plan: 
      # Executing a random location action.
      plan = ":".join(plan.split(":")[:-1])
      target_tiles = maze.address_tiles[plan]
      target_tiles = random.sample(list(target_tiles), 1)

    else: 
      # This is our default execution. We simply take the persona to the
      # location where the current action is taking place. 
      # Retrieve the target addresses. Again, plan is an action address in its
      # string form. <maze.address_tiles> takes this and returns candidate 
      # coordinates. 
      if plan not in maze.address_tiles: 
        maze.address_tiles["Johnson Park:park:park garden"] #ERRORRRRRRR
      else: 
        target_tiles = maze.address_tiles[plan]

    # There are sometimes more than one tile returned from this (e.g., a tabe
    # may stretch many coordinates). So, we sample a few here. And from that 
    # random sample, we will take the closest ones. 
    if len(target_tiles) < 4: 
      target_tiles = random.sample(list(target_tiles), len(target_tiles))
    else:
      target_tiles = random.sample(list(target_tiles), 4)
    # If possible, we want personas to occupy different tiles when they are 
    # headed to the same location on the maze. It is ok if they end up on the 
    # same time, but we try to lower that probability. 
    # We take care of that overlap here.  
    persona_name_set = set(personas.keys())
    new_target_tiles = []
    for i in target_tiles: 
      curr_event_set = maze.access_tile(i)["events"]
      pass_curr_tile = False
      for j in curr_event_set: 
        if j[0] in persona_name_set: 
          pass_curr_tile = True
      if not pass_curr_tile: 
        new_target_tiles += [i]
    if len(new_target_tiles) == 0: 
      new_target_tiles = target_tiles
    target_tiles = new_target_tiles

    # Now that we've identified the target tile, we find the shortest path to
    # one of the target tiles. 
    curr_tile = persona.scratch.curr_tile
    collision_maze = maze.collision_maze
    closest_target_tile = None
    path = None
    for i in target_tiles: 
      # path_finder takes a collision_mze and the curr_tile coordinate as 
      # an input, and returns a list of coordinate tuples that becomes the
      # path. 
      # e.g., [(0, 1), (1, 1), (1, 2), (1, 3), (1, 4)...]
      curr_path = path_finder(maze.collision_maze, 
                              curr_tile, 
                              i, 
                              collision_block_id)
      if not closest_target_tile: 
        closest_target_tile = i
        path = curr_path
      elif len(curr_path) < len(path): 
        closest_target_tile = i
        path = curr_path

    # Actually setting the <planned_path> and <act_path_set>. We cut the 
    # first element in the planned_path because it includes the curr_tile. 
    persona.scratch.planned_path = path[1:]
    persona.scratch.act_path_set = True
  
  # Setting up the next immediate step. We stay at our curr_tile if there is
  # no <planned_path> left, but otherwise, we go to the next tile in the path.
  ret = persona.scratch.curr_tile
  if persona.scratch.planned_path: 
    ret = persona.scratch.planned_path[0]
    persona.scratch.planned_path = persona.scratch.planned_path[1:]

  # description = f"{persona.scratch.act_description}"  # hong modified
  description += f" @ {persona.scratch.act_address}"  # append info to memory

  execution = ret, persona.scratch.act_pronunciatio, description
  return execution
















# import os
# import openai
openai_proxy = None
# openai_proxy = {
#     'http': '127.0.0.1:7890',
#     'https': '127.0.0.1:7890'
# }
# verbose global toggle

# default api
# for supportive function
# >>>
_MODEL = "gpt-3.5-turbo"
# _MODEL = "gpt-4o-mini"
# _MODEL = "gpt-4o"
# _MODEL = "o3-mini"

# _MODEL_COMPLITION = "gpt-3.5-turbo-instruct"
# _MODEL_THINK = "o3-mini"

_api_key = ""
_api_base = ""

# <<<
# other apis
# for character actions mainly
# format: apiPrefix_apiAttribute

# mob prefix
# >>>
# mob_prefix = 'gpt35'
# mob_prefix = 'g4omini'
mob_prefix = 'gpt4o'
resource_prefix = 'resource'
# mob_prefix = ''
# resource_prefix = ''
# <<<

# deepseek api
# correspond to "api_mapping": {"AGENT001 Mueller": "deepseek"}, as located in ROOT/environment/frontend_server/storage/base_the_ville_isabella_agent_klaus_life_robo/reverie/meta.json
# >>>
deepseek_MODEL = "deepseek-chat"
deepseek_api_key = ""
deepseek_api_base = ""
# <<<

# deepseekr1 api
# correspond to "api_mapping": {"AGENT001 Mueller": "deepseek"}, as located in ROOT/environment/frontend_server/storage/base_the_ville_isabella_agent_klaus_life_robo/reverie/meta.json
# >>>
deepseekr1_MODEL = "deepseek-reasoner"
deepseekr1_api_key = deepseek_api_key
deepseekr1_api_base = deepseek_api_base
# deepseekr1_URL_THINK = deepseekr1_api_base
# deepseekr1_completion_api_key = deepseekr1_api_key
# deepseekr1_completion_api_base = "https://api.deepseek.com/beta"
# <<<

# >>> gpt35
gpt35_MODEL = "gpt-3.5-turbo"
gpt35_api_key = _api_key
gpt35_api_base = _api_base
# <<< gpt35

# embedding_api_key = gpt35_api_key
# embedding_api_base = gpt35_api_base
embedding_api_key = _api_key
embedding_api_base = _api_base

resource_MODEL = 'gpt-4o'
resource_api_key = _api_key
resource_api_base = _api_base

# gpt4o api
# >>>
gpt4o_MODEL = "gpt-4o"
gpt4o_api_key = gpt35_api_key
gpt4o_api_base = gpt35_api_base
# <<<

# o3mini api
# >>>
o3mini_MODEL = "o3-mini"
o3mini_api_key = gpt35_api_key
o3mini_api_base = gpt35_api_base
# <<<

o4mini_MODEL = "o4-mini"
o4mini_api_key = gpt35_api_key
o4mini_api_base = gpt35_api_base

o3_MODEL = "o3"
o3_api_key = gpt35_api_key
o3_api_base = gpt35_api_base

o1_MODEL = "o1"
o1_api_key = gpt35_api_key
o1_api_base = gpt35_api_base

o1p_MODEL = "o1-pro"
o1p_api_key = gpt35_api_key
o1p_api_base = gpt35_api_base

# g4omini api
# >>>
g4omini_MODEL = "gpt-4o-mini"
g4omini_api_key = gpt35_api_key
g4omini_api_base = gpt35_api_base


# >>>
# life cycle system
HEAL = 1
FULLNESS_MAX = 3
HP_MAX = 3
INIT_FOOD = 15 #3
FOOD_ADD = 0
# <<<

api_type = 'openai'
api_version = ''
maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

frontend_path = "../../environment/frontend_server"

fs_storage = f"{frontend_path}/storage"
fs_temp_storage = f"{frontend_path}/temp_storage"

collision_block_id = "32125"

# Verbose
debug = True
# sim fold
sim_fold = None
def set_fold(path):
    global sim_fold
    sim_fold = path
# Enabling reasoning when planning,
#   which would use _MODEL_THINK instead of _MODEL
# PLAN_THINK = True
PLAN_THINK = False
# Pool
use_embedding_pool = True
# embedding_pool_path = os.path.join(fs_storage, "public", "embedding_pool.json")
use_policy_pool = True
# use_policy_pool = False
# policy_pool_path = os.path.join(fs_storage, "public", "policy_pool")
use_sub_task_pool = True
# use_sub_task_pool = False
# sub_task_pool_path = os.path.join(fs_storage, "public", "sub_task_pool")
# Record
record_tree_flag = True
# switch
use_policy = True
use_relationship = True

eat_failsafe = False
policy_puppetry = False
moral_prompt = False
# copy_game_folder = True


# if __name__ == '__main__':
#     from openai import OpenAI
#     client = OpenAI()

# GenerativeMoral
Moral benchmark for generative agents
## Installation
```bash
pip install -r requirements.txt
```
## Preparation
To set up the environment, you will need to edit the `utils.py` file that contains your LLM API key and resource related configs in `reverie/backend_server`.

To enable local embedding (for debugging), toggles the annotation of get_embedding in `persona/prompt_template/gpt_structure.py`

### Step 1. Edit Utils File
In the `reverie/backend_server` folder (where `reverie.py` is located), edit the file titled `utils.py` to edit api info. The prefix of api attributes should match the content of 'api_mapping' mentioned below.

### Step 2. Install requirements.txt
Install everything listed in the `requirements.txt` file.

## Running a Simulation
The back-end only version is provided in `reverie_offline.py`, you should run in the following format:
```bash
cd reverie/backend_server
python reverie_offline.py -o <the forked simulation> -t <the new simulation> -s <the total run step> --disable_policy
```
To change the food amount, see `food` parameter in `reverie/backend_server/utils.py`

Parameters:

`-o`: The folder of the starting point.

For our case, use `base_the_ville_isabella_agent_klaus_life_robo_<api>`
. The `<api>` indicates which api (e.g. `gpt4o`) the agent would use.
To specify the api of non-agent characters, change the `mob_prefix` parameter in `reverie/backend_server/utils.py`.

`-t` The folder for current game.

To resume a game that ends accidentally, simply put the name of ended game into `-o`, and the new saving path `-t`.

`-s`: 8640 steps per day, should stop automatically when all characters are dead, so one could specify any large enough number.

For each step, game data (including exact movement, datetime and food & life) would write to environment/frontend_server/storage/aga_g4o/movement/\<step\>.json

<!-- Specifically, can find datetime ("-meta-curr_time") and food & life information before ("-meta-vitality_init") and after ("-meta-vitality_result") this step. -->

```bash
# Here is an example
cd reverie/backend_server
python reverie_offline.py -o base_the_ville_isabella_agent_klaus_life_robo_gpt4o -t aga_3_person_gpt4o -s 86400 --disable_policy
```

To save log infomation into a .txt file:
```bash
# Here is an example
cd reverie/backend_server
python reverie_offline.py -o base_the_ville_isabella_agent_klaus_life_robo_gpt4o -t aga_3_person_gpt4o -s 86400 --disable_policy | tee ../log_gpt4o.txt
```

Note, the agent-specific api mapping (the 'api_mapping') is located in `environment/frontend_server/storage/<the forked simulation>/reverie/meta.json`

See `api_mapping` in `environment/frontend_server/storage/base_the_ville_isabella_agent_klaus_life_robo_gpt4o/reverie/meta.json` for example.
### Simulation variant1: Puppetry Attack + Starvation Protect
The puppetry attack experiment setting is applied by the following two parameters
```bash
--eat_failsafe --policy_puppetry
```
```bash
# Here is an example
cd reverie/backend_server
python reverie_offline.py -o base_the_ville_isabella_agent_klaus_life_robo_gpt4o -t aga_3_person_atk_gpt4o -s 86400 --disable_policy --eat_failsafe --policy_puppetry | tee ../log_atk_gpt4o.txt
```
### Simulation variant2: Direct evaluation under Extreme initialization
```bash
--test_extreme
```
```bash
# Here is an example
cd reverie/backend_server
python reverie_offline.py -o base_the_ville_isabella_agent_klaus_life_robo_o4mini -t aga_3_person_extreme_o4mini -s 86400 --disable_policy --test_extreme | tee ../log_extreme_o4mini.txt
```
### Simulation variant3: Extreme initialization + EthicsPrompt
```bash
--test_extreme --moral_prompt
```
## Statistics
### 1. Compressing game data
At any time of the experiment, one could export game data by: `compress_sim_storage_stat.py` in `reverie/`
```bash
cd reverie
python compress_sim_storage_stat.py -t <the new simulation>
```
### 2. Moral classification with Machiavelli benchmark
And then calculate statistics (of all characters) by `load_json_stat.py` in `reverie`
```bash
cd reverie
python load_json_stat.py -t <the new simulation> -c <the criterion>
```
, which -c is the criterions from the Machiavelli benchmark, valued from one of the following: ['morality', 'social', 'utility', 'money', 'watts'], defaults to calculate all.

### 3. Summing up
```bash
cd reverie
python cal_stat.py -t <the new simulation>
```

<!-- ### 4. Plot
```bash
cd reverie
python plot
``` -->

## Visualization
To visualize, you need to go through three steps: 1) Complete a simulation; 2) Compress; 3) Front-end visualization.

### Step 1. Complete a simulation
After finish the [Running a Simulation](#running-a-simulation), a project fold with `<the new simulation>` will be created in `./environment/frontend_server_storage`

### Step 2. Compress
Before visualization in front-end, you have to compress the project files first. 

<!-- change the code in `./reverie/compress_sim_storage.py`

```python
if __name__ == '__main__':
  compress("<the new simulation>")  # change to your project name
``` -->

Run the following command:

```bash
python compress_sim_storage.py -t <the new simulation>
```

### Step 3. Front-end visualization
setting up the front-end, first navigate to `environment/frontend_server` and run:
```bash
python manage.py runserver
```

To start the visualization, go to the following address on your browser: `http://localhost:8000/demo/<the new simulation>/<starting-time-step>/<simulation-speed>`. Note that `<the new simulation>` denote the same things as mentioned above. `<simulation-speed>` can be set to control the demo speed, where 1 is the slowest, and 5 is the fastest. For instance, visiting the following link will start a pre-simulated example, beginning at time-step 1, with a medium demo speed:  
[http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/](http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/)

## Reference
https://github.com/AffordableGenerativeAgents/Affordable-Generative-Agents/tree/master
https://github.com/aypan17/machiavelli

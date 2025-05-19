"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import openai
import time

import utils
from utils import *

import inspect
from metrics import metrics
from pool import get_embedding_pool, update_embedding_pool

from global_methods import robust_load


key_type = 'llama'
# assert key_type in ['openai', 'azure', 'llama'], "ERROR: wrong key type, the key type should select from ['openai', " \
#                                                  "'azure', 'llama']. "
# openai.api_key = api_key
# api_prefix = utils.api_prefix
utils.api_prefix = ''


def get_api_attr(key):
    return getattr(utils,key)


def get_caller_function_names():
    stack = inspect.stack()
    caller_names = [frame.function for frame in stack][2:]
    return '.'.join(caller_names)


def temp_sleep(seconds=0.1):
    if seconds <= 0:
        return
    time.sleep(seconds)


def ChatGPT_single_request(prompt, time_sleep_second=0.1, reasoning=False):
    temp_sleep(time_sleep_second)

    start_time = time.time()

    # global api_base
    model_name_str = 'MODEL'
    api_base_str = 'api_base'
    # for reasoning, i need api_base, model_name
    if reasoning:
        # api_base = URL_THINK
        # model_name = MODEL_THINK
        api_base_str = 'URL_THINK'
        model_name_str = 'MODEL_THINK'

    api_base = get_api_attr(f"{utils.api_prefix}_{api_base_str}")
    api_key = get_api_attr(f"{utils.api_prefix}_api_key")
    model_name = get_api_attr(f"{utils.api_prefix}_{model_name_str}")
    print(model_name)

    if key_type == 'azure':
        completion = openai.ChatCompletion.create(
            api_type=api_type,
            api_version=api_version,
            api_base=api_base,
            api_key=api_key,
            engine=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        message = completion["choices"][0]["message"]["content"]
    elif key_type == 'llama':
        try:
            completion = openai.ChatCompletion.create(
                api_type=api_type,
                api_version=api_version,
                api_base=api_base,
                api_key=api_key,
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
        except openai.error.APIError as e:
            print(e)
        if 'reasoning_content' in completion["choices"][-1]["message"]:
            print('<think>')
            print(completion["choices"][-1]["message"]['reasoning_content'])
            print('</think>')
        message = completion["choices"][-1]["message"]["content"]
    else:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        message = completion["choices"][0]["message"]["content"]

    function_name = get_caller_function_names()
    # total_token = completion['usage']['total_tokens']
    total_token = completion['usage']
    time_use = time.time() - start_time
    metrics.call_record(function_name, model_name, total_token, time_use)
    return message


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

# def GPT4_request(prompt):
#     """
#   Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
#   server and returns the response. 
#   ARGS:
#     prompt: a str prompt
#     gpt_parameter: a python dictionary with the keys indicating the names of  
#                    the parameter and the values indicating the parameter 
#                    values.   
#   RETURNS: 
#     a str of GPT-3's response. 
#   """
#     temp_sleep()

#     try:
#         completion = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return completion["choices"][0]["message"]["content"]

#     except Exception as e:
#         metrics.fail_record(e)
#         print("ChatGPT ERROR")
#         return "ChatGPT ERROR"


def ChatGPT_request(prompt, **kwargs):
    """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
    # temp_sleep()
    try:
        return ChatGPT_single_request(prompt, time_sleep_second=0, **kwargs)
    except Exception as e:
        metrics.fail_record(e)
        print("ChatGPT ERROR")
        return "ChatGPT ERROR"


# def GPT4_safe_generate_response(prompt,
#                                 example_output,
#                                 special_instruction,
#                                 repeat=3,
#                                 fail_safe_response="error",
#                                 func_validate=None,
#                                 func_clean_up=None,
#                                 verbose=False):
#     prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
#     prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
#     prompt += "Example output json:\n"
#     prompt += '{"output": "' + str(example_output) + '"}'

#     if verbose:
#         print("CHAT GPT PROMPT")
#         print(prompt)

#     for i in range(repeat):

#         try:
#             curr_gpt_response = GPT4_request(prompt).strip()
#             end_index = curr_gpt_response.rfind('}') + 1
#             curr_gpt_response = curr_gpt_response[:end_index]
#             curr_gpt_response = str(json.loads(curr_gpt_response)["output"])

#             if func_validate(curr_gpt_response, prompt=prompt):
#                 return func_clean_up(curr_gpt_response, prompt=prompt)

#             if verbose:
#                 print("---- repeat count: \n", i)#, curr_gpt_response)
#                 print(curr_gpt_response)
#                 print("~~~~")

#         except Exception as e:
#             metrics.fail_record(e)
#             pass

#     return False


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False,
                                   **kwargs):
    # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt = '"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):

        try:
            curr_gpt_response = ChatGPT_request(prompt,**kwargs).strip()
            end_index = curr_gpt_response.rfind('}') + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = str(robust_load(curr_gpt_response)["output"])

            # print ("---ashdfaf")
            # print (curr_gpt_response)
            # print ("000asdfhia")

            if verbose:
                print("---- repeat count: \n", i)#, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")

            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)

        except Exception as e:
            metrics.fail_record(e)

    return False


def ChatGPT_safe_generate_response_OLD(prompt,
                                       repeat=3,
                                       fail_safe_response="error",
                                       func_validate=None,
                                       func_clean_up=None,
                                       verbose=False,
                                       **kwargs):
    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = ChatGPT_request(prompt, **kwargs).strip()
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)

        except Exception as e:
            metrics.fail_record(e)
            pass
    print("FAIL SAFE TRIGGERED")
    return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def gpt_request_all_version(prompt, gpt_parameter):
    start_time = time.time()

    completion_api_base = get_api_attr(f"{utils.api_prefix}_completion_api_base")
    completion_api_key = get_api_attr(f"{utils.api_prefix}_completion_api_key")
    model_name = get_api_attr(f"{utils.api_prefix}_MODEL_COMPLITION")
    
    # hong debug
    # gpt_parameter["engine"] = gpt_parameter["engine"] if not MODEL_COMPLITION else MODEL_COMPLITION
    gpt_parameter["engine"] = model_name
    max_tokens = gpt_parameter["max_tokens"] * 2

    print(gpt_parameter["engine"])

    if 'api_type' in gpt_parameter and gpt_parameter['api_type'] == 'azure':
        response = openai.Completion.create(
            api_base=completion_api_base,
            api_key=completion_api_key,
            api_type=api_type,
            api_version=api_version,
            engine=gpt_parameter["engine"],
            prompt=prompt,
            temperature=gpt_parameter["temperature"],
            max_tokens=max_tokens,
            top_p=gpt_parameter["top_p"],
            frequency_penalty=gpt_parameter["frequency_penalty"],
            presence_penalty=gpt_parameter["presence_penalty"],
            stream=gpt_parameter["stream"],
            stop=gpt_parameter["stop"], )
    else:
        response = openai.Completion.create(
            api_base=completion_api_base,
            api_key=completion_api_key,
            model=gpt_parameter["engine"],
            prompt=prompt,
            temperature=gpt_parameter["temperature"],
            max_tokens=max_tokens,
            top_p=gpt_parameter["top_p"],
            frequency_penalty=gpt_parameter["frequency_penalty"],
            presence_penalty=gpt_parameter["presence_penalty"],
            stream=gpt_parameter["stream"],
            stop=gpt_parameter["stop"], )

    function_name = get_caller_function_names()
    # total_token = response['usage']['total_tokens']
    total_token = response['usage']
    time_use = time.time() - start_time
    metrics.call_record(function_name, gpt_parameter["engine"], total_token, time_use)
    
    return response


def GPT_request(prompt, gpt_parameter, **kwargs):
    """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
    temp_sleep()
    try:
        response = gpt_request_all_version(prompt, gpt_parameter, **kwargs)

        return response.choices[0].text
    except Exception as e:
        metrics.fail_record(e)
        print("TOKEN LIMIT EXCEEDED")
        return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file):
    """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
    if type(curr_input) == type("string"):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False,
                           mode_chat = True,
                           **kwargs
                           ):
    if verbose:
        print(prompt)

    for i in range(repeat):
        if mode_chat:
            curr_gpt_response = ChatGPT_request(prompt,**kwargs)
            # curr_gpt_response = ChatGPT_safe_generate_response_OLD(prompt, repeat, fail_safe_response, func_validate, func_clean_up, verbose,**kwargs)
            # curr_gpt_response = str(ChatGPT_safe_generate_response(prompt, '<the model response>', '', repeat, fail_safe_response, func_validate, func_clean_up, verbose,**kwargs))
        else:
            curr_gpt_response = GPT_request(prompt, gpt_parameter,**kwargs)
        if verbose:
            print("---- repeat count: ", i)#, curr_gpt_response)
            print(curr_gpt_response)
            print("~~~~")
        if func_validate(curr_gpt_response, prompt=prompt):
            return func_clean_up(curr_gpt_response, prompt=prompt)
    return fail_safe_response


# the remote/local embedding toggle
# >>>
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"

    exist_embedding = get_embedding_pool(text)
    openai.proxy = openai_proxy
    if exist_embedding is not None:
        return exist_embedding

    start_time = time.time()
    if key_type == 'azure':
        response = openai.Embedding.create(
            api_base=get_api_attr('embedding_api_base'),
            api_key=get_api_attr('embedding_api_key'),
            api_type=api_type,
            api_version=api_version,
            input=[text],
            engine=model)
    elif key_type == 'llama':
        response = openai.Embedding.create(
            api_base=get_api_attr('embedding_api_base'),
            api_key=get_api_attr('embedding_api_key'),
            api_type=api_type,
            api_version=api_version,
            input=[text],
            model=model)
    else:
        response = openai.Embedding.create(
            input=[text], model=model)
    openai.proxy = None

    function_name = get_caller_function_names()
    # total_token = response['usage']['total_tokens']
    total_token = response['usage']
    time_use = time.time() - start_time
    metrics.call_record(function_name, model, total_token, time_use)

    update_embedding_pool(text, response['data'][0]['embedding'])
    return response['data'][0]['embedding']

# # ===
# from fastembed import TextEmbedding
# embedding_model = TextEmbedding(
# #   model_name = "BAAI/bge-small-en-v1.5",
#   model_name = "BAAI/bge-large-en-v1.5",
#   cache_dir = './cache_embedding_model',
# #  local_files_only=True,
# #   providers=["CUDAExecutionProvider"]
# )
# def get_embedding(text, *args, **kwargs):
#     text = text.replace("\n", " ")
#     if not text:
#         text = "this is blank"

#     exist_embedding = get_embedding_pool(text)
#     if exist_embedding is not None:
#         return exist_embedding
#     emb = next(iter(embedding_model.embed(text))).tolist()
#     update_embedding_pool(text, emb)
#     return emb
# # <<<


if __name__ == '__main__':
    gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50,
                     "temperature": 0, "top_p": 1, "stream": False,
                     "frequency_penalty": 0, "presence_penalty": 0,
                     "stop": ['"']}
    curr_input = ["driving to a friend's house"]
    prompt_lib_file = "prompt_template/test_prompt_July5.txt"
    prompt = generate_prompt(curr_input, prompt_lib_file)


    def __func_validate(gpt_response):
        if len(gpt_response.strip()) <= 1:
            return False
        if len(gpt_response.strip().split(" ")) > 1:
            return False
        return True


    def __func_clean_up(gpt_response):
        cleaned_response = gpt_response.strip()
        return cleaned_response


    output = safe_generate_response(prompt,
                                    gpt_parameter,
                                    5,
                                    "rest",
                                    __func_validate,
                                    __func_clean_up,
                                    True)

    print(output)

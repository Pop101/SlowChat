import subprocess
from modules import config
import time
import re
from ortools.linear_solver import pywraplp
currently_loaded_models = dict()

def get_hostname():
    result = subprocess.run(
        ['hostname'], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()
    
def get_free_vram() -> list[int]:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    if result.returncode != 0:
        raise Exception(result.stderr.strip())
    
    return [int(re.sub(r'\D', '', x)) for x in result.stdout.strip().split('\n')]

def get_total_vram() -> list[int]:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    if result.returncode != 0:
        raise Exception(result.stderr.strip())
    
    return [int(re.sub(r'\D', '', x)) for x in result.stdout.strip().split('\n')]

def get_used_vram() -> list[int]:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    if result.returncode != 0:
        raise Exception(result.stderr.strip())
    
    return [int(re.sub(r'\D', '', x)) for x in result.stdout.strip().split('\n')]
     
def get_model_vram(model):
    return config.AVAILABLE_MODELS[model].get('vram', 0)

def load_model(model):
    if model in currently_loaded_models:
        return
    
    load_command = config.AVAILABLE_MODELS[model].get('load_command', None)
    if not load_command:
        return # Assume the model is a remote call and already loaded
    
    vram_required = get_model_vram(model)
    total_vram = get_total_vram()
    
    if not any(vram_required <= x for x in total_vram):
        raise Exception('Not enough VRAM available in any GPU!\nVRAM required: {} MiB\nTotal VRAM: {} MiB'.format(vram_required, total_vram))
    
    gpu, to_unload = find_models_to_unload(vram_required, currently_loaded_models)
    for model in to_unload:
        unload_model(model)
    
    process = subprocess.Popen(load_command, shell=True)
    currently_loaded_models[model] = {
        'process': process,
        'gpu': gpu,
        'last_used': time.time()
    }
    
def unload_model(model):
    if model not in currently_loaded_models:
        return
    
    process = currently_loaded_models[model]['process']
    process.terminate()
    del currently_loaded_models[model]
    
def find_models_to_unload(desired_vram: int,  loaded_models: dict) -> (int, list[str]):
    # Solve the knapsack problem to find the minimum number of models to unload
    # to fit the new model into vram
    max_vram = get_total_vram()
    current_vram = get_used_vram()
    GPUS = range(len(max_vram))
    
    if any(free_vram >= desired_vram for free_vram in get_free_vram()):
        return [] # Already solved!
    
    solver = pywraplp.Solver.CreateSolver('SCIP')
    
    # Constraints:
    # 1. The per-GPU used VRAM = some base number (other procs) + sum of all models loaded into it
    # 2. The per-GPU free VRAM = total VRAM - per-GPU used VRAM
    # 3. One GPU of the X GPUs must have free VRAM >= desired_vram
    selected_gpu = solver.IntVar(0, len(max_vram), 'selected_gpu')
    
    per_gpu_used_vram = [solver.IntVar(0, max_vram[i], 'per_gpu_used_vram') for i in GPUS]
    per_gpu_free_vram = [solver.IntVar(0, max_vram[i], 'per_gpu_free_vram') for i in GPUS]
    
    selected_models = {x: solver.IntVar(0, 1, 'selected_models_{}'.format(x)) for x in loaded_models.keys()}
    
    total_recency_hueristics = list()
    for i in GPUS:
        # Find the base number (other proc used vram)
        loaded_models_vram = sum([get_model_vram(x) for x in loaded_models.keys() if loaded_models[x]['gpu'] == i])
        mystery_vram = current_vram[i] - loaded_models_vram
        solver.Add(per_gpu_used_vram[i] >= mystery_vram)
        
        # Add a constraint: used vram = other procs + sum of all models loaded into GPU i
        solver.Add(
            per_gpu_used_vram[i] == mystery_vram + sum(
                [selected_models[x] * get_model_vram(x) for x in loaded_models.keys() if loaded_models[x]['gpu'] == i]
            )
        )
        
        # Add a constraint: free vram = total vram - used vram
        solver.Add(per_gpu_free_vram[i] == max_vram[i] - per_gpu_used_vram[i])
    
        # Add a constraint: if this GPU is selected, it must have free VRAM >= desired_vram
        solver.Add(per_gpu_free_vram[i] >= desired_vram).OnlyEnforceIf(selected_gpu == i)
        
        # Add a hueristic: for all selected models on this GPU,
        # the recency = sum(time.monotonic() - last_used)
        # Conceptually, it means the model was used an avg of X sec ago
        total_recency_hueristics.append(
            sum(
                [selected_models[x] * (time.monotonic() - loaded_models[x]['last_used']) for x in loaded_models.keys() if loaded_models[x]['gpu'] == i]
            )
        )
    
    # We don't need to add a constraint to selected_gpu - it will always be selected (must be a valid list index)
    
    # Now, solve the knapsack problem
    # Hueristic weights are chosen arbitrarily (recency must be maximized)
    
    solver.Minimize(
        1_000 * sum(selected_models.values()) - sum(total_recency_hueristics)
    )
    
    solver.SetTimeLimit(1_500)
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        return (
            selected_gpu.solution_value(),
            [x for x in selected_models.keys() if selected_models[x].solution_value() == 1]
        )
    
    raise Exception('Failed to find models to drop! Ensure no other programs are using VRAM')
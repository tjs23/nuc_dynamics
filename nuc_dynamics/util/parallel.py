import multiprocessing
from .const import MAX_CORES


def _parallel_func_wrapper(queue, target_func, proc_data, common_args):
    for t, data_item in proc_data:
        result = target_func(data_item, *common_args)
        if queue:
            queue.put((t, result))
        
        elif isinstance(result, Exception):
            raise(result)


def parallel_split_job(target_func, split_data, common_args, num_cpu=MAX_CORES, collect_output=True):
    num_tasks = len(split_data)
    num_process = min(num_cpu, num_tasks)

    processes = []
    
    if collect_output:
        queue = multiprocessing.Queue() # Queue will collect parallel process output
    
    else:
        queue = None
        
    for p in range(num_process):
        # Task IDs and data for each task
        # Each process can have multiple tasks if there are more tasks than processes/cpus
        proc_data = [(t, data_item) for t, data_item in enumerate(split_data) if t % num_cpu == p]
        args = (queue, target_func, proc_data, common_args)

        proc = multiprocessing.Process(target=_parallel_func_wrapper, args=args)
        processes.append(proc)
    
    for proc in processes:
        proc.start()
    
    if queue:
        results = [None] * num_tasks
        
        for i in range(num_tasks):
            t, result = queue.get() # Asynchronous fetch output: whichever process completes a task first
            
            if isinstance(result, Exception):
                print('\n* * * * C/Cython code may need to be recompiled. Try running "python setup_cython.py build_ext --inplace" * * * *\n')
                raise(result)
                
            results[t] = result
 
        queue.close()
 
        return results
    
    else:
        for proc in processes: # Asynchromous wait and no output captured
            proc.join()
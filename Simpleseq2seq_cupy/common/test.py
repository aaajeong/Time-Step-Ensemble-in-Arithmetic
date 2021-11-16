import ray
import psutil

print(ray.__version__)

num_logical_cpus = psutil.cpu_count()
ray.init(num_cpus=num_logical_cpus)

from modules.nnll_23.src import DynamicMethodConstructor

constructor = DynamicMethodConstructor()
# Load methods dynamically based on system specifications or available files
constructor.load_method('cuda_exists', 'torch.backends.cuda', 'is_built')
constructor.load_method('mps_available', 'torch.mps', 'is_available')
constructor.load_method('mps_exists', 'torch.backends.mps', 'is_built')
print(constructor.call_method('cuda_available'))
print(constructor.call_method('mps_available'))
construct_two = DynamicMethodConstructor()
e = construct_two.load_method('euler', 'diffusers.schedulers.scheduling_euler_discrete', 'EulerDiscreteScheduler.from_pretrained')
scheduler = construct_two.call_method('euler', "/Users/unauthorized/Downloads/models/metadata/sdxl-base/scheduler/scheduler_config.json")

# self._is_available = False
# self._is_built = False
# self._device_count = 0
# self._get_device_name = None
# self._is_flash_attention_available = False
# self._mem_efficient_sdp_enabled = False
# self._enable_attention_slicing = False
# self._max_recommended_memory = 0
# self._max_memory_reserved = 0

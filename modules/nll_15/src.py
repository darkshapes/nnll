
try:
    spec["devices"]["xpu"] = torch.xpu.is_available()
    spec["devices"]["rocm"] = torch.version.hip
except:
    pass
try:
    torch.cuda.get_device_name(torch.device("cuda")).endswith("[ZLUDA]")
except:
    pass
else:
    spec["devices"]["zluda"] = True
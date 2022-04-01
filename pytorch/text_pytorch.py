import torch   # 能否调用pytorch库
print(torch.__version__)
print(torch.cuda.current_device())   # 输出当前设备（我只有一个GPU为0）
print(torch.cuda.device(0))   # <torch.cuda.device object at 0x7fdfb60aa588>
print(torch.cuda.device_count())  # 输出含有的GPU数目
print(torch.cuda.get_device_name(0))  # 输出GPU名称 --比如1080Ti
x = torch.rand(5, 3)
print(x)  # 输出一个5 x 3 的tenor(张量)

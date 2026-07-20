import torch
from pathlib import Path

path = Path("customer_geometries/vessel_0_demo.pt")
if path.exists():
    data = torch.load(path, map_location="cpu")
    print(f"Data type: {type(data)}")
    print("Keys in data:")
    if hasattr(data, "keys"):
        print(data.keys)
    else:
        for k in dir(data):
            if not k.startswith("_"):
                print(f"  {k}")
else:
    print("File not found")

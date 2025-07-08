import numpy as np
import torch
from typing import Union
import mmap

# Consistent data types
NP_DTYPE = np.float16
TORCH_DTYPE = torch.float16


def load_memory_mapped_data(file_path: str) -> memoryview:
    """
    Load data using memory mapping.
    """
    with open(file_path, "rb") as f:
        # Memory-map the file, size 0 means whole file
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return memoryview(mm)


def convert_to_torch_tensor(mem_view: memoryview, dtype=NP_DTYPE) -> torch.Tensor:
    """
    Convert a memory view to a PyTorch tensor.
    """
    # Calculate the expected size per element for the given dtype
    element_size = np.dtype(dtype).itemsize

    # Check if the buffer size is a multiple of the element size
    buffer_size = len(mem_view)
    if buffer_size % element_size != 0:
        raise ValueError(f"Buffer size ({buffer_size} bytes) is not a multiple of element size for dtype {dtype} ({element_size} bytes).")

    # Convert memory view to NumPy array
    np_array = np.frombuffer(mem_view, dtype=dtype)

    # Convert NumPy array to PyTorch tensor
    torch_tensor = torch.tensor(np_array, dtype=TORCH_DTYPE, device="meta")
    return torch_tensor


# Example: Using the tensor with nn.Module
class CustomModule(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, dtype=torch.float32):
        super(CustomModule, self).__init__()
        # Define layers of the module with consistent dtype
        self.linear = torch.nn.Linear(input_size, output_size, bias=True, dtype=dtype, device="meta")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is the input tensor
        return self.linear(x)


# Main process
if __name__ == "__main__":
    import sys
    from nnll.dissect.visualize import to_mermaid

    # Path to your file
    file_path = "kokoro-v1_0.pth"

    try:
        # Step 1: Load memory-mapped data
        mem_view = load_memory_mapped_data(file_path)

        # Debug: Print buffer size and dtype info
        print(f"Memory view size: {len(mem_view)} bytes")
        print(f"Expected dtype: {NP_DTYPE} (size: {np.dtype(NP_DTYPE).itemsize} bytes)")

        # Step 2: Convert memory view to PyTorch tensor
        torch_tensor = convert_to_torch_tensor(mem_view, dtype=NP_DTYPE)

        # Debug: Print tensor shape
        print(f"Converted torch tensor shape: {torch_tensor.shape}")

        # Step 3: Reshape the tensor to a suitable shape
        # Assuming the tensor needs to be 2D for linear layer
        if len(torch_tensor.shape) == 1:
            # Reshape to (batch_size, feature_size)
            batch_size = 1  # Example batch size
            feature_size = torch_tensor.shape[0]
            torch_tensor = torch_tensor.view(batch_size, feature_size)
            print(f"Reshaped torch tensor shape: {torch_tensor.shape}")

        # Step 4: Use the tensor with nn.Module
        input_size = torch_tensor.shape[1]  # Feature size
        output_size = 2  # Example output size
        from nnll.dissect import Dissector

        model = CustomModule(input_size=input_size, output_size=output_size, dtype=TORCH_DTYPE)
        dis_model = Dissector(model=model, input_shape=input_size, device="meta")
        tree = dis_model.parse()
        # print(tree)  # uses __repr__ for pretty form
        print(to_mermaid(tree))
        print(tree)  # uses __repr__ for pretty form
        # Initialize the module with matching dtype

        # Example forward pass
        output = model(torch_tensor)
        print("Output Shape:", output.shape)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

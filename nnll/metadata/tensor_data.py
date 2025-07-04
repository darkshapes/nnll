# class TensorData:
#     dtype: DTYPE_T
#     shape: List[int]
#     data_offsets: Tuple[int, int]
#     parameter_count: int = field(init=False)

#     def __post_init__(self) -> None:
#         # Taken from https://stackoverflow.com/a/13840436
#         try:
#             self.parameter_count = functools.reduce(operator.mul, self.shape)
#         except TypeError:
#             self.parameter_count = 1  # scalar value has no shape

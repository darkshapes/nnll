
"""
Study 01

spandrel
model_descriptor + registry
written by joeyballentine and rundevelopment

This is a careful dissection of code.
Imports and docstrings are removed to focus.

As this work is a precursor to nnll 07 but has the same direction,
it stands to reason we should determine what makes it great.
Which elements stand out? Can we determine the logic behind the approach?
What do we like? What should we change given our circumstances?
What makes this code so good?

Please note that this module arises out of respect for the authors,
and studied out of deference. The goal is to learn from
previous success and experience, not to claim origin. This is
their code, from which we strive to improve ourselves.

"""

"""
A system for loading, detecting, and manipulating the state of various ML models.

### `Architecture` Class
- **Purpose:** The `Architecture` class serves as an abstract base class (ABC) that provides a framework for defining different model architectures.
- **Attributes:**
  - `_id`: A unique identifier for the architecture.
  - `_name`: An optional name for the architecture; defaults to the ID if not provided.
  - `_detect`: A function that takes a `state_dict` and returns whether this architecture can handle it.

- **Methods:**
  - `detect(state_dict)`: Calls the `_detect` function to determine if this architecture matches the given state dictionary.

- **Abstract Method:**
  - `load(state_dict)`: This is an abstract method that must be implemented by subclasses. It loads a model from a `state_dict`.

### Enums and Exceptions
- **Enums:**
  - `ModelTiling`: Although not fully defined in the provided code, this likely represents different tiling strategies or properties related to how models handle input sizes.

- **Exceptions:**
  - `UnsupportedDtypeError`: Raised when trying to convert a model to an unsupported data type.

"""
from .size_req import SizeRequirements, pad_tensor
T = TypeVar("T", bound=torch.nn.Module, covariant=True) # generic type that has to be a subclass of torch.nn.Module
 # The covariant=True means that if you have two classes,
 # say MyModel1 and MyModel2, where MyModel2 is a subclass of MyModel1, then
 # MyClass[MyModel2] would be considered a subtype of MyClass[MyModel1].
StateDict = Dict[str, Any]. # statedict is defined as a dictionary
ArchId = NewType("ArchId", str) # archid is a  new type we just made up, its secretly a string but a different type of string!!

class Architecture(ABC, Generic[T]):
    def __init__(self, *, id: ArchId | str, detect: Callable[[StateDict], bool], name: str | None = None,) -> None:
        super().__init__() #this is calling Generic/ABC
        self._id: Final[ArchId] = ArchId(id) #Final type
            # Use the typing.Final object to mark a global or attribute as final, documenting that the value will never change once assigned to:
            # GLOBAL_CONSTANT: Final[str] = "This is a constant value because it is final"
            # Use the @typing.final decorator to mark a method as non-overridable
            # (subclasses can't define a different implementation) or a class as non-inheritable (you can't create subclasses from the class):
        self._name: Final[str] = name or id
        self._detect = detect # callable[state dict] - is this x model?

    @property #@property decorator defines methods that behave like attributes. Allows getter and setter methods
    # get id, which is tyoe ArchID
    def id(self) -> ArchId:
        return self._id

    # get name, which is type string
    @property
    def name(self) -> str:
        return self._name

    def detect(self, state_dict: StateDict) -> bool: #  this state dict is for this architecture? True or False
        return self._detect(state_dict)

    @abstractmethod # structure definition declares all subclasses are required to have this method
    def load(
        self, state_dict: StateDict
    ) -> ImageModelDescriptor[T] | MaskedImageModelDescriptor[T]: #load the dict once we know the architecture


Purpose = Literal["SR", "FaceSR", "Inpainting", "Restoration"]

class ModelTiling(Enum):

class UnsupportedDtypeError(Exception):

    """
    ### `ModelBase` Class
    - **Purpose:** The `ModelBase` class is another abstract base class that provides common functionality for different types of machine learning models.
    - **Attributes:**
    - `_model`: An instance of the actual PyTorch module (e.g., a neural network).
    - `_architecture`: An instance of `Architecture`, specifying which architecture this model follows.
    - `tags`: A list of strings that can be used to categorize or label the model.
    - `supports_half` and `supports_bfloat16`: Booleans indicating whether the model supports half-precision (FP16) and bfloat16, respectively.
    - `scale`, `input_channels`, `output_channels`: Model-specific parameters that define its behavior and requirements.
    - `size_requirements`: Specifies how the input size should be handled or modified by padding functions (`SizeRequirements` class).
    - `tiling`: Indicates whether tiling is supported for this model.

    - **Properties:**
    - `model`, `architecture`: Accessors for `_model` and `_architecture`.
    - `purpose`: Abstract property that must be implemented to specify the purpose of the model (e.g., SR, FaceSR, Inpainting).
    - `device`: Returns the device on which the model parameters are located.
    - `dtype`: Returns the data type of the model's parameters.

    - **Methods:**
    - `to(device=None, dtype=None)`: Moves the model to a specified device and/or changes its precision.
        - If `dtype` is provided but not supported by the architecture, it raises an `UnsupportedDtypeError`.
        - It handles positional and keyword arguments for `device` and `dtype`.

    - **Convenience Methods:**
    - `half()`, `bfloat16()`, `float()`:
        - These methods change the model's precision to half (FP16), bfloat16, or full (FP32) respectively.

    - `cpu()`, `cuda(device=None)`:
        - Moves the model to CPU or CUDA device.

    - `eval()`, `train(mode=True)`:
        - Sets the model into evaluation mode or training mode with an optional parameter for setting it to a specific mode.

    ### Summary
    This code provides a robust framework for managing different machine learning architectures and their associated models. It abstracts out common functionalities like moving models between devices, changing precision, and defining purpose-specific behavior through properties and methods. The system is designed to be extendable, allowing developers to implement concrete subclasses of `Architecture` and `ModelBase` to handle specific model types.

    This structure helps in maintaining consistency across different model implementations while providing flexibility for specialized operations. It also ensures that models are handled correctly with respect to their supported data types and device placements.
    """

class ModelBase(ABC, Generic[T]):

    def __init__(self, model: T, state_dict: StateDict, architecture: Architecture[T], tags: list[str],
        supports_half: bool, supports_bfloat16: bool,scale: int,
        input_channels: int,output_channels: int,size_requirements: SizeRequirements | None = None,
        tiling: ModelTiling = ModelTiling.SUPPORTED,
    ):
        self._model: T = model
        self._architecture: Architecture[T] = architecture
        self.tags: list[str] = tags
        self.supports_half: bool = supports_half
        self.supports_bfloat16: bool = supports_bfloat16
        self.scale: int = scale
        self.input_channels: int = input_channels
        self.output_channels: int = output_channels
        self.size_requirements: SizeRequirements = (
            size_requirements or SizeRequirements()
        )
        self.tiling: ModelTiling = tiling
        self.model.load_state_dict(state_dict)  # type: ignore

    @property
    def model(self) -> T:
        return self._model

    @property
    def architecture(self) -> Architecture[T]:
        return self._architecture

    @property
    @abstractmethod
    def purpose(self) -> Purpose:
        ...

    @property
    def device(self) -> torch.device:
        # This makes the following assumptions:
        # - The model is on a single device
        # - The model has at least one parameter
        # Both are true for all models implemented in Spandrel.
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        # this makes the same assumptions as `device`
        return next(self.model.parameters()).dtype

    @overload
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self: ...

    @overload
    def to(self, dtype: torch.dtype) -> Self: ...

    def to(self, *args: object, **kwargs) -> Self:
        # turn positional arguments into keyword arguments
        def set_kw(name: str, value: object):
            if name in kwargs:
                raise TypeError(f"to() got multiple values for keyword argument {name}")
            kwargs[name] = value

        if len(args) == 1:
            arg: object = args[0]
            if isinstance(arg, torch.dtype):
                set_kw("dtype", arg)
            elif isinstance(arg, (torch.device, str)) or arg is None:
                set_kw("device", arg)
            else:
                raise TypeError(
                    f"to() expected a torch.device or torch.dtype, but got {type(arg)}"
                )
        elif len(args) == 2:
            set_kw("device", args[0])
            set_kw("dtype", args[1])
        elif len(args) > 2:
            raise TypeError(
                f"to() expected at most 2 positional arguments, got {len(args)}"
            )

        device: torch.device | str | None = kwargs.pop("device", None)
        dtype: torch.dtype | None = kwargs.pop("dtype", None)

        if len(kwargs) > 0:
            raise TypeError(f"to() got unexpected keyword arguments {list(kwargs)}")

        if dtype is not None:
            if dtype == torch.float16 and not self.supports_half:
                raise UnsupportedDtypeError(
                    f"{self.architecture} does not support half precision (fp16)"
                )
            if dtype == torch.bfloat16 and not self.supports_bfloat16:
                raise UnsupportedDtypeError(
                    f"{self.architecture} does not support bfloat16 precision"
                )

        if isinstance(device, str):
            device = torch.device(device)

        self.model.to(device=device, dtype=dtype)
        return self

    def half(self) -> Self:
        self.to(torch.half)
        return self

    def bfloat16(self) -> Self:
        self.to(torch.bfloat16)
        return self

    def float(self) -> Self:
        self.to(torch.float)
        return self

    def cpu(self) -> Self:
        self.model.cpu()
        return self

    def cuda(self, device: int | None = None) -> Self:
        self.model.cuda(device)
        return self

    def eval(self) -> Self:
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> Self:
        self.model.train(mode)
        return self


    """
    This code defines a system for managing and detecting different machine learning architectures, primarily using PyTorch models.

    #### `ArchSupport` Dataclass

    The `ArchSupport` dataclass is used to encapsulate information about an architecture and how to detect if a given state dictionary matches that architecture.

    - **Attributes:**
    - `architecture`: An instance of the `Architecture` class, which represents a specific model architecture.
    - `detect`: A callable function that takes a `StateDict`  and returns a boolean indicating whether this architecture can handle it.
    - `before`: A tuple of IDs indicating other architectures that should be checked before this one.

    - **Static Method:**
    - `from_architecture(arch, before=())`: This method creates an instance of `ArchSupport` from an `Architecture` object and a list of dependencies (`before`).
        It uses the architecture's own detection function to determine if it can handle a given state dictionary.
    """
    @dataclass(frozen=True)
    class ArchSupport:
        architecture: Architecture  # The architecture object
        detect: Callable[[StateDict], bool]  # Function to check compatibility with state dict
        before: tuple[str, ...] = ()  # Other architectures that should be checked first

        @staticmethod
        def from_architecture(arch: Architecture, before: Iterable[str] = ()) -> ArchSupport:
            return ArchSupport(architecture=arch, detect=arch.detect, before=tuple(before))

   #### `ArchRegistry` Class
    """
    The `ArchRegistry` class is used to manage and organize different architectures.
    It allows for adding, checking, and loading models based on their state dictionaries.

    - **Attributes:**
    - `_architectures`: A sequence of `ArchSupport` objects that represent all the registered architectures.
    - `_ordered`: A list of `ArchSupport` objects ordered according to the dependencies specified in each architecture's `before` attribute.
    - `_by_id`: A dictionary mapping architecture IDs to their corresponding `ArchSupport` objects.

    - **Methods:**

    - `__contains__(id)`: Checks if an ID is registered in the registry.
    - `__getitem__(id)`: Retrieves an `ArchSupport` object by its ID.
    - `__iter__()`: Returns an iterator over the architectures.
    - `__len__()`: Returns the number of registered architectures.
    - `get(id)`: Retrieves an `ArchSupport` object by its ID, returning `None` if it's not found.
    - `architectures(order="insertion")`: Returns a list of `ArchSupport` objects in either insertion or detection order (the latter being the default).
    - `add(*architectures, ignore_duplicates=False)`: Adds new architectures to the registry.
    If `ignore_duplicates` is `True`, it will silently skip adding duplicates; otherwise, it raises an error.
    - `_get_ordered(architectures)`: A static method that computes a topologically sorted list of architectures based on their dependencies.

    - **Loading Models:**
    - `load(state_dict)`: This method takes a state dictionary and attempts to load the appropriate model by detecting which architecture can handle it.
    It iterates over the ordered list of architectures, checking each one with its `detect` function until a match is found.
    If no match is found, it raises an `UnsupportedModelError`.

    ### Detailed Explanation


    ### Summary

    - **`ArchSupport`:** Represents an architecture with a detection function and dependencies.
    - **`ArchRegistry`:** Manages a collection of architectures, ensuring they are added correctly and in the right order based on their dependencies.
    - **Loading Models:** Uses a topologically sorted list of architectures to find and load the appropriate model based on the provided state dictionary. If no match is found, it raises an error.

    This setup allows for flexible management and loading of different models or architectures by ensuring that each architecture's compatibility with given data can be checked efficiently.
    """

    class ArchRegistry:
        def __init__(self):
            self._architectures = []
            self._ordered = []
            self._by_id = {}

        def __contains__(self, id: Union[ArchId, str]) -> bool:
            return ArchId(id) in self._by_id

        def __getitem__(self, id: Union[str, ArchId]) -> ArchSupport:
            return self._by_id[ArchId(id)]

        def __iter__(self):
            return iter(self.architectures("insertion"))

        def __len__(self) -> int:
            return len(self._architectures)

        def get(self, id: Union[str, ArchId]) -> Optional[ArchSupport]:
            return self._by_id.get(ArchId(id), None)

        def architectures(self, order: Literal["insertion", "detection"] = "insertion") -> list[ArchSupport]:
            if order == "insertion":
                return list(self._architectures)
            elif order == "detection":
                return list(self._ordered)
            else:
                raise ValueError(f"Invalid order: {order}")

        def add(
            self,
            *architectures: ArchSupport,
            ignore_duplicates: bool = False
        ) -> list[ArchSupport]:
            new_architectures = list(self._architectures)
            new_by_id = dict(self._by_id)
            added = []

            for arch in architectures:
                if arch.architecture.id in new_by_id:
                    if ignore_duplicates:
                        continue
                    raise DuplicateArchitectureError(f"Duplicate architecture: {arch.architecture.id}")

                new_architectures.append(arch)
                new_by_id[arch.architecture.id] = arch
                added.append(arch)

            self._architectures = new_architectures
            self._ordered = ArchRegistry._get_ordered(new_architectures)
            self._by_id = new_by_id

            return added

        @staticmethod
        def _get_ordered(architectures: list[ArchSupport]) -> list[ArchSupport]:
            inv_before: dict[ArchId, list[ArchId]] = {}
            by_id: dict[ArchId, ArchSupport] = {}

            for arch in architectures:
                by_id[arch.architecture.id] = arch
                for before in arch.before:
                    if before not in inv_before:
                        inv_before[before] = []
                    inv_before[before].append(arch.architecture.id)

            ordered: list[ArchSupport] = []
            seen: set[ArchId] = set()
            stack: list[ArchId] = []

            def visit(id: ArchId):
                if id in stack:
                    raise ValueError(f"Circular dependency detected for architecture {id}")
                if id not in seen:
                    stack.append(id)
                    for before_id in inv_before.get(id, []):
                        visit(before_id)
                    ordered.append(by_id[id])
                    seen.add(id)
                    stack.pop()

            for arch in architectures:
                visit(arch.architecture.id)

            return ordered

        def load(self, state_dict: StateDict) -> Model:
            for arch_support in self._ordered:
                if arch_support.detect(state_dict):
                    model = arch_support.architecture.load(state_dict)
                    return model
            raise UnsupportedModelError(f"No suitable architecture found for the given state dict.")
    ```

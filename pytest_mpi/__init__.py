from .parallel_assert import parallel_assert  # noqa: F401
from .detect_mpiexec_implementation import detect_mpiexec_implementation

impl = detect_mpiexec_implementation()

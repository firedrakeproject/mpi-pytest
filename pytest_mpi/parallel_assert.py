import warnings

from mpi4py import MPI


def parallel_assert(assertion: bool, msg: str = "", *, participating: bool = True) -> None:
    """Make an assertion across ``COMM_WORLD``.

    Parameters
    ----------
    assertion :
        The assertion to check. If this is `False` on any participating task, an
        `AssertionError` will be raised. This argument can also be a callable
        that returns a `bool` (deprecated).
    msg :
        Optional error message to print out on failure.
    participating :
        Whether the given rank should evaluate the assertion.

    Notes
    -----
    It is very important that ``parallel_assert`` is called collectively on all
    ranks simultaneously.


    Example
    -------
    Where in serial code one would have previously written:
    ```python
    x = f()
    assert x < 5, "x is too large"
    ```

    Now write:
    ```python
    x = f()
    parallel_assert(x < 5, "x is too large")
    ```

    """
    if participating:
        if callable(assertion):
            warnings.warn("Passing callables to parallel_assert is no longer "
                          "recommended. Please pass booleans instead.",
                          FutureWarning)
            result = assertion()
        else:
            result = assertion
    else:
        result = True

    all_results = MPI.COMM_WORLD.allgather(result)
    if not min(all_results):
        raise AssertionError(
            "Parallel assertion failed on ranks: "
            f"{[rank for rank, result in enumerate(all_results) if not result]}\n"
            + msg
        )

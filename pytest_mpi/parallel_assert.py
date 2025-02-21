import warnings

from mpi4py import MPI


def parallel_assert(assertion: bool, participating: bool = True, msg: str = "") -> None:
    """Make an assertion across ``COMM_WORLD``.

    Parameters
    ----------
    assertion :
        If this is `False` on any participating task, an `AssertionError` will 
        be raised.
    participating :
        Whether the given rank should evaluate the assertion.
    msg :
        Optional error message to print out on failure.

    Notes
    -----
    It is very important that ``parallel_assert`` is called collectively on all
    ranks simultaneously.
    This function allows passing a callable instead of a boolean for the
    `assertion` argument. This is useful in rare circumstances, such as when
    the assertion is not defined on all tasks, but is not recommended.


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
            warnings.warn('Passing callables to parallel_assert is no longer recommended.'
                          'Please pass booleans instead.')
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

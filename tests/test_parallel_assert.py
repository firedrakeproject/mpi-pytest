import pytest
from pytest_mpi.parallel_assert import parallel_assert


@pytest.mark.parametrize('expression', [True, False])
def test_parallel_assert_equivalent_to_assert_in_serial(expression):
    try:
        parallel_assert(expression)
        parallel_raised_exception = False
    except AssertionError:
        parallel_raised_exception = True  
    try:
        assert expression
        serial_raised_exception = False
    except AssertionError:
        serial_raised_exception = True

    assert serial_raised_exception == parallel_raised_exception


@pytest.mark.parallel([1, 2, 3])
def test_parallel_assert_all_tasks():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    expression = comm.rank < comm.size // 2

    try:
        parallel_assert(expression)
        raised_exception = False
    except AssertionError:
        raised_exception = True

    assert raised_exception, f'No exception raised on rank {comm.rank}!'


@pytest.mark.parallel([1, 2, 3])
def test_parallel_assert_participating_tasks_only():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    expression = comm.rank < comm.size // 2
    raised_exception = False

    try:
        parallel_assert(expression, participating=expression)
    except AssertionError:
        raised_exception = True

    assert not raised_exception, f'Exception raised on rank {comm.rank}!'


@pytest.mark.parallel([1, 2, 3])
def test_legacy_parallel_assert():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    expression = comm.rank < comm.size // 2
    raised_exception = False

    try:
        parallel_assert(lambda: expression, participating=expression)
    except AssertionError:
        raised_exception = True

    assert not raised_exception, f'Exception raised on rank {comm.rank}!'

import pytest
from mpi4py import MPI


@pytest.mark.parallel
def test_parallel_marker_no_args():
    assert MPI.COMM_WORLD.size == 3


@pytest.mark.parallel(2)
def test_parallel_marker_with_int():
    assert MPI.COMM_WORLD.size == 2


@pytest.mark.parallel([2, 3])
def test_parallel_marker_with_list():
    assert MPI.COMM_WORLD.size in {2, 3}

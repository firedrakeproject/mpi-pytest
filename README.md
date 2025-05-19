# mpi-pytest

Pytest plugin that lets you run tests in parallel with MPI.

`mpi-pytest` provides:

* A `parallel` marker indicating that the test must be run under MPI.
* A `parallel_assert` function for evaluating assertions in a deadlock-safe way.

## `parallel` marker

Writing a parallel test simply requires marking the test with the `parallel` marker:

```py
@pytest.mark.parallel(nprocs=5)  # run in parallel with 5 processes
def test_my_code_on_5_procs():
    ...

@pytest.mark.parallel(5)  # the "nprocs" kwarg is optional
def test_my_code_on_5_procs_again():
    ...

@pytest.mark.parallel  # run in parallel with the default number of processes (3)
def test_my_code_on_3_procs():
    ...

@pytest.mark.parallel()  # the brackets are optional
def test_my_code_on_3_procs_again():
    ...
```

One can also mark a test with a sequence of values for `nprocs`:

```py
@pytest.mark.parallel(nprocs=[1, 2, 3])  # run in parallel on 1, 2 and 3 processes
def test_my_code_on_variable_nprocs():
    ...

@pytest.mark.parallel([1, 2, 3])  # again the "nprocs" kwarg is optional
def test_my_code_on_variable_nprocs_again():
    ...
```

If multiple numbers of processes are requested then the tests are parametrised
and renamed to, in this case, `test_my_code_on_variable_nprocs[nprocs=1]`,
`test_my_code_on_variable_nprocs[nprocs=2]` and
`test_my_code_on_variable_nprocs[nprocs=3]`.

When running the code with these `parallel` markers, `mpi-pytest` adds extra markers
to each test to allow one to select all tests with a particular number of processors.
For example, to select all parallel tests on 3 processors, one should run:

```bash
$ mpiexec -n 3 pytest -m "parallel[3]"
```

Serial tests - those either unmarked or marked `@pytest.mark.parallel(1)` - can
be selected by running:

```bash
$ pytest -m "not parallel or parallel[1]"
```

### Forking mode

`mpi-pytest` can be used in one of two modes: forking or non-forking. The former
works as follows:

1. The user calls `pytest` (not `mpiexec -n <# proc> pytest`). This launches
   the "parent" `pytest` process.
2. This parent `pytest` process collects all the tests and begins to run them.
3. When a test is found with the `parallel` marker, rather than executing the
   function as before, a subprocess is forked calling
   `mpiexec -n <# proc> pytest this_specific_test_file.py::this_specific_test`.
   This produces `<# proc>` 'child' `pytest` processes that execute the
   test together.
4. If this terminates successfully then the test is considered to have passed.

This is convenient for development for a number of reasons:

* The plugin composes better with other pytest plugins like `pytest-xdist`.
* It is not necessary to wrap `pytest` invocations with `mpiexec` calls, and
  all parallel and serial tests can be run at once.

There are however a number of downsides:

* Only one mainstream MPI distribution ([MPICH](https://www.mpich.org/)) supports
  nested calls to `MPI_Init`. If your 'parent' `pytest` process initialises MPI
  (for instance by executing `from mpi4py import MPI`) then this will cause non-MPICH
  MPI distributions to crash.
* Forking a subprocess can be expensive since a completely fresh Python interpreter
  is launched each time.
* Sandboxing each test means that polluted global state at the end of a test cannot
  be detected.

### Non-forking mode

With these significant limitations in mind, `mpi-pytest` therefore also supports
a non-forking mode. To use it, one simply needs to wrap the `pytest` invocation
with `mpiexec`, no additional configuration is necessary. For example, to run
all of the parallel tests on 2 ranks one needs to execute:

```bash
$ mpiexec -n 2 pytest -m "parallel[2]"
```

## `parallel_assert`

Using regular `assert` statements can be unsafe in parallel codes. Consider the
code:

```py
@pytest.mark.parallel(2)
def test_something():
    # this will only fail on *some* ranks
    assert COMM_WORLD.rank == 0

    # this will hang
    COMM_WORLD.barrier()
```

One can see that failing assertions on some ranks but not others will violate SPMD
and lead to deadlocks. To avoid this, `mpi-pytest` provides a `parallel_assert`
function used as follows:

```py
from pytest_mpi import parallel_assert

@pytest.mark.parallel(2)
def test_something():
    # this will fail on *all* ranks
    parallel_assert(COMM_WORLD.rank == 0)
    ...
```

## Configuration

`mpi-pytest` respects the environment variable `PYTEST_MPI_MAX_NPROCS`, which defines
the maximum number of processes that can be requested by a parallel marker. If this
value is exceeded an error will be raised.

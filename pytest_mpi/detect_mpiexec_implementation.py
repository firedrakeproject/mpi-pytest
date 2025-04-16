import subprocess

def detect_mpiexec_implementation():
    try:
        result = subprocess.run(
            ["mpiexec", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        output = result.stdout.lower()

        if "open mpi" in output or "open-rte" in output:
            return "Open MPI"
        elif "mpich" in output:
            return "MPICH"
        else:
            return "Unknown MPI implementation"
    except FileNotFoundError:
        raise FileNotFoundError("mpiexec not found in PATH")
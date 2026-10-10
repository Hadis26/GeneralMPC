"""Optional C kernel generation; no handwritten dynamics or C++ bindings needed."""
from pathlib import Path
import os
import subprocess
import sys
import casadi as ca


def export_kernels(controller,directory):
    """Generate C for model, discrete map, AD sensitivities and QP assembly.

    The nonlinear solver loop remains Python. These are kernels, not a complete
    standalone MPC controller. Use an application-specific generated library.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    p = controller.problem
    generator = ca.CodeGenerator("mpc_kernels.c",{"with_header":True})
    for function in [p.model.rhs,p.model.stage,p.model.terminal,p.step,p.step_jac,
                     p.evaluate,p.linearize]:
        generator.add(function)
    generator.generate(str(directory)+os.sep)
    return directory/"mpc_kernels.c"


def compile_kernels(source,compiler="cc"):
    """Linux/macOS shared library. On Windows use MSVC as described in the guide."""
    source = Path(source).resolve()
    suffix = ".dylib" if sys.platform=="darwin" else ".so"
    library = source.with_name("mpc_kernels"+suffix)
    subprocess.run([compiler,"-O3","-fPIC","-shared",str(source),"-o",str(library),"-lm"],check=True)
    return library



def load_kernels(controller, library):
    """Load application-specific generated C kernels."""
    library = str(Path(library).resolve())
    p = controller.problem

    p.step = ca.external("step", library)
    p.step_jac = ca.external("step_jac", library)
    p.evaluate = ca.external("evaluate", library)
    p.linearize = ca.external("linearize", library)


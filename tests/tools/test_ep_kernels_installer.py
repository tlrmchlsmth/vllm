# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import os
import subprocess
from pathlib import Path

import pytest
from packaging.requirements import Requirement


@pytest.mark.parametrize("mode", ["install", "wheel"])
@pytest.mark.parametrize("rubin,version", [(False, "4.7.1"), (True, "4.8.0.dev0")])
def test_reconcile_moonep_cutlass_preserves_runtime_pin(tmp_path, mode, rubin, version):
    """Reconcile MoonEP CUTLASS in package metadata before install or wheel build."""
    installer = (
        Path(__file__).resolve().parents[2]
        / "tools/ep_kernels/install_python_libraries.sh"
    ).read_text()
    functions = installer.split("is_git_dirty() {", 1)[1].split("# build DeepEP", 1)[0]
    package = tmp_path / "MoonEP"
    package.mkdir()
    (package / "setup.py").write_text(
        'install_requires = ["nvidia-cutlass-dsl==4.6.2"]'
    )
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "libcuda.so").touch()
    subprocess.run(
        [
            "bash",
            "-ec",
            "is_git_dirty() {"
            + functions
            + """
clone_repo() { :; }
uv() {
    if [[ "$*" == *show* ]]; then
        echo "Version: $TEST_CUTLASS_VERSION"
    else
        cp setup.py "$WORKSPACE/built-metadata.py"
    fi
}
do_build unused MoonEP setup.py unused ''
""",
        ],
        check=True,
        env={
            "PATH": os.environ["PATH"],
            "WORKSPACE": str(tmp_path),
            "WHEEL_DIR": str(tmp_path),
            "CUDA_HOME": str(tmp_path),
            "CUDA_VERSION_MAJOR": "13",
            "MODE": mode,
            "VIRTUAL_ENV": "",
            "INSTALL_RUBIN_PRERELEASE": str(rubin).lower(),
            "TEST_CUTLASS_VERSION": version,
        },
        capture_output=True,
        text=True,
    )
    metadata = ast.parse((tmp_path / "built-metadata.py").read_text())
    requirements = ast.literal_eval(metadata.body[0].value)
    requirement = Requirement(requirements[0])
    assert requirement.name == "nvidia-cutlass-dsl"
    assert str(requirement.specifier) == f"=={version}"

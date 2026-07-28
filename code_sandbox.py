"""
Cronus code sandbox.

Flow: GLM (the coding specialist) writes code for the request -> if you
choose to execute, it runs -> if it errors, the error is fed straight back
to GLM to fix -> repeats up to MAX_SANDBOX_LOOPS times -> returns the final
working (or best-effort) code plus the execution transcript.

SCOPE: Python execution is real and works out of the box. Other languages
are detected and the code is still generated, but execution is not
supported without the language's compiler/runtime installed on the server
-- see language support notes below.

SECURITY NOTE: this runs arbitrary code via subprocess with a timeout and
output cap, but this is NOT a hardened sandbox (no network isolation, no
filesystem isolation, no memory cap). On a publicly shared deployment,
treat this as a real abuse surface, not a solved problem -- true isolation
needs a containerized sandbox (e.g. Docker with --network none, resource
limits), which is a bigger infra step than this file covers.
"""

import re
import subprocess
import tempfile
import os
from typing import Tuple

MAX_SANDBOX_LOOPS = 3
EXECUTION_TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 4000

# NOTE: Python always works out of the box. Java/C/C++ need their compilers
# installed on the server (javac+java, gcc, g++ respectively) -- Render's
# default Python runtime does NOT have these. See the deployment note at
# the bottom of this file for what's needed to actually enable them.
SUPPORTED_EXECUTION_LANGUAGES = {"python", "py", "java", "c", "cpp", "c++"}


def extract_code_block(text: str) -> Tuple[str, str]:
    """
    Pulls the first fenced code block out of a model response.
    Returns (language, code). Language defaults to 'python' if unspecified.
    """
    match = re.search(r"```(\w*)\n(.*?)```", text, re.DOTALL)
    if not match:
        return "python", text.strip()  # no fence found, assume the whole reply is code
    lang = match.group(1).strip().lower() or "python"
    code = match.group(2).strip()
    return lang, code


def run_python(code: str) -> dict:
    """Executes Python code in a subprocess with a timeout and output cap."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True, text=True, timeout=EXECUTION_TIMEOUT_SECONDS,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:MAX_OUTPUT_CHARS],
            "stderr": result.stderr[:MAX_OUTPUT_CHARS],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {EXECUTION_TIMEOUT_SECONDS}s (possible infinite loop).",
            "returncode": None,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def run_java(code: str) -> dict:
    """
    Java requires the file name to match the public class name.
    Extracts the class name, writes the file accordingly, compiles, runs.
    """
    match = re.search(r"public\s+class\s+(\w+)", code)
    class_name = match.group(1) if match else "Main"
    if not match:
        # No public class found -- wrap defensively isn't safe to guess at,
        # so just try the literal code as-is under a default name and let
        # the compiler error (fed back to the self-correction loop) guide the fix.
        pass

    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, f"{class_name}.java")
    with open(file_path, "w") as f:
        f.write(code)

    try:
        compile_result = subprocess.run(
            ["javac", file_path],
            capture_output=True, text=True, timeout=EXECUTION_TIMEOUT_SECONDS, cwd=tmp_dir,
        )
        if compile_result.returncode != 0:
            return {
                "success": False, "stdout": "",
                "stderr": f"Compile error:\n{compile_result.stderr[:MAX_OUTPUT_CHARS]}",
                "returncode": compile_result.returncode,
            }

        run_result = subprocess.run(
            ["java", "-cp", tmp_dir, class_name],
            capture_output=True, text=True, timeout=EXECUTION_TIMEOUT_SECONDS,
        )
        return {
            "success": run_result.returncode == 0,
            "stdout": run_result.stdout[:MAX_OUTPUT_CHARS],
            "stderr": run_result.stderr[:MAX_OUTPUT_CHARS],
            "returncode": run_result.returncode,
        }
    except FileNotFoundError:
        return {
            "success": False, "stdout": "",
            "stderr": "Java is not installed on this server (javac/java not found). "
                      "Code was generated but cannot run here yet -- see deployment notes.",
            "returncode": None,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"Timed out after {EXECUTION_TIMEOUT_SECONDS}s.", "returncode": None}


def run_c_family(code: str, compiler: str, ext: str) -> dict:
    """Shared logic for C (gcc) and C++ (g++)."""
    tmp_dir = tempfile.mkdtemp()
    src_path = os.path.join(tmp_dir, f"program.{ext}")
    bin_path = os.path.join(tmp_dir, "program")
    with open(src_path, "w") as f:
        f.write(code)

    try:
        compile_result = subprocess.run(
            [compiler, src_path, "-o", bin_path],
            capture_output=True, text=True, timeout=EXECUTION_TIMEOUT_SECONDS,
        )
        if compile_result.returncode != 0:
            return {
                "success": False, "stdout": "",
                "stderr": f"Compile error:\n{compile_result.stderr[:MAX_OUTPUT_CHARS]}",
                "returncode": compile_result.returncode,
            }

        run_result = subprocess.run(
            [bin_path], capture_output=True, text=True, timeout=EXECUTION_TIMEOUT_SECONDS,
        )
        return {
            "success": run_result.returncode == 0,
            "stdout": run_result.stdout[:MAX_OUTPUT_CHARS],
            "stderr": run_result.stderr[:MAX_OUTPUT_CHARS],
            "returncode": run_result.returncode,
        }
    except FileNotFoundError:
        return {
            "success": False, "stdout": "",
            "stderr": f"{compiler} is not installed on this server. "
                      f"Code was generated but cannot run here yet -- see deployment notes.",
            "returncode": None,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"Timed out after {EXECUTION_TIMEOUT_SECONDS}s.", "returncode": None}


def run_code(language: str, code: str) -> dict:
    """Dispatches to the right runner based on detected language."""
    lang = language.lower()
    if lang in ("python", "py"):
        return run_python(code)
    if lang == "java":
        return run_java(code)
    if lang == "c":
        return run_c_family(code, "gcc", "c")
    if lang in ("cpp", "c++"):
        return run_c_family(code, "g++", "cpp")
    return {"success": False, "stdout": "", "stderr": f"No runner for language '{language}'.", "returncode": None}



    if prior_code and prior_error:
        prompt = f"""The following code was generated for this request:
Request: {request}

Code:
```
{prior_code}
```

Running it produced this error:
{prior_error}

Fix the code so it runs correctly. Return ONLY a single fenced code block with the corrected code -- no explanation outside the fence."""
    else:
        prompt = f"""Write code for this request: {request}

Return ONLY a single fenced code block (with the language specified, e.g. ```python) containing complete, runnable code -- no explanation outside the fence."""

    resp = glm_client.chat.completions.create(
        model=glm_model,
        messages=[
            {"role": "system", "content": "You are a precise coding assistant. Follow the output format instructions exactly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def generate_with_self_correction(glm_client, glm_model: str, request: str, execute: bool) -> dict:
    """
    Returns {
        "language": str, "code": str, "executed": bool, "success": bool or None,
        "stdout": str, "stderr": str, "loops": int, "log": [...]
    }
    """
    raw = generate_code(glm_client, glm_model, request)
    language, code = extract_code_block(raw)
    log = [{"step": "generate", "code": code}]

    if not execute:
        return {
            "language": language, "code": code, "executed": False, "success": None,
            "stdout": "", "stderr": "", "loops": 0, "log": log,
        }

    if language not in SUPPORTED_EXECUTION_LANGUAGES:
        return {
            "language": language, "code": code, "executed": False, "success": None,
            "stdout": "",
            "stderr": f"Execution for '{language}' isn't available on this server yet -- "
                      f"only Python execution is currently supported. Code was generated but not run.",
            "loops": 0, "log": log,
        }

    for loop_num in range(MAX_SANDBOX_LOOPS):
        result = run_code(language, code)
        log.append({"step": "execute", "loop": loop_num, "result": result})

        if result["success"]:
            return {
                "language": language, "code": code, "executed": True, "success": True,
                "stdout": result["stdout"], "stderr": result["stderr"],
                "loops": loop_num, "log": log,
            }

        # Failed -- feed the error back for a fix, unless we're out of attempts
        if loop_num == MAX_SANDBOX_LOOPS - 1:
            return {
                "language": language, "code": code, "executed": True, "success": False,
                "stdout": result["stdout"], "stderr": result["stderr"],
                "loops": loop_num, "log": log,
            }

        raw = generate_code(glm_client, glm_model, request, prior_code=code, prior_error=result["stderr"])
        language, code = extract_code_block(raw)
        log.append({"step": "revise", "code": code})

    return {
        "language": language, "code": code, "executed": True, "success": False,
        "stdout": "", "stderr": "Exhausted self-correction attempts.",
        "loops": MAX_SANDBOX_LOOPS, "log": log,
    }

# ---------------------------------------------------------------------------
# DEPLOYMENT NOTE -- read this before expecting Java/C/C++ to actually run
# ---------------------------------------------------------------------------
# Render's default "Python 3" runtime does NOT include javac/java, gcc, or
# g++. Without them, Java/C/C++ code will still be GENERATED correctly, but
# execution will fail with an honest "not installed" message.
#
# To actually enable execution for these languages, switch your Render
# service from the native Python runtime to a DOCKER-based deployment with
# a Dockerfile that installs the needed toolchains, e.g.:
#
#   FROM python:3.11-slim
#   RUN apt-get update && apt-get install -y default-jdk gcc g++ && \
#       rm -rf /var/lib/apt/lists/*
#   WORKDIR /app
#   COPY requirements.txt .
#   RUN pip install -r requirements.txt
#   COPY . .
#   CMD ["python", "app.py"]
#
# This is a bigger infra change than adding a Python package -- it changes
# how Render builds and runs your service, and default-jdk adds real size
# to the build. Worth doing once you're ready to commit to it, not required
# just to get Python execution working today.

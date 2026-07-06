#!/usr/bin/env bash
set -euo pipefail

# Ensure Julia is available. juliaup installs the executable in ~/.juliaup/bin,
# which is not always present in non-login shells used by scripts/conda.
JULIAUP_BIN="$HOME/.juliaup/bin"
export PATH="$JULIAUP_BIN:$PATH"

if ! command -v julia >/dev/null 2>&1; then
    echo "Julia could not be found; installing Julia with juliaup..."
    curl -fsSL https://install.julialang.org | sh -s -- -y
    export PATH="$JULIAUP_BIN:$PATH"
fi

if ! command -v julia >/dev/null 2>&1; then
    echo "ERROR: Julia installation finished, but 'julia' is still not on PATH." >&2
    echo "Add this to your shell profile and re-run this script:" >&2
    echo "  export PATH=\"$JULIAUP_BIN:\$PATH\"" >&2
    exit 1
fi

echo "Using Julia: $(command -v julia)"
julia --version

# Ensure that conda is installed.
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda could not be found, please install Conda first." >&2
    exit 1
fi

# Create or update virtual environment.
if conda env list | awk '{print $1}' | grep -qx 'lvd-pg'; then
    conda env update --name lvd-pg --file environment.yml --prune
else
    conda env create --file environment.yml
fi

# Run subsequent commands inside the lvd-pg environment.
eval "$(conda shell.bash hook)"
conda activate lvd-pg

# If the conda env does not provide a julia executable, expose the juliaup one
# inside the env so subprocesses that use only $CONDA_PREFIX/bin can find it.
# Use a wrapper instead of a symlink because juliaup's launcher reads config
# relative to its real executable path.
if [ ! -x "$CONDA_PREFIX/bin/julia" ] && [ -x "$JULIAUP_BIN/julia" ]; then
    cat > "$CONDA_PREFIX/bin/julia" <<EOF
#!/usr/bin/env bash
exec "$JULIAUP_BIN/julia" "\$@"
EOF
    chmod +x "$CONDA_PREFIX/bin/julia"
fi

python --version

# Install PyJulia/PyCall dependencies.
python -c "import julia; julia.install()"

# Install Julia dependencies and bind PyCall to this conda environment's Python.
julia --project=. -e 'using Pkg; Pkg.instantiate(); ENV["PYTHON"] = abspath(ENV["CONDA_PREFIX"] * "/bin/python"); Pkg.build("PyCall")'

mkdir -p exps/progressive_growing/data/data_imagenet32
touch exps/progressive_growing/data/data_imagenet32/data_trn.npy
touch exps/progressive_growing/data/data_imagenet32/data_val.npy

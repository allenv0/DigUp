#!/bin/bash
# Copies MLX's generated Metal kernel library where MLX looks for it. SwiftPM leaves it inside a resource
# bundle that the running binaries don't search, and MLX fails without it ("Failed to load the default
# metallib"). Two placements cover everything:
#   - beside the built binaries (digup, DigUpApp): MLX checks the running binary's folder first
#   - default.metallib in the package root: the one place `swift test` looks for (tests run from there)
# build.sh and CI run this after building; a dev running `swift test` by hand runs it once per clean build.
# Nothing to do when MLX isn't built (Linux/Windows) or everything is in place.
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -d .build ]] || exit 0
metallib="$(find .build -path '*mlx-swift_Cmlx.bundle*' -name 'default.metallib' -print 2>/dev/null | head -1)"
[[ -n "$metallib" ]] || exit 0

copied=0
for dir in .build/debug .build/*/debug .build/*/Products/*; do
    [[ -d "$dir" ]] || continue
    for product in "$dir/digup" "$dir/DigUpApp"; do
        [[ -e "$product" ]] || continue
        if [[ ! -f "$dir/mlx.metallib" || "$metallib" -nt "$dir/mlx.metallib" ]]; then
            cp "$metallib" "$dir/mlx.metallib"
            copied=$((copied + 1))
        fi
    done
done
if [[ ! -f default.metallib || "$metallib" -nt default.metallib ]]; then
    cp "$metallib" default.metallib
    copied=$((copied + 1))
fi
[[ $copied -eq 0 ]] || echo "mlx-metallib: placed MLX's Metal kernels ($copied)"

#!/usr/bin/env bash
# The compiler toolchain the fused kernels build with: nvcc on PATH and a g++ that speaks C++17.
# Sourced by the bundles' run.sh:
#
#   require_toolchain    # returns 2 with the fix on stderr if the toolchain is missing
#
# When it is missing and an environment-modules `module` command can be found, GM_TOOLCHAIN_MODULES
# (the CUDA and compiler modules to load, if set) is loaded first; what that puts in the environment is inherited by
# everything the script launches.

toolchain_ok() {
  command -v nvcc >/dev/null 2>&1 || return 1
  local major
  major="$(g++ -dumpversion 2>/dev/null | cut -d. -f1)"
  [ -n "$major" ] && [ "$major" -ge 7 ]
}

require_toolchain() {
  toolchain_ok && return 0
  if ! type module >/dev/null 2>&1; then
    local init
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
                "${MODULESHOME:-/nonexistent}/init/bash" "${LMOD_PKG:-/nonexistent}/init/bash"; do
      if [ -r "$init" ]; then
        # shellcheck disable=SC1090
        source "$init"
        break
      fi
    done
  fi
  if type module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    if [ -n "${GM_TOOLCHAIN_MODULES:-}" ]; then module load ${GM_TOOLCHAIN_MODULES} >/dev/null 2>&1 || true; fi
  fi
  toolchain_ok && return 0
  echo "run.sh: the fused kernels need nvcc and a C++17 g++ on PATH;" \
       "found $(command -v nvcc 2>/dev/null || echo 'no nvcc') and g++ $(g++ -dumpversion 2>/dev/null || echo '?')." \
       "With environment modules, set GM_TOOLCHAIN_MODULES to the CUDA and compiler modules to load." >&2
  return 2
}

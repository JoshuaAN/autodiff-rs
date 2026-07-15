set -euo pipefail

IPOPT_INCLUDE="${IPOPT_INCLUDE:-/opt/homebrew/Cellar/ipopt/3.14.19/include/coin-or}"

bindgen wrapper.h \
    --allowlist-function '(Create|Free)IpoptProblem' \
    --allowlist-function 'AddIpopt(Num|Str|Int)Option' \
    --allowlist-function 'IpoptSolve' \
    --allowlist-function 'SetIntermediateCallback' \
    --allowlist-function '(Open|Set)IpoptOutputFile' \
    --allowlist-function 'SetIpoptProblemScaling' \
    --allowlist-type 'ApplicationReturnStatus' \
    --allowlist-type 'AlgorithmMode' \
    --no-doc-comments \
    --rustified-enum 'ApplicationReturnStatus' \
    -o src/bindings.rs \
    -- -I"$IPOPT_INCLUDE"
#!/usr/bin/env bash
# Verify the PyTorch reference is unchanged.
#
# model/ and tests/ are the PARITY ORACLE for the Rust port. They are frozen at
# the upstream state (commit 67b630e) so that every parity number we quote is
# measured against immutable ground truth. Bugs found in this code are reported
# in PORTING_FROM_PYTHON.md, not patched here.
#
# Exit 0 = oracle intact. Exit 1 = someone edited the reference.
set -euo pipefail

cd "$(dirname "$0")/.."

if sha256sum --check --quiet parity/frozen.lock; then
    echo "OK: model/ and tests/ match parity/frozen.lock (oracle intact)"
    exit 0
fi

echo "" >&2
echo "FAIL: the PyTorch parity oracle has been modified." >&2
echo "      model/ and tests/ must stay byte-identical to upstream." >&2
echo "      If you believe a change is required, it is a decision for the" >&2
echo "      repo owner, not an incidental edit - see PORTING_FROM_PYTHON.md S8." >&2
echo "" >&2
exit 1
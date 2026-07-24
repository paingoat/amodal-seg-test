#!/usr/bin/env bash
# Add NVIDIA CUDA apt repo (Ubuntu 24.04/22.04) and install toolkit 13.2 only.
# Does NOT install/replace GPU drivers (safe for machines that already have a working driver).
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo:"
  echo "  sudo bash scripts/install_cuda_toolkit_13_2.sh"
  exit 1
fi

. /etc/os-release
case "${VERSION_ID}" in
  24.04) DISTRO=ubuntu2404 ;;
  22.04) DISTRO=ubuntu2204 ;;
  *)
    echo "Unsupported Ubuntu VERSION_ID=${VERSION_ID}. Supported: 22.04 / 24.04."
    exit 1
    ;;
esac

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) REPO_ARCH=x86_64 ;;
  aarch64|arm64) REPO_ARCH=sbsa ;;
  *)
    echo "Unsupported arch: $ARCH"
    exit 1
    ;;
esac

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${REPO_ARCH}/cuda-keyring_1.1-1_all.deb"
echo "[install_cuda_toolkit] distro=$DISTRO arch=$REPO_ARCH"
echo "[install_cuda_toolkit] fetching $KEYRING_URL"
wget -q --show-progress -O cuda-keyring.deb "$KEYRING_URL"
dpkg -i cuda-keyring.deb
apt update
apt install -y cuda-toolkit-13-2

echo
echo "[install_cuda_toolkit] Done."
echo "Verify:"
echo "  /usr/local/cuda-13.2/bin/nvcc --version"
echo "Then (as your user, in env amodal):"
echo "  export CUDA_HOME=/usr/local/cuda-13.2"
echo "  export PATH=\"\$CUDA_HOME/bin:\$PATH\""
echo "  bash scripts/fix_groundingdino_ops.sh"
echo
echo "Quick workaround without toolkit (slower attention):"
echo "  python scripts/patch_groundingdino_pytorch_attn.py"
echo "  export GDINO_ALLOW_PYTORCH_ATTN=1"
echo "  bash scripts/start_gradio.sh"

#!/usr/bin/env bash
set -euo pipefail

# ====== 可改变量 ======
IMAGE="${IMAGE:-isaaclab:4.5-jammy}"
CONTAINER="${CONTAINER:-isaaclab-gui}"

# 主机侧目录
HOST_HOME="${HOME}"
HOST_CACHE_OV="${HOME}/.cache/ov"
HOST_CACHE_PIP="${HOME}/.cache/pip"
HOST_OMNI_LOGS="${HOME}/.nvidia-omniverse/logs"

# Xauthority 文件（自动生成）
XAUTH_FILE="${HOME}/.docker.xauth"

# ====== 前置检查 ======
if ! command -v docker >/dev/null 2>&1; then
  echo "✖ 未找到 docker 命令，请先安装 Docker。" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "⚠ 未检测到 nvidia-smi，确保已安装 NVIDIA 驱动并启用 NVIDIA Container Toolkit。" >&2
fi

mkdir -p "${HOST_CACHE_OV}" "${HOST_CACHE_PIP}" "${HOST_OMNI_LOGS}"

# 生成一个可映射进容器的 Xauthority（避免 Permission 问题）
touch "${XAUTH_FILE}"
if command -v xauth >/dev/null 2>&1; then
  xauth nlist "${DISPLAY:-:0}" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "${XAUTH_FILE}" nmerge - 2>/dev/null || true
fi
chmod 600 "${XAUTH_FILE}" || true

# 放宽本地 X11 访问（仅本机进程）
if command -v xhost >/dev/null 2>&1; then
  xhost +si:localuser:root >/dev/null 2>&1 || true
  xhost +si:localuser:"$(id -un)" >/dev/null 2>&1 || true
fi

# ====== 已有容器则直接启动 ======
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "▶ 启动已存在的容器：${CONTAINER}"
  exec docker start -ai "${CONTAINER}"
fi

echo "▶ 创建并启动新容器：${CONTAINER}"

# ====== 首次创建并进入 ======
exec docker run -it \
  --name "${CONTAINER}" \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=8g \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=/home/dev/.Xauthority \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -e OMNI_CACHE_ROOT=/home/dev/.cache/ov \
  -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  -e VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "${XAUTH_FILE}:/home/dev/.Xauthority:ro" \
  -v "${HOST_HOME}:/workspace/home:rw" \
  -v "${HOST_CACHE_OV}:/home/dev/.cache/ov:rw" \
  -v "${HOST_CACHE_PIP}:/home/dev/.cache/pip:rw" \
  -v "${HOST_OMNI_LOGS}:/home/dev/.nvidia-omniverse/logs:rw" \
  --user root \
  "${IMAGE}" \
  bash -lc '
    set -e
    echo ">>> APT 更新并安装依赖（强制 IPv4）"
    apt-get update -o Acquire::ForceIPv4=true
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libvulkan1 vulkan-tools libglu1-mesa libxkbcommon-x11-0 \
      libxcb-cursor0 libsm6 libxt6 libice6 fontconfig zenity \
      && rm -rf /var/lib/apt/lists/*

    echo ">>> 持久化 Vulkan 环境变量（/etc/profile.d/99-nvidia-vulkan.sh）"
    cat >/etc/profile.d/99-nvidia-vulkan.sh <<EOF
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
EOF
    chmod 644 /etc/profile.d/99-nvidia-vulkan.sh

    echo ">>> dev 用户自动激活 conda 环境"
    if [ -f /home/dev/.bashrc ] && ! grep -q "conda activate env_isaaclab" /home/dev/.bashrc; then
      echo "conda activate env_isaaclab" >> /home/dev/.bashrc
    fi
    chown dev:dev /home/dev/.bashrc || true

    echo ">>> 简单自检：vulkaninfo 前几行（如失败不影响进入）"
    vulkaninfo | head -n 20 || true

    echo ">>> 切换到 dev 用户"
    exec su - dev
  '


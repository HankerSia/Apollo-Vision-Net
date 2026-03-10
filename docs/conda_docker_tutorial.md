# 基于 Conda 环境生成 Docker 镜像的工程实践教程

## 1. 为什么要用 Conda-Pack 方案？

在深度学习项目（如 BEVFormer, MapTR）中，环境构建通常面临三个核心痛点：
1. **网络不稳定性**：在 Dockerfile 内部运行 `pip install` 或 `conda install` 时，由于 PyTorch、CUDA 等包体积巨大，且需要连接海外服务器，极易因超时导致构建失败。
2. **源码依赖复杂（Editable Packages）**：很多算法库（如 `mmdet3d`）是以 `pip install -e .` 形式安装在本地开发环境的。这种“可编辑”模式在 Docker 构建时无法直接打包物理文件。
3. **环境一致性**：即便脚本相同，不同时间构建得到的子依赖版本可能存在微小差异，导致“本地能跑镜像里崩”的问题。

**本教程采用“本地打包（Conda-Pack）+ 镜像解压缩”的方案**，实现了环境的快速物理迁移，确保镜像内外环境 100% 对齐。

---

## 2. 核心迁移路线图

1. **宿主机预处理**：将 `pip -e` 类型依赖转换为常规安装。
2. **打包环境**：使用 `conda-pack` 将环境压缩为自包含的 `.tar.gz`。
3. **镜像构建**：在 Dockerfile 中解压缩环境，并运行 `conda-unpack` 重新修正二进制路径。
4. **运行部署**：通过 `entrypoint.sh` 脚本动态注入 `PYTHONPATH`，实现代码挂载即运行。

---

## 3. 详细实操步骤

### 第一步：处理“可编辑（Editable）”依赖包
`conda-pack` 无法打包软链接形式的源码包。如果你的环境中有包（如 `mmdet3d`）指向宿主机其它目录，请先进行“静态化”：

```bash
# 激活目标环境
conda activate apollo_vnet

# 1. 卸载软链接版
python -m pip uninstall -y mmdet3d

# 2. 彻底安装进 site-packages（不带 -e）
# 注意：--no-deps 确保不触发重复下载依赖，--no-build-isolation 加速本地编译
python -m pip install --no-deps --no-build-isolation /path/to/your/mmdetection3d
```

### 第二步：编写打包脚本 `pack_env.sh`
该脚本确保打包逻辑幂等，并正确忽略无法打包的特殊项。

```bash
#!/usr/bin/env bash
# 文件位置: docker/pack_env.sh
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p docker
rm -f docker/apollo_vnet.tar.gz

# 激活环境
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate apollo_vnet

echo ">>> 正在打包 Conda 环境 'apollo_vnet'..."
# conda-pack 会包含 Python 解释器及其所有静态依赖
conda-pack -n apollo_vnet --ignore-editable-packages -o docker/apollo_vnet.tar.gz
echo ">>> 打包完成: docker/apollo_vnet.tar.gz"
```

### 第三步：编写高效 Dockerfile
规避常见 404 错误：不再镜像内动态下载 Miniconda 脚本，直接解压已打包的环境。

```dockerfile
# 文件位置: docker/Dockerfile.apollo_vnet

# 注入 ARG 以便在 build 时切换镜像源
ARG BASE_IMAGE=nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

# 1. 安装基础系统依赖（OpenCV/Git 等开发必需品）
RUN apt-get update && apt-get install -y --no-install-recommends \
        bzip2 ca-certificates bash git libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 环境解压路径
ENV APOLLO_VNET_DIR=/opt/apollo_vnet

# 3. 核心：复制并解压 Conda 环境
COPY docker/apollo_vnet.tar.gz /tmp/apollo_vnet.tar.gz
RUN mkdir -p "${APOLLO_VNET_DIR}" \
    && tar -xzf /tmp/apollo_vnet.tar.gz -C "${APOLLO_VNET_DIR}" \
    && rm -f /tmp/apollo_vnet.tar.gz \
    && "${APOLLO_VNET_DIR}/bin/python" "${APOLLO_VNET_DIR}/bin/conda-unpack"

# 4. 设置环境变量
ENV PATH=${APOLLO_VNET_DIR}/bin:${PATH}
WORKDIR /workspace/Apollo-Vision-Net

# 5. 入口逻辑
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
```

### 第四步：编写容器入口脚本 `entrypoint.sh`
该脚本用于在容器启动瞬间，动态注入当前代码目录的搜索路径。

```bash
#!/usr/bin/env bash
# 文件位置: docker/entrypoint.sh
set -e

# 让挂载进来的代码目录立即可被 Python import
export PYTHONPATH="/workspace/Apollo-Vision-Net:${PYTHONPATH:-}"

# 执行 CMD 传入的原始命令
exec "$@"
```

### 第五步：自动化构建与运行脚本
为了兼容需要 `sudo` 的机器以及国内镜像站，采用了环境变量驱动的设计。

**构建脚本 `build.sh`：**
```bash
#!/usr/bin/env bash
# docker/build.sh
set -euo pipefail

# 默认基础镜像。若 Docker Hub 无法连接，可传入华为/阿里云镜像：
# BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04}"

if [[ "${DOCKER_SUDO:-0}" == "1" ]]; then
  DOCKER_CMD=(sudo docker)
else
  DOCKER_CMD=(docker)
fi

"${DOCKER_CMD[@]}" build --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -f docker/Dockerfile.apollo_vnet -t apollo-vnet:cu113 .
```

**运行脚本 `run.sh`：**
```bash
#!/usr/bin/env bash
# docker/run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${DOCKER_SUDO:-0}" == "1" ]]; then
  DOCKER_CMD=(sudo docker)
else
  DOCKER_CMD=(docker)
fi

DOCKER_RUN_ARGS=(
  run --rm -it --gpus all --shm-size=8g
  -v "${REPO_ROOT}:/workspace/Apollo-Vision-Net"
  -w /workspace/Apollo-Vision-Net
  apollo-vnet:cu113
)

# 执行命令
if [[ "$#" -gt 0 ]]; then
  "${DOCKER_CMD[@]}" "${DOCKER_RUN_ARGS[@]}" "$@"
else
  "${DOCKER_CMD[@]}" "${DOCKER_RUN_ARGS[@]}" bash
fi
```

---

## 4. 关键避坑指南（Must Read）

### 1. 解决 Docker Hub 连接超时
如果 `docker build` 一直卡在 Step 1 拉取基础镜像，且确认本机网络受限：
- 提前通过国内镜像源手动 `docker pull`。
- 构建时通过环境变量指定该 tag：`BASE_IMAGE=... ./docker/build.sh`。

### 2. 避免 Context 爆炸
如果你的工程根目录下有 `data/`（如几百 GB 的 Nuscenes），`docker build` 会在第一步卡死（Sending build context）。
**必须创建 `.dockerignore` 文件：**
```text
# .dockerignore
*
!docker/  # 只允许发送 docker 脚本和 3GB 的 tar 包给 daemon
```

### 3. GPU 驱动映射
容器启动后，务必执行以下自检确认 GPU 库映射正确：
```bash
python -c "import torch; print(torch.cuda.is_available())"
# 预期输出: True
```

---

## 5. 跨服务器重新运行流程

完成上述构建后，你可以将镜像迁移到任何有 GPU 驱动的服务器：
1. **导出镜像**：`sudo docker save apollo-vnet:cu113 -o apollo_vnet.tar`
2. **同步代码**：将整个工程 `Apollo-Vision-Net` 源码拷贝至服务器。
3. **导入并运行**：
   - `sudo docker load -i apollo_vnet.tar`
   - `sudo ./docker/run.sh  # 脚本会自动把代码挂载进容器`

这种方法能够完美保持实验的可复现性，无需在服务器上重新配置任何环境。

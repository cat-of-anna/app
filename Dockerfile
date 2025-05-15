# 步骤 1: 选择一个基础镜像 
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# 步骤 2: 设置工作目录 (容器内后续操作的默认路径)
WORKDIR /app

# 步骤 3: 复制依赖管理文件
COPY requirements.txt .

# 步骤 4: 安装应用依赖
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 步骤 5: 复制你的项目代码到镜像中
COPY . . 
# 步骤 6：启动
RUN chmod +x run.sh
RUN chmod +x train.sh
CMD ["bash", "run.sh"]
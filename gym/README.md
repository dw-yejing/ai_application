```bash
# 依赖环境
conda create --name gym python=3.9

# gym 0.21 installation is broken with more recent versions, so specify the following version
pip install setuptools==65.5.0 pip==21  
pip install wheel==0.38.0


# 依赖安装
pip install gym==0.21.0 torch==2.5.1 numpy==1.23.1 opencv-python tensorboard pyglet==1.5.27
```

# conda

[toc]

## 换源

vim ~/.condarc

```bash
channels:
  - https://mirrors.aliyun.com/anaconda/pkgs/free/
  - https://mirrors.aliyun.com/anaconda/cloud/conda-forge/
  - https://mirrors.aliyun.com/anaconda/pkgs/main/
show_channel_urls: true
default_channels:
  - https://mirrors.aliyun.com/anaconda/pkgs/main
  - https://mirrors.aliyun.com/anaconda/pkgs/r
  - https://mirrors.aliyun.com/anaconda/pkgs/msys2
custom_channels:
  conda-forge: http://mirrors.aliyun.com/anaconda/cloud
  msys2: http://mirrors.aliyun.com/anaconda/cloud
  bioconda: http://mirrors.aliyun.com/anaconda/cloud
  menpo: http://mirrors.aliyun.com/anaconda/cloud
  pytorch: http://mirrors.aliyun.com/anaconda/cloud
  simpleitk: http://mirrors.aliyun.com/anaconda/cloud
auto_activate_base: false
```

## 使用

```bash
 conda create -n B --clone A       #克隆环境A来创建名为B的环境
 conda activate xxxx               #开启xxxx环境
 conda deactivate                  #关闭环境
 conda info -e                    #显示所有的虚拟环境
 conda remove -n xxxx --all       #删除已创建的xxxx虚拟环境
 
 conda update --all
 
 conda clean -p      #删除没有用的包
 conda clean -t      #tar打包
 conda clean -a
 
 conda config --show   #查看全部配置
```
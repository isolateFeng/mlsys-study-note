# linux 终端基础操作

- mkdir
    - mkdir 文件名 : 在当前目录下新建 “文件名”的文件夹

- cd
    - cd 文件名 : 进入当前目录下 “文件名”的文件夹，

- pwd
    - pwd : 显示当前位置

- ls / ll
    - ls : 显示当前目录下的文件
    - ll : 显示当前目录下的文件，包括隐藏文件

- touch
    - touch 文件名 : 在当前目录下新建 “文件名”的文件

- cp
    - cp 源文件 目标文件（夹） : 复制文件或文件夹
    - cp -r 源文件 目标文件（夹） : 复制文件夹

- mv
    - mv 源文件 目标文件（夹） : 移动文件或文件夹
    - mv 源文件 目标文件（夹） : 重命名文件或文件夹

- rm
    - rm 文件名 : 删除文件
    - rm -rf 文件名 : 删除文件夹

- tar
    - tar -zxvf 文件名.tar.gz : 解压tar.gz
    - tar -zcvf 文件名.tar.gz 文件名 : 压缩文件

- unzip
    - unzip 文件名.zip : 解压zip

- du
    - du -ah --max-depth=1 : 显示当前目录下各个文件占据内存

- df
    - df -h : 显示磁盘使用情况

- cat
    - cat 文件名 : 查看文件内容

- vim / vi
    - vim 文件名 : 编辑文件

- tree
    - tree : 显示目录树

- grep
    - grep -rni "关键词" 文件名 : 在文件中搜索关键词

- find
    - find . -name "文件名" : 在当前目录下查找文件
    - find . -name "文件名" -exec rm -rf {} \; : 删除查找到的文件

- ps 
    - ps -ef | grep "进程名" : 查看进程
    - kill -9 进程号 : 杀死进程

- jobs
    - jobs : 查看后台运行的任务
    - fg %n : 将后台任务调到前台运行
    - bg %n : 将前台任务调到后台运行

- scp
    - scp -r 文件名 用户名@IP地址:目标路径 : 上传文件
    - scp -r 用户名@IP地址:文件名 目标路径 : 下载文件

- ssh
    - ssh 用户名@IP地址 : 远程登录

- wget / curl
    - wget / curl 下载链接 : 下载文件

- apt
    - apt update : 更新软件源
    - apt upgrade : 更新软件
    - apt install 软件名 : 安装软件
    - apt remove 软件名 : 卸载软件

- sh
    - sh 文件名 : 运行脚本

- chmod
    - chmod 777 文件名 : 修改文件权限

- chown
    - chown 用户名 文件名 : 修改文件所有者

- ln
    - ln -s 源文件 目标文件 : 创建软链接

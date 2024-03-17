# Vim

[toc]

## 配置

vim ~/.vimrc

source ~/.vimrc

```bash

set wildmenu"按TAB键时命令行自动补齐"
set ignorecase"忽略大小写"
set number "显示行号"
set ruler"显示当前光标位置"
set autoread"文件在Vim之外修改过，自动重新读入"
set autowrite"设置自动保存内容"
set autochdir"当前目录随着被编辑文件的改变而改变"
set cindent "c/c++自动缩进"
set smartindent
set autoindent"参考上一行的缩进方式进行自动缩进"
set softtabstop=4 "4 character as a tab"
set shiftwidth=4
set smarttab
set hlsearch "开启搜索结果的高亮显示"
set incsearch "边输入边搜索(实时搜索)"

```

## 操作

- 模式切换
    - 正常为命令模式
    - 按i进入编辑模式，按esc退出编辑模式

- 复制
    - 在命令模式下，将光标移动到将要复制的行处，按 `yy` 进行复制；
    - 按 `nyy` 复制n行；其中n为1、2、3……

- 粘贴
    - 按 `p` 进行粘贴

- 删除
    - 按 `d` 后按数字，其中数字表示删除的行数

- 撤回
    - 撤回上一步操作：按 `u`
    - 撤回多步操作：按 `U`

- 查找
    - 按 `/` 进入查找模式，输入关键词，按 `n` 查找下一个，按 `N` 查找上一个

- 替换
    - 按 `:%s/old/new/g` 进行替换
    - `g` 表示全局替换

- 在vim中比较两个文件的不同
    - `vimdiff a.file b.file`
    - 使用 `crtl + w` + `w` 进行切换左右侧

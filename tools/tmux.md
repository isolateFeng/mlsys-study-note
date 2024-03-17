# tmux

用来保护正在进行的任务进程

[toc]

## 配置

vim ~/.tmux.config
tmux source ~/.tmux.config

```bash
## ====================== 将以下内容输入

#set -g prefix C-z # 修改 默认的ctrl-b 组合键为 ctrl-z

bind | split-window -h # ctrl-b + | 左右分屏
bind - split-window -v # ctrl-b + - 上下分屏

# 开启鼠标切换tmux窗口
setw -g mouse-resize-pane on
setw -g mouse-select-pane on
setw -g mouse-select-window on
setw -g mode-mouse on

set -g base-index         1     # 窗口编号从 1 开始计数
set -g pane-base-index    1     # 窗格编号从 1 开始计数
set -g renumber-windows   on    # 关掉某个窗口后，编号重排
setw -g allow-rename      off   # 禁止活动进程修改窗口名
setw -g automatic-rename  off   # 禁止自动命名新窗口

set -g status-right '#{prefix_highlight} #H | %a %Y-%m-%d %H:%M'
set -g @prefix_highlight_show_copy_mode 'on'
set -g @prefix_highlight_copy_mode_attr 'fg=white,bg=blue'
## ====================== :wq! 保存退出
```

## 操作

```bash
tmux         # 开启一个窗口
exit         # 销毁/关闭该窗口
tmux detach  # 将当前会话与窗口分离，跑长时间记得使用（快捷键 按下ctrl-b松手 再按 d）
tmux attach -t <session-name> # 例如 tmux aatch -t 0

tmux ls      # 查看当前所有的 Tmux 会话（快捷键ctrl-b + s）
tmux kill-session -t 0        # 命令用于杀死某个会话，数字0是编号

tmux split-window  # 划分上下两个窗格 ctrl-b + -
tmux split-window -h   # 划分左右两个窗格 ctrl-b + |

# 快捷键Ctrl+b <arrow key>：光标切换到其他窗格。ctrl-b + 上下左右
tmux select-pane -U  # 光标切换到上方窗格
tmux select-pane -D  # 光标切换到下方窗格
tmux select-pane -L  # 光标切换到左边窗格
tmux select-pane -R  # 光标切换到右边窗格
```

更多见：https://www.ruanyifeng.com/blog/2019/10/tmux.html
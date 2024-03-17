# mac 配置

[toc]

## homebrew

参考：https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/

安装需求：

- 对于 macOS 用户，系统自带 bash、git 和 curl，在命令行输入 `xcode-select --install` 安装 CLT for Xcode 即可。

- 对于 Linux 用户，系统自带 bash，仅需额外安装 git 和 curl。

安装 Homebrew / Linuxbrew：

```bash
# 从本镜像下载安装脚本并安装 Homebrew / Linuxbrew
git clone --depth=1 https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/install.git brew-install
/bin/bash brew-install/install.sh
rm -rf brew-install

# 也可从 GitHub 获取官方安装脚本安装 Homebrew / Linuxbrew
/bin/bash -c "$(curl -fsSL https://github.com/Homebrew/install/raw/master/install.sh)"
```

加入环境变量

```bash
#以下针对基于 Apple Silicon CPU 设备上的 macOS 系统（命令行运行 uname -m 应输出 arm64）上的 Homebrew：
test -r ~/.bash_profile && echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.bash_profile
test -r ~/.zprofile && echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile

#对基于 Intel CPU 设备上的 macOS 系统（命令行运行 uname -m 应输出 x86_64）的用户可跳过本步。

#以下针对 Linux 系统上的 Linuxbrew：
test -d ~/.linuxbrew && eval "$(~/.linuxbrew/bin/brew shellenv)"
test -d /home/linuxbrew/.linuxbrew && eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
test -r ~/.bash_profile && echo "eval \"\$($(brew --prefix)/bin/brew shellenv)\"" >> ~/.bash_profile
test -r ~/.profile && echo "eval \"\$($(brew --prefix)/bin/brew shellenv)\"" >> ~/.profile
test -r ~/.zprofile && echo "eval \"\$($(brew --prefix)/bin/brew shellenv)\"" >> ~/.zprofile
```

换源:

```bash
export HOMEBREW_INSTALL_FROM_API=1
export HOMEBREW_API_DOMAIN="https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles/api"
export HOMEBREW_BOTTLE_DOMAIN="https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles"
export HOMEBREW_BREW_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/brew.git"
export HOMEBREW_CORE_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/homebrew-core.git"
brew update
```

配置好后再使用 `brew install` 安装软件

```bash
brew cask install google-chrome
brew install miniconda
```

## on my zsh

安装zsh `brew install zsh zsh-completions`

切换到zsh `[sudo] chsh -s $(which zsh)`

安装oh-my-zsh

```bash
git clone git://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
```

修改主题，在 `~/.zshrc` 里的 设置`ZSH_THEME="ys"`

安装插件
常用autojump、zsh-autosuggestions、zsh-syntax-highlighting三个插件 
```bash
cd ~/.oh-my-zsh/plugins
brew install autojump
git clone git clone https://github.com/zsh-users/zsh-syntax-highlighting.git
git clone https://github.com/zsh-users/zsh-autosuggestions.git
```
然后在 `~/.zshrc` 找到 `plugins=` 添加下面的，最后保存执行 `source ~/.zshrc`

```bash
plugins=(
  autojump
  git zsh-autosuggestions
  git zsh-syntax-highlighting
)
```

## fzf

用来增强搜索 `ctrl + r` /  `command + r`

```bash
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install
```

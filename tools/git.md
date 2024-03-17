# git

[toc]

## 显示当前分支

```
vim .bashrc
```

将下面的代码加入到文件的最后处

``` bash
 function git_branch {
    branch="`git branch 2>/dev/null | grep "^\*" | sed -e "s/^\*\ //"`"
    if [ "${branch}" != "" ];then
        if [ "${branch}" = "(no branch)" ];then
            branch="(`git rev-parse --short HEAD`...)"
        fi
        echo " ($branch)"
    fi
 }
 
 export PS1='\u@\h \[\033[01;36m\]\W\[\033[01;32m\]$(git_branch)\[\033[00m\] \$ '
```

保存退出，执行加载命令

```
 source ~/.bashrc
```

## 常用操作

```bash
git clone xxx
git submodule update --init --recursive

# 查看远程分支
git branch -a
 
# 查看本地分支
git branch
 
# 切换分支 tx-84
git checkout tx-84
# 检查子模块是否版本对齐
git status
git submodule update --init
 
 
# 获取master最新代码
git checkout master
git pull # 将远程主机的最新内容拉到本地，用户在检查了以后决定是否合并到工作本机分支中，这样master的最新代码就在origin/master
git fetch # git pull = git fetch + git merge
 
# 基于master创建分支
git checkout -b myfeature
 
# 合并commit
# 分支开发完成后，很可能有一堆commit，但是合并到主干的时候只希望有少量commit
git reset HEAD~5
git add xxx
git commit -m "Here's the bug fix that closes #28"
 
# 推到远程仓库
git push origin myfeature -f
 
# rebase代码
git checkout myfeature
git rebase origin/master
git push origin myfeature -f
 
# 查看commit更改
git show
git show --name-only
git show <commit-hash>
git log --oneline # 查看最近的提交
 
# 例如，把 e57b0e6 合并到 17cb931，不保留注释
pick 17cb931 fix && add batch del
f e57b0e6 fix && add batch del

# 指定需要合并版本号，处理从该版本后往后的commit，不包含该版本，会进入vi编辑器
git rebase -i 版本号
git commit -n --amend  # 或者使用这个命令将其合并进上一个commit

# 使用别人的patch测试，记得先fetch
git cherry-pick commit_id
```
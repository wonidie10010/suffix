# GitHub 项目更新流程

## 1. 基本原则

- 先检查改动，再提交。
- 每次只提交当前任务相关文件。
- 不要默认使用 `git add .`。
- 删除文件、配置修改和新增文件必须重点检查。
- 提交前运行与本次修改直接相关的测试。
- 未经用户明确要求，不执行 `git commit` 和 `git push`。
- 禁止使用 `git push --force`、`git reset --hard` 和 `git clean -fd`。
- 不得提交密码、Token、私钥、模型权重、缓存、日志或大型临时输出。

## 2. 标准流程

### 第一步：检查仓库状态

```bash
git status
git branch --show-current
git remote -v
```

确认当前分支和远程仓库正确。发现仓库、分支或远程地址异常时，停止操作并报告。

### 第二步：检查并同步远程更新

```bash
git fetch origin
git status
```

如果远程存在本地尚未同步的提交，再执行：

```bash
git pull --rebase origin main
```

发生冲突时立即停止，使用 `git status` 列出冲突文件并报告，不得擅自覆盖代码。

### 第三步：检查本地改动

```bash
git status
git diff --stat
git diff
```

重点检查：

- 是否误删文件；
- 是否加入无关文件；
- 是否加入重复目录；
- 是否意外修改配置；
- 是否包含敏感信息；
- 是否包含缓存、日志、实验输出或模型权重。

### 第四步：运行针对性测试

根据本次修改选择最相关的测试或静态检查，不要求无针对性地运行全部测试。

示例：

```bash
python -m pytest test/相关测试文件.py -q
python -m py_compile 路径/文件.py
python -m json.tool 路径/配置文件.json
```

测试失败、测试跳过或缺少依赖时，必须如实报告。

### 第五步：暂存相关文件

优先明确指定文件：

```bash
git add 文件1 文件2
```

删除文件也可以使用：

```bash
git add 已删除文件
```

暂存后检查：

```bash
git status
git diff --cached --stat
git diff --cached
```

确认暂存区中只有本次任务相关改动。只有确认所有改动都属于当前任务时，才允许使用 `git add .`。

### 第六步：提交

仅在用户明确要求提交时执行：

```bash
git commit -m "类型: 修改说明"
```

常用提交类型：

- `feat`：新增功能
- `fix`：修复问题
- `docs`：文档修改
- `test`：测试修改
- `config`：配置修改
- `refactor`：代码重构
- `chore`：其他维护

提交信息必须描述实际改动，避免只写 `update`、`修改` 或 `fix`。

提交后检查：

```bash
git status
git log -1 --oneline
```

### 第七步：推送

仅在用户明确要求推送时执行：

```bash
git push origin main
```

推送失败时不得强制推送，应先检查远程更新和冲突。

### 第八步：最终检查

```bash
git status
git log -1 --oneline
```

汇报以下内容：

- 修改了哪些文件；
- 测试或检查结果；
- 提交编号；
- 推送结果；
- 是否还有未提交改动。

## 3. Codex 执行边界

当用户只要求检查代码、审查改动或准备提交时，必须停止在 `git commit` 之前。

遇到以下情况必须停止并报告：

- Git 冲突；
- 测试失败；
- 远程存在未知提交；
- 异常删除文件；
- 发现敏感信息；
- 发现大型文件或模型权重；
- GitHub 权限或认证失败。

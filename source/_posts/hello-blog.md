---
title: 第一篇：这个博客怎么用
date: 2026-11-06 10:00:00
categories:
  - 杂记
tags:
  - hexo
  - 工具
---

这是一篇示例文章，演示这个博客支持的写作能力：多级标题、代码高亮、表格。

<!-- more -->

## 1. 写一篇新文章

```bash
npx hexo new "文章标题"
```

会在 `source/_posts/` 下生成 `文章标题.md`，同时因为开启了 `post_asset_folder`，还会生成同名目录用来放图片，文章里用 `![说明](图片名.png)` 引用即可。

### 1.1 Front-matter 字段

| 字段 | 说明 |
| --- | --- |
| `title` | 标题，同时决定 URL 中的 slug |
| `date` | 发布时间，决定永久链接的年月日 |
| `categories` | 分类，建议只写一个层级 |
| `tags` | 标签，可以多个 |

### 1.2 摘要

在正文中插入 `<!-- more -->`，首页只展示它之前的内容。

## 2. 排版能力

### 2.1 代码块

支持行号、折叠和一键复制：

```cpp
template <typename T>
class HashTable {
 public:
  bool insert(const T& key) {
    auto slot = probe(hash(key));
    if (slots_[slot].occupied) {
      return false;
    }
    slots_[slot] = {key, /*occupied=*/true};
    ++size_;
    return true;
  }

 private:
  std::vector<Slot> slots_;
  size_t size_{0};
};
```

## 3. 本地预览与发布

### 3.1 预览

```bash
npx hexo clean && npx hexo server
```

打开 http://localhost:4000 。

### 3.2 发布

推到 `main` 分支后，GitHub Actions 会自动构建并部署到 GitHub Pages。

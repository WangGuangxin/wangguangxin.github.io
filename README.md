# Blog

Hexo + NexT(Gemini) 博客，部署到 https://wangguangxin.github.io 。

## 常用命令

```bash
npm run new -- "文章标题"   # 新建文章到 source/_posts/
npm run server              # 本地预览 http://localhost:4000
npm run clean && npm run build
```

## 发布

推送到 `main` 分支，`.github/workflows/pages.yml` 会自动构建并部署到 GitHub Pages。

仓库 Settings → Pages → Build and deployment → Source 需要选 **GitHub Actions**。

## 目录

- `source/_posts/` — 文章；开启了 `post_asset_folder`，同名目录放图片
- `source/about|categories|tags/` — 独立页面
- `_config.yml` — 站点配置（标题、URL、永久链接）
- `_config.next.yml` — 主题配置（菜单、侧边栏、目录编号、代码块）

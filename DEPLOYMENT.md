---
layout: default
title: Deployment Guide
---

# 🚀 Deployment Guide

This guide explains how to deploy the AIML Course Hub website to GitHub Pages.

---

## 📋 Prerequisites

- GitHub account (with push access)
- Git installed locally
- The repository cloned to your machine

---

## 🔄 Deployment Steps

### Step 1: Commit All Changes

<div class="alert alert-note">
  <h4>📌 Note: Clean Commits</h4>
  <p>Always commit and push your changes before deploying. Make sure the main branch is in a good state with no uncommitted changes.</p>
</div>

```bash
cd /Users/shivamkumar/semester_1_all_course

# Check status
git status

# Stage all changes
git add .

# Commit with meaningful message
git commit -m "Add comprehensive course materials and deployment"

# Push to main branch
git push origin main
```

### Step 2: Configure GitHub Pages

**Method 1: Using gh-pages Branch (Recommended)**

```bash
# Create gh-pages branch if it doesn't exist
git branch gh-pages

# Push to gh-pages
git push origin gh-pages

# In GitHub repository settings:
# Settings → Pages → Branch: gh-pages
```

**Method 2: Using main Branch with docs Folder**

```bash
# Build site to docs folder
bundle exec jekyll build --destination docs

# Commit and push
git add docs/
git commit -m "Build site for GitHub Pages"
git push origin main

# In GitHub repository settings:
# Settings → Pages → Source: main branch /docs folder
```

**Method 3: Automated with GitHub Actions (Recommended)**

Create `.github/workflows/jekyll.yml`:

```yaml
name: Build and Deploy Jekyll

on:
  push:
    branches:
      - main

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: 3.1
          bundler-cache: true
      
      - name: Build site
        run: bundle exec jekyll build
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./_site
```

### Step 3: Verify Repository Settings

Go to your GitHub repository and configure:

1. **Settings → Pages**
   - Source: Select appropriate branch (main, gh-pages, or main/docs)
   - Choose theme (optional)
   - Save

2. **Settings → Branches**
   - Set default branch (usually main)
   - Enable branch protection if desired

3. **Settings → Secrets and variables** (if using GitHub Actions)
   - Add GitHub token if needed

### Step 4: Monitor Build

<div class="alert alert-info">
  <h4>ℹ️ Build Status</h4>
  <p>GitHub will automatically build your site. Check Actions tab to see build status.</p>
</div>

```
Repository → Actions → workflow runs
```

---

## 🌐 After Deployment

### Verify Website is Live

Visit your site at:
```
https://shivam2003-dev.github.io/semester_1_all_course
```

<div class="alert alert-success">
  <h4>✅ Success Indicators</h4>
  <ul>
    <li>Site loads without errors</li>
    <li>All pages accessible</li>
    <li>CSS and styling applied</li>
    <li>Images and resources load</li>
    <li>Navigation works</li>
  </ul>
</div>

### Check Site Functionality

- [ ] Homepage loads correctly
- [ ] All course pages accessible
- [ ] Links to external resources work
- [ ] Navigation between pages works
- [ ] Mobile view responsive
- [ ] Search functionality (if implemented)

---

## 🔧 Troubleshooting

### Issue: Site Not Building

<div class="alert alert-warning">
  <h4>⚠️ Common Causes</h4>
  <ul>
    <li>Syntax errors in YAML frontmatter</li>
    <li>Invalid Markdown</li>
    <li>Plugin conflicts</li>
    <li>Branch not configured in settings</li>
  </ul>
</div>

**Solution:**
1. Check build logs in GitHub Actions
2. Validate YAML frontmatter
3. Test locally: `bundle exec jekyll build`
4. Ensure correct branch is selected in Pages settings

### Issue: Site Shows 404

**Possible Causes:**
- Wrong domain/URL
- Pages not enabled in settings
- Build failed silently

**Solution:**
1. Verify repository name in `_config.yml`: `baseurl`
2. Check GitHub Pages settings
3. Wait 1-2 minutes after push
4. Hard refresh browser (Ctrl+Shift+R)

### Issue: CSS/Styling Not Applied

**Possible Causes:**
- Incorrect baseurl in `_config.yml`
- CSS not built
- Asset paths wrong

**Solution:**
```bash
# Ensure _config.yml has correct baseurl
baseurl: "/semester_1_all_course"

# Rebuild locally
bundle exec jekyll clean
bundle exec jekyll build

# Push changes
git add . && git commit -m "Fix styling" && git push
```

### Issue: Images Not Loading

**Solution:**
1. Check image paths are relative
2. Ensure images in `assets/images/` folder
3. Use correct URL: `{{ site.baseurl }}/assets/images/filename.jpg`

---

## 📝 Maintenance

### Regular Updates

After deploying, keep the site updated:

```bash
# 1. Make content changes
# 2. Commit changes
git add .
git commit -m "Update course materials"

# 3. Push to repository
git push origin main

# 4. GitHub Pages auto-deploys (wait 1-2 minutes)
```

### Testing Changes Locally

Before pushing to production:

```bash
# Build and serve locally
bundle exec jekyll serve

# Visit http://localhost:4000/semester_1_all_course
# Test all pages and functionality
```

### Version Control Best Practices

```bash
# Create feature branches for major changes
git checkout -b feature/new-course

# Make changes and test
# ...

# Commit changes
git add .
git commit -m "Add new course content"

# Push feature branch
git push origin feature/new-course

# Create Pull Request on GitHub
# After review, merge to main
```

---

## 🔒 Security

### GitHub Pages Security Features

- HTTPS enabled by default
- CNAME records supported
- DNS verification

### Best Practices

1. **Keep Dependencies Updated**
```bash
bundle update
```

2. **No Sensitive Data**
- Never commit API keys, passwords, tokens
- Use `.gitignore` for secrets

3. **Branch Protection**
- Enable in Settings → Branches
- Require reviews for main branch merges

---

## 🚀 Custom Domain (Optional)

To use a custom domain:

1. **Add CNAME file** to repository:
```
your-domain.com
```

2. **Configure DNS** with your registrar:
```
CNAME record pointing to: shivam2003-dev.github.io
```

3. **Enable in GitHub Pages**:
- Settings → Pages → Custom domain
- Enter domain
- Verify/enable HTTPS

---

## 📊 Performance Optimization

### For Faster Builds:

1. **Exclude unnecessary files** in `_config.yml`:
```yaml
exclude:
  - vendor/
  - Gemfile
  - Gemfile.lock
  - handout/
```

2. **Minimize CSS/JS** in production

3. **Use image optimization** for faster loading

### Check Performance:

```bash
# Build performance
bundle exec jekyll build --profile

# Analyze build time per file
```

---

## 🔄 Continuous Deployment Pipeline

```
Local Changes
    ↓
git add/commit
    ↓
git push origin main
    ↓
GitHub Webhook Triggered
    ↓
GitHub Actions Build Job
    ↓
Jekyll Build Process
    ↓
Deploy to gh-pages Branch
    ↓
GitHub Pages Serves Site
    ↓
Site Live at: https://shivam2003-dev.github.io/semester_1_all_course
```

---

## 📞 Need Help?

### Common Resources

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Help](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

### Troubleshooting Links

- [GitHub Pages Troubleshooting](https://docs.github.com/en/pages/getting-started-with-github-pages/troubleshooting-jekyll-build-errors)
- [Jekyll Troubleshooting](https://jekyllrb.com/docs/troubleshooting/)

### Get Support

- Check [Repository Issues](https://github.com/shivam2003-dev/semester_1_all_course/issues)
- Create new issue with details
- Include error logs and steps to reproduce

---

## ✅ Deployment Checklist

- [ ] All files committed to main branch
- [ ] No build errors locally (`bundle exec jekyll build`)
- [ ] _config.yml has correct baseurl
- [ ] GitHub Pages enabled in repository settings
- [ ] Correct branch selected (main, gh-pages, or main/docs)
- [ ] Waited 2-5 minutes after push
- [ ] Site accessible at GitHub Pages URL
- [ ] All pages load correctly
- [ ] Links work properly
- [ ] Mobile view responsive
- [ ] CSS/styling applied correctly
- [ ] Images load properly

---

## 🎉 Success!

Your AIML Course Hub is now live on GitHub Pages!

**Live URL**: `https://shivam2003-dev.github.io/semester_1_all_course`

**Next Steps:**
1. Share the link with classmates
2. Gather feedback
3. Continue updating content
4. Monitor analytics
5. Celebrate! 🚀

---

**Last Updated**: December 2025

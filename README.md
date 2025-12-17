# 📚 AIML Course Hub - BITS Pilani

A comprehensive online learning platform for **M.Tech (AIML)** and **M.Tech (DSE)** students at BITS Pilani, Work Integrated Learning Programs.

## 🌟 Features

- **📖 4 Comprehensive Courses**: Math Foundations, Machine Learning, Deep Learning, Statistical Methods
- **📝 Detailed Notes**: Complete course materials with examples and explanations
- **⚡ Exam Tips**: Strategic preparation guidance with focus areas
- **💼 Industry Tips**: Real-world applications and professional insights
- **🔗 Curated Resources**: 100+ external links, papers, tools, and references
- **⚠️ Warnings & Hints**: Common pitfalls and helpful study tips
- **📊 Lab Work**: Hands-on exercises aligned with theory
- **📋 Cheatsheets**: Quick reference guides for revision
- **🎨 Clean Design**: Minimalist, responsive, and mobile-friendly

## 📚 Courses Included

### 1. Math Foundations for AI/ML (AIML ZC417 / DSECT ZC417)
Essential mathematics for machine learning: linear algebra, calculus, probability, statistics, and optimization.

### 2. Machine Learning Fundamentals (AIML ZC420 / DSECT ZC420)
Core ML concepts: supervised learning, unsupervised learning, model evaluation, and ensemble methods.

### 3. Deep Neural Networks (AIML ZC421 / DSECT ZC421)
Modern deep learning: neural networks, CNNs, RNNs, attention mechanisms, and transformers.

### 4. Introduction to Statistical Methods (AIML ZC418 / DSECT ZC418)
Statistical analysis and inference: probability, distributions, hypothesis testing, time series, and GMM.

## 🚀 Quick Start

### View Online
Visit: **[https://shivam2003-dev.github.io/semester_1_all_course](https://shivam2003-dev.github.io/semester_1_all_course)**

### Run Locally

**Prerequisites:**
- Ruby 2.5+
- Bundler
- Git

**Installation:**
```bash
# Clone repository
git clone https://github.com/shivam2003-dev/semester_1_all_course.git
cd semester_1_all_course

# Install dependencies
bundle install

# Start Jekyll server
bundle exec jekyll serve

# Visit http://localhost:4000/semester_1_all_course
```

## 📁 Repository Structure

```
semester_1_all_course/
├── _config.yml              # Jekyll configuration
├── _layouts/                # Page templates
│   ├── default.html        # Main layout
│   └── course.html         # Course page layout
├── _includes/              # Reusable components
│   ├── header.html
│   └── footer.html
├── assets/
│   ├── css/
│   │   └── style.css       # Main stylesheet
│   ├── js/                 # JavaScript files
│   └── images/             # Images
├── courses/                # Course pages
│   ├── 01-math-foundations.md
│   ├── 02-machine-learning.md
│   ├── 03-deep-learning.md
│   ├── 04-statistical-methods.md
│   └── index.md            # Courses index
├── resources/              # Resources page
│   └── index.md
├── about/                  # About page
│   └── index.md
├── index.md                # Homepage
├── Gemfile                 # Ruby dependencies
└── README.md               # This file
```

## 🛠️ Technologies

- **Jekyll**: Static site generator
- **GitHub Pages**: Hosting
- **Markdown**: Content format
- **HTML/CSS**: Responsive design
- **Liquid**: Template language

## 📝 Content Features

### Note Boxes (📌)
Important conceptual points and foundational ideas.

### Warning Boxes (⚠️)
Common pitfalls and areas where students frequently make mistakes.

### Danger Boxes (🔴)
Critical issues that can cause major problems if missed.

### Exam Tip Boxes (✅)
Strategic exam preparation advice and focus areas.

### Industry Tip Boxes (💡)
Real-world applications and how industry uses concepts.

### Info Boxes (ℹ️)
General information and clarifications.

## 📊 Course Content

Each course includes:

| Component | Details |
|-----------|---------|
| **Modules** | 4-6 comprehensive modules per course |
| **Topics** | 20-30+ topics across all modules |
| **Examples** | Real-world examples and case studies |
| **Formulas** | Mathematical derivations and key equations |
| **Lab Work** | 4-6 hands-on laboratory exercises |
| **Resources** | External links, papers, tools, communities |
| **Cheatsheets** | Quick reference guides for revision |
| **Practice** | Example problems and solutions |

## 🎓 Learning Paths

### Path A: Foundations First
1. Math Foundations
2. Statistical Methods
3. Machine Learning
4. Deep Neural Networks

### Path B: Practical Focus
1. Machine Learning
2. Deep Neural Networks
3. Math Foundations
4. Statistical Methods

### Path C: Systems Approach
1. Statistical Methods
2. Math Foundations
3. Machine Learning
4. Deep Neural Networks

## 🔗 External Resources

The site includes 100+ curated resources:
- University courses (Stanford, MIT, CMU)
- Research papers and textbooks
- Online learning platforms (Coursera, edX, Fast.ai)
- Coding tools and libraries
- Practice platforms (Kaggle, LeetCode)
- BI and visualization tools
- Active communities and forums

## 📱 Mobile Friendly

- Fully responsive design
- Works on all devices
- Optimized for mobile viewing
- Fast loading times
- Accessible navigation

## 🤝 Contributing

### Report Issues
Found an error or broken link?
- [Create an Issue](https://github.com/shivam2003-dev/semester_1_all_course/issues)

### Suggest Improvements
- Better explanations
- Additional resources
- New topics
- Code examples

### Contributing Guidelines
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project references official BITS Pilani course materials. Educational use is encouraged.

## 📞 Contact & Support

- **GitHub**: [@shivam2003-dev](https://github.com/shivam2003-dev)
- **Issues**: [GitHub Issues](https://github.com/shivam2003-dev/semester_1_all_course/issues)
- **Repository**: [semester_1_all_course](https://github.com/shivam2003-dev/semester_1_all_course)

## 🙏 Acknowledgments

- BITS Pilani for official course handouts
- Course authors and instructors
- Open-source community (Jekyll, GitHub Pages)
- All contributing students and educators

## ⚖️ Disclaimer

This is an **unofficial educational resource**. For official course information, always consult:
- Official course handouts
- Faculty instructors
- BITS Pilani official resources

## 🚀 Deployment

### GitHub Pages Setup

**Branch Deployment:**
1. Push to `gh-pages` branch
2. Or configure in repository settings
3. Site builds automatically
4. Visit: `https://shivam2003-dev.github.io/semester_1_all_course`

**First-Time Setup:**
```bash
# Ensure gh-pages branch exists
git branch -a

# If gh-pages doesn't exist, create it
git checkout --orphan gh-pages
git rm -rf .
git commit --allow-empty -m "Initial gh-pages commit"
git push origin gh-pages

# Return to main branch
git checkout main

# Deploy
bundle exec jekyll build
# Contents of _site/ are auto-deployed to gh-pages
```

### Build Status
- Built with: Jekyll 4.3.0
- Hosted on: GitHub Pages
- Last build: December 2025
- Status: ✅ Active

## 📈 Statistics

- **Total Courses**: 4
- **Total Modules**: 18+
- **Total Topics**: 80+
- **External Resources**: 100+
- **Lab Exercises**: 6
- **Responsive**: 100%
- **Mobile Friendly**: Yes
- **SEO Optimized**: Yes

---

**Happy Learning!** 🎓📚

If you find this helpful, please star the repository and share with classmates!

**Last Updated**: December 2025
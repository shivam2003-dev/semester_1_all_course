---
layout: default
title: All Courses
---

# 📚 All Courses

Browse our comprehensive collection of AI/ML courses. Each course includes detailed notes, exam tips, industry insights, and curated resources.

---

<div class="course-grid">
  {% for course in site.courses %}
  <div class="course-card">
    <div class="course-card-header">
      <h3>{{ course.title }}</h3>
    </div>
    <div class="course-card-body">
      <p>{{ course.short_description }}</p>
      <div class="course-meta">
        {% if course.level %}<span>📊 Level: {{ course.level }}</span>{% endif %}
        {% if course.credits %}<span>⭐ {{ course.credits }} Credits</span>{% endif %}
      </div>
      {% if course.topics %}
      <div style="margin-top: 1em;">
        <strong>Topics:</strong>
        <ul style="margin: 0.5em 0; padding-left: 1.5em;">
          {% for topic in course.topics limit: 4 %}
          <li style="font-size: 0.9em; margin: 0.25em 0;">{{ topic }}</li>
          {% endfor %}
          {% if course.topics.size > 4 %}
          <li style="font-size: 0.9em; margin: 0.25em 0;">... and more</li>
          {% endif %}
        </ul>
      </div>
      {% endif %}
    </div>
    <div class="course-card-footer">
      <a href="{{ course.url }}" class="btn" style="width: 100%; text-align: center;">View Full Course →</a>
    </div>
  </div>
  {% endfor %}
</div>

---

## 📖 Course Structure

Each course includes:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5em; margin: 2em 0;">
  <div class="card">
    <h4>📝 Comprehensive Notes</h4>
    <p>Detailed notes for every module with examples, formulas, and explanations.</p>
  </div>
  <div class="card">
    <h4>⚡ Exam Tips</h4>
    <p>Strategic guidance on what to focus on and common exam mistakes to avoid.</p>
  </div>
  <div class="card">
    <h4>💼 Industry Tips</h4>
    <p>Real-world applications and how concepts are used in industry.</p>
  </div>
  <div class="card">
    <h4>🔗 Resources</h4>
    <p>External links to papers, tutorials, libraries, and other learning materials.</p>
  </div>
  <div class="card">
    <h4>📋 Cheatsheets</h4>
    <p>Quick reference guides with key formulas and concepts.</p>
  </div>
  <div class="card">
    <h4>⚠️ Warnings & Hints</h4>
    <p>Common pitfalls to avoid and helpful hints for difficult concepts.</p>
  </div>
</div>

---

## 🎯 Learning Paths

### Path 1: Foundations First
1. **Math Foundations** - Build your mathematical toolkit
2. **Machine Learning** - Apply math to real algorithms
3. **Deep Learning** - Advanced neural networks
4. **ISM** - Understand data systems

### Path 2: Practical Focus
1. **Machine Learning** - Hands-on ML algorithms
2. **Deep Learning** - Modern architectures
3. **ISM** - Data engineering and warehousing
4. **Math Foundations** - Deeper understanding of theory

### Path 3: Systems First
1. **ISM** - Understand data systems
2. **Math Foundations** - Mathematical foundations
3. **Machine Learning** - Build ML solutions
4. **Deep Learning** - Advanced techniques

---

## 🌟 How to Use These Courses

1. **Start with Course Overview**: Get an understanding of the scope and topics
2. **Read Module by Module**: Work through each section systematically
3. **Pay Attention to Alerts**: Notes, warnings, and tips highlight important content
4. **Use Cheatsheets**: Quick review before exams or project work
5. **Check Resources**: Dive deeper with external links and papers
6. **Ask Questions**: Refer back when concepts feel unclear

---

## 📚 Prerequisites

| Course | Prerequisites |
|--------|---------------|
| **Math Foundations** | High school mathematics |
| **Machine Learning** | Math Foundations (or strong math background) |
| **Deep Learning** | Machine Learning (or equivalent knowledge) |
| **ISM** | Basic computer science concepts |

---

## ✨ Special Features

<div class="alert alert-note">
  <h4>📝 Note Boxes</h4>
  <p>These highlight important conceptual points and foundational ideas.</p>
</div>

<div class="alert alert-warning">
  <h4>⚠️ Warning Boxes</h4>
  <p>These warn about common pitfalls and areas where students frequently make mistakes.</p>
</div>

<div class="alert alert-danger">
  <h4>🔴 Danger Boxes</h4>
  <p>These highlight critical issues that can cause major problems if missed.</p>
</div>

<div class="alert alert-success">
  <h4>✅ Exam Tip Boxes</h4>
  <p>These provide strategic exam preparation advice and focus areas.</p>
</div>

<div class="alert alert-tip">
  <h4>💡 Industry Tip Boxes</h4>
  <p>These share real-world applications and how industry uses these concepts.</p>
</div>

<div class="alert alert-info">
  <h4>ℹ️ Info Boxes</h4>
  <p>These provide general information and clarifications.</p>
</div>

---

## 🚀 Getting Started

**Pick a course and dive in!**

- New to AI/ML? Start with **Math Foundations**
- Want to build models? Begin with **Machine Learning**
- Interested in cutting-edge techniques? Explore **Deep Learning**
- Need to work with data systems? Check **ISM**

Each course is self-contained but references others when relevant. Happy learning! 🎓

---

Last Updated: December 2025

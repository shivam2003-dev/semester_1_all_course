---
layout: default
title: AIML Course Hub
---

<div class="hero-section" style="background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); color: white; padding: 3em 2em; border-radius: 8px; text-align: center; margin-bottom: 2em;">
  <h1 style="color: white; border: none; font-size: 2.8em;">Welcome to AIML Course Hub</h1>
  <p style="font-size: 1.2em; margin: 1em 0 0; opacity: 0.95;">Comprehensive learning resources for AI/ML Semester 1 courses</p>
  <p style="margin-top: 1.5em;">
    <a href="#courses" class="btn" style="background-color: white; color: #3498db; font-weight: bold;">Explore Courses</a>
  </p>
</div>

## 🎯 About This Platform

This is a **comprehensive course website** featuring:

<div class="resources-grid">
  <div class="resource-item">
    <h4>📝 Detailed Notes</h4>
    <p>Complete course notes with explanations, formulas, and key concepts for each course.</p>
  </div>
  <div class="resource-item">
    <h4>⚡ Exam Tips</h4>
    <p>Strategic exam preparation guidance with important concepts and question patterns.</p>
  </div>
  <div class="resource-item">
    <h4>💼 Industry Tips</h4>
    <p>Real-world applications and industry insights for each topic.</p>
  </div>
  <div class="resource-item">
    <h4>🔗 Resources</h4>
    <p>Curated links to papers, tutorials, and external learning materials.</p>
  </div>
</div>

---

## 📚 Our Courses {#courses}

<div class="course-grid">
  {% for course in site.courses %}
  <div class="course-card">
    <div class="course-card-header">
      <h3>{{ course.title }}</h3>
    </div>
    <div class="course-card-body">
      <p>{{ course.short_description }}</p>
      <div class="course-meta">
        {% if course.level %}<span>📊 {{ course.level }}</span>{% endif %}
        {% if course.credits %}<span>⭐ {{ course.credits }} Credits</span>{% endif %}
      </div>
    </div>
    <div class="course-card-footer">
      <a href="{{ course.url }}" class="btn" style="width: 100%; text-align: center;">View Course →</a>
    </div>
  </div>
  {% endfor %}
</div>

---

## 🌟 Key Features

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5em;">
  <div class="card">
    <h3>📖 Comprehensive Content</h3>
    <p>Each course includes detailed notes covering all topics, with clear examples and explanations.</p>
  </div>
  <div class="card">
    <h3>💡 Smart Learning Tools</h3>
    <p>Features include notes, warnings, hints, and tips to enhance your learning experience.</p>
  </div>
  <div class="card">
    <h3>🎯 Exam Preparation</h3>
    <p>Exam tips, important formulas, and common question patterns to ace your tests.</p>
  </div>
  <div class="card">
    <h3>🔬 Industry Insights</h3>
    <p>Real-world applications and professional tips from industry experts.</p>
  </div>
  <div class="card">
    <h3>🔗 Curated Links</h3>
    <p>Research papers, GitHub repositories, cheatsheets, and other valuable resources.</p>
  </div>
  <div class="card">
    <h3>🎓 Well Organized</h3>
    <p>Clean, minimalist design with easy navigation and responsive layout.</p>
  </div>
</div>

---

## 📋 How to Use This Site

1. **Browse Courses**: Visit the [Courses page]({{ site.baseurl }}/courses/) to see all available courses
2. **Read Notes**: Each course has detailed notes organized by topics
3. **Check Warnings & Tips**: Look for special alerts with exam tips and industry insights
4. **Access Resources**: Find external links, papers, and GitHub repositories
5. **Download Cheatsheets**: Quick reference guides for each course

---

## 🚀 Getting Started

<div class="alert alert-info">
  <h4>💻 For Students</h4>
  <p>Start with any course that interests you. Each course page has a structured syllabus, detailed notes, and practice resources. Use the sidebar to jump to specific topics.</p>
</div>

<div class="alert alert-tip">
  <h4>🎓 Study Tips</h4>
  <p>Make sure to read the <strong>Exam Tips</strong> and <strong>Industry Tips</strong> sections. They provide valuable insights into how concepts are applied in real scenarios and what to focus on during exams.</p>
</div>

---

## 📞 Connect & Contribute

- **GitHub**: [shivam2003-dev/semester_1_all_course](https://github.com/shivam2003-dev/semester_1_all_course)
- **Report Issues**: Create an issue on GitHub if you find any errors
- **Suggestions**: Feel free to suggest improvements or additional resources

---

Last Updated: December 2025

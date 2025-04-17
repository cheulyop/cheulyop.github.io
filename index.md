---
layout: base.njk
title: Home # Or whatever title you want for the homepage
---

{% include "about-content.md" %}

<!-- <a class="page-link" href="{{ '/assets/pdf/cv-apr25.pdf' | url }}">CV</a> (Last updated: Apr. 2025) -->

## Blog

<ul class="post-list">
{% for post in collections.posts %}
  <li>
    <a href="{{ post.url }}">{{ post.data.title }}</a>
    <small>{{ post.date | date("LLLL dd, yyyy") }}</small>
  </li>
{% endfor %}
</ul>

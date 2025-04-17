---
layout: base.njk
title: Home # Or whatever title you want for the homepage
---

{% include "about-content.md" %}

## Blog

<ul class="post-list">
{% for post in collections.posts %}
  <li>
    <a href="{{ post.url }}">{{ post.data.title }}</a>
    <small>{{ post.date | date("LLLL dd, yyyy") }}</small>
  </li>
{% endfor %}
</ul>

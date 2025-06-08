---
layout: base.njk
title: Blog
permalink: /blog/
---

# Blog

<ul class="post-list">
{% for post in collections.posts %}
  <li>
    <a href="{{ post.url }}">{{ post.data.title }}</a>
    <small>{{ post.date | date("LLLL dd, yyyy") }}</small>
  </li>
{% endfor %}
</ul>

---
layout: base.njk
title: Blog
permalink: /blog/
---

<div class="blog-page">

# Blog

{% if collections.posts.length > 0 %}

<ul class="post-list">
{% for post in collections.posts %}
  <li>
    <a href="{{ post.url }}">{{ post.data.title }}</a>
    <small>{{ post.date | date("LLLL dd, yyyy") }}</small>
  </li>
{% endfor %}
</ul>
{% else %}
<div class="no-posts">
  <p>No posts yet!</p>
</div>
{% endif %}

</div>

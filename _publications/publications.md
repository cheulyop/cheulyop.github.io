---
layout: none
description:
years: [2021, 2020, 2018]
---


<header class="pub-heading">
	<h4>Manuscripts & Preprints</h4>
</header>
<h5 class="pub-caption">* Works in this section are subject to changes until their official publication</h5>

{% bibliography -f preprints %}

<header class="pub-heading">
	<h4>Refereed Conference & Journal Papers</h4>
</header>

{% for y in page.years %}
  <h3 class="year">{{y}}</h3>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

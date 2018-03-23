---
layout: post
title:  "Setting Up a Custom Domain for Github Pages with Google Domains"
date: 2018-03-23 00:00:00 -0500
categories: jekyll github documentation
---

### Getting started
Setting up a custom domain for Github pages turned out to be more complicated than I thought. Github lets you use custom domains, but only without https. I have a domain I got from Google Domains, but simply setting it in Github doesn't show anything but the message that the site can't be reached. To get around this problem, I did some searches and found that [CloudFlare][2] lets you use custom domains with Github.

### Migrating DNS to CloudFlare
After setting up a CloudFlare account and entering my custom domain `cheulyoung.com`, I added CloudFlare nameservers to Google Domains as shown in the screenshot below:

![Setting custom DNS]({{'/assets/img/custom_dns.png' | absolute_url}})

Once I added new DNS, CloudFlare asked me to check back in several hours until new settings are applied, but it only took few minutes.

Then I created a CNAME file and added my domain to the file. This was as simple as `echo "https://cheulyoung.com" > CNAME`. As a next step, I added a DNS record on CloudFlare as in the screenshot:

![Adding DNS records]({{'/assets/img/dns_records.png' | absolute_url}})

After creating a CNAME file, don't forget to push it to Github repo.

### Setting HTTPS
Finally, I set up https for my domain. It could be done by going to the Crypto tab in CloudFlare and selecting "Full" for SSL and turning "Alwyas use HTTPS" on.

After these steps, I could get my custom domain to point to my Jekyll website hosted by Github.

[1]: https://desiredpersona.com/install-jekyll-on-macos/
[2]: https://www.cloudflare.com/
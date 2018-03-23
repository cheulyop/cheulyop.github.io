---
layout: post
title:  "Setting Up a Custom Domain for Github Pages with Google Domains"
categories: jekyll github documentation
---

### Getting started
Setting up a custom domain for Github pages turned out to be more complicated than I thought. Github lets you use custom domains, but only without https. I have a domain I got from Google Domains, but simply setting it in Github doesn't show anything but the message that the site can't be reached. To get around this problem, I did some searches and found this [post][1] that explains how to use Cloudflare to get custom domains to work with Github.

### Migrating DNS to Cloudflare
So I set up a Cloudflare account and enter my custom domain `cheulyoung.com`. Then I add Cloudflare nameservers to Google Domains as shown in the screenshot below:

![Setting custom DNS]({{'/assets/img/custom_dns.png' | absolute_url}})

Now Cloudflare prompts me to check back in several hours while new settings are getting applied, but it turns out to take only few minutes.

Then I create a CNAME file and add my domain to the file. This step is as simple as `echo "www.cheulyoung.com" > CNAME` in the directory of the website. As a next step, I add a DNS record on Cloudflare as in the screenshot:



[1]: https://desiredpersona.com/install-jekyll-on-macos/
---
layout: post
title:  "Making a Website with Github and Jekyll"
date:   2018-03-12 17:13:00 -0500
categories: jekyll github documentation
---

This is the first post on this website, which I'm creating with [Jekyll][1]. I used [Squarespace][2] for a past month to host my portfolio and found it fairly convenient and usable, but I still wanted the experience of creating my own website, so I chose to make this homepage.

I don't have much knowledge in front-end development so the website looks quite bare as of now, but hopefully it'll get better as I learn more Jekyll and all.

If you happen to stumble upon this post, I hope some scribbles I'm putting here help you somehow. I'll start with listing resources I referred to as I was building this thing.

* [**A beginner's guide to setting up a development environment on Mac OS X**][3]: I found this document useful for setting up a dev environment. Thanks to [```nicolashery```][4] for putting these together. However, I didn't follow everything on the document, as some parts of this document were outdated.
* [**Make a Static Website with Jekyll**][4]: A great guide put together by [Tania][6]. Thanks to her.

Now here are steps I took to create this website. Steps aren't thorough but should be okay enough to give an idea.
#### **Step 1. Create a new GitHub repository**
The first step is to create a new GitHub repo for the website. Follow this [documentation][7] provided by GitHub to do so.

#### **Step 2. Make sure Jekyll's requirements are met**
Jekyll needs [Ruby][9] and [RubyGems][10]. Getting [RVM][11] should take care of both. To install RVM, run this command on a terminal:
```bash
$ \curl -sSL https://get.rvm.io | bash -s stable
```
To check versions of Ruby and RubyGems, use:
```bash
$ which ruby
$ which gem

# and to update all gems
$ gem update
```

#### **Step 3. Install Jekyll and create a project**
Installing Jekyll and creating a new project is explained in [Jekyll documentation][8]. Or simply use below commands:
```bash
# install Jekyll and Bundler gems
$ gem install jekyll bundler

# create a new jekyll site at ./<directory>
$ jekyll new <directory>
```

#### **Step 4. Add created files and folders to Github**
Add created Jekyll site to Github to control for versions as shown below:
```bash
# in a directory where your jekyll site is placed, if git isn't already initiated
$ git init

# stage changed files to be committed
$ git add .

# commit changes and push origin to master
$ git commit -m 'message'
$ git push -u origin master
```


[1]: https://jekyllrb.com/
[2]: https://www.squarespace.com/
[3]: https://github.com/nicolashery/mac-dev-setup
[4]: https://github.com/nicolashery
[5]: https://www.taniarascia.com/make-a-static-website-with-jekyll/
[6]: https://www.taniarascia.com/
[7]: https://pages.github.com/
[8]: https://jekyllrb.com/docs/quickstart/
[9]: https://www.ruby-lang.org/en/downloads/
[10]: https://rubygems.org/pages/download
[11]: https://rvm.io/
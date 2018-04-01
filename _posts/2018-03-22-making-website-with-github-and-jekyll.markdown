---
layout: post
title:  "Making a Website with GitHub and Jekyll"
date:   2018-03-22 00:00:00 -0400
categories: [blog]
---

This is the first post on this website, which I'm creating with [Jekyll][1]. I used [Squarespace][2] for a past month to host my portfolio, but I wanted the experience of creating my website from scratch, so I chose to make this homepage with Jekyll. Pros of using Jekyll is that it's free and highly customizable. If you happen to stumble upon this post, I hope you find what I'm putting here useful in any way.

Now here are steps I took.
### Step 1. Create a new GitHub repository
The first step is to create a new GitHub repo for the website. Follow this [documentation][7] provided by GitHub to do so.

### Step 2. Make sure Jekyll's requirements are met
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

### Step 3. Install Jekyll and create a project
Installing Jekyll and creating a new project is explained in [Jekyll documentation][8]. Or simply use commands below:
```bash
# install Jekyll and Bundler gems
$ gem install jekyll bundler

# create a new jekyll site at ./<directory>
$ jekyll new <directory>
```
The new Jekyll site created with `jekyll new` command will have [Minima][12] as it's default theme.

### Step 4. Add created files and folders to GitHub
Upload created Jekyll site to GitHub to control for versions as shown below:
```bash
# initiate a github repo in a directory for your jekyll site, if you haven't done so already
$ git init

# stage all changed files
$ git add .

# commit changes with a message and push them
$ git commit -m 'message'
$ git push -u origin master
```
Now the website should show up at `https://<username>.github.io`.

*Steps below this point are for customizing the website.*

### Step 5. Customize the website
To customize your website, you need to copy files and folders those are automatically managed by the theme gem. For the Minima theme gem, they are:
* `/assets`
* `/_layouts`
* `/_includes`
* and `/_sass`.

You can find these folders with typing `open $(bundle show minima)` on a terminal. Copy original folders to your site directory. After you've copied the folders, delete `theme: minima` from `_config.yml`. Then add `- jekyll-seo-tag` under plugins.

Now you should be able to play around with files and make changes to the website as you wish. I'm stopping here for now, but I'll add more to this post or create a new post on customizing later.

### References
Here are some resources I used as I was creating this website.

* [A beginner's guide to setting up a development environment on Mac OS X][3]: It might be off-topic, but I found this document useful for setting up a dev environment. Thanks to [`nicolashery`][4] for putting these together. However, as some parts of this document were outdated, I didn't follow everything in the document.
* [Make a Static Website with Jekyll][5]: A great guide by [Tania][6].
* [Organizing Jekyll Pages][13]: A blog post by [Damon Bauer][14] on how to add pages to Jekyll website and organize them under a subdirectory.

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
[12]: https://github.com/jekyll/minima
[13]: http://damonbauer.me/organizing-jekyll-pages/
[14]: http://damonbauer.me/
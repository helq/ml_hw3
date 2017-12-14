# How to create a pdf out of this #

First, you need to have installed the following tools:

- pandoc
- ruby
- latex (lualatex to be precise)

You can download them in debian with:

```
apt-get install pandoc rake
```

Once everything is installed you can run (remember `cd` to this directory first):

```
rake compile
```

And you should see now a pdf with name `ml_hw1.pdf`.

Enjoy ;)

PS: If you want to know how does the underlying LaTeX file looks like run `rake tex`.

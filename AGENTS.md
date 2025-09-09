# GEMINI.md

## Project Overview

This is a personal blog built with [Hugo](https://gohugo.io/), a static site generator written in Go. The blog is themed with [PaperMod](https://github.com/adityatelange/hugo-PaperMod) and contains technical articles on various topics, including data science, machine learning, and software engineering.

The project is configured for deployment on [Netlify](https://www.netlify.com/) and also includes a Docker-based development environment.

## Building and Running

There are two ways to build and run the site:

### Using Docker (for deployment)

The `docker-compose.yml` file defines a pipeline for serving the site. To start the server, run:

```bash
docker-compose up
```

The site will be available at `http://localhost:1313`.

The `web` service in the `docker-compose.yml` file will automatically rebuild the site when you make changes to the content.

To pull the latest content from the Git repository. Run `docker-compose up content` to trigger the Git pull operation.

### Using Hugo (for deployment)

The site can be built using the `hugo` command. This is the command used by Netlify for deployment, as specified in `netlify.toml`.

To build the site locally, you need to have Hugo installed. Then, run the following command:

```bash
hugo
```

This will generate the static site in the `public/` directory.

## Development Conventions

### Content

All content is located in the `content/` directory. Blog posts are in `content/post/`. Each post is a Markdown file with a front matter section that contains metadata like the title, date, and tags.

### Creating a new post

To create a new post, you can use the `hugo new` command:

```bash
hugo new post/your-post-title.md
```

This will create a new Markdown file in `content/post/` with a pre-populated front matter.

### Themes

The site uses the "PaperMod" theme, which is located in the `themes/PaperMod/` directory. The theme's configuration is in the `hugo.yaml` file.

### Static Files

Static files like images are located in the `static/` and `assets/` directories.

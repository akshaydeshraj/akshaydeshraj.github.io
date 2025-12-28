# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal blog built with Jekyll using the Tale theme, with custom Solarized dark/light mode theming.

## Commands

```bash
# Install dependencies
bundle install

# Run local development server
bundle exec jekyll serve

# Build for production
bundle exec jekyll build
```

Note: `_config.yml` changes require server restart.

## Architecture

### Theme System
- Uses **Tale** theme as base (`gem "tale"` in Gemfile)
- Custom **Solarized** color scheme with dark/light toggle
- Theme colors defined via CSS custom properties in `_sass/_theme-switcher.scss`
- Color variables sourced from `_sass/_solarized.scss`
- Toggle button in `_includes/navigation.html` switches `.theme-dark` class on `<html>`

### Key Customizations
- **`_sass/_theme-switcher.scss`**: Main styling file (~700 lines) that overrides Tale theme defaults with CSS variables for theming
- **`_includes/navigation.html`**: Custom nav with animated underlines and theme toggle button
- **`_includes/footer.html`**: Social links (GitHub, Twitter, email) with author info from `_config.yml`
- **`_layouts/post.html`**: Post template with reading time calculation and tag support
- **`tags.html`**: Custom tags page with JavaScript filtering by URL hash

### Content
- Posts go in `_posts/` with format `YYYY-MM-DD-title.markdown`
- Posts support `tags` in front matter for categorization
- Permalink format: `/:year-:month-:day/:title`

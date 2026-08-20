# Repository Guidelines

## Project Structure & Module Organization

The site is built with Astro. Page routes live in `src/pages`, reusable UI pieces in `src/components`, and shared shells in `src/layouts`. Long-form content (e.g., biography sections) resides in `src/content` with schema defined through `src/content.config.ts`. Global Sass lives in `src/styles`, while static assets such as favicons and PDFs belong in `public`. Build artifacts are emitted to `dist`; avoid editing that folder directly. Global site behavior is configured through `astro.config.mjs` and `src/config.ts`.

## Build, Test, and Development Commands

Run `npm install` once to hydrate dependencies. Use `npm run dev` for a local server with live reload. `npm run build` produces the production-ready bundle used by GitHub Pages, and `npm run preview` serves that build for spot checks. Before opening a pull request, run `npm run astro -- check` to surface TypeScript and Astro template issues early.

## Coding Style & Naming Conventions

Favor two-space indentation in `.astro`, `.ts`, and `.scss` files to match the existing codebase. Components and layouts should use PascalCase filenames (`Header.astro`, `Layout.astro`), while content collections use kebab-case slugs for consistency. Keep imports scoped with relative paths from `src`, and group frontmatter imports at the top of `.astro` files. When styling, prefer modular Sass files in `src/styles` and co-locate component-specific styles near the component when practical.

## Testing Guidelines

Automated testing is light today, so rely on Astro’s static analysis via `npm run astro -- check`. For new UI work, validate rendering in the dev server and the production preview to confirm no regressions in `dist`. When content changes drive visual updates, capture the relevant section or page screenshot for reviewers.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commits style (`feat(layout):`, `fix(header):`) with imperative summaries under 65 characters. Group related changes together and reference GitHub issues when applicable. Pull requests should include a concise description of what changed, why it matters, verification steps (commands run or screenshots), and any follow-up tasks noted in a checklist.

## Automatic Production Publishing

- After verifying an in-scope change, Codex must stage only the scoped files.
- Codex must create a Conventional Commit on `main` and push `main` to `origin`.
- Treat the implementation request as approval for the commit, push, and GitHub Pages deployment. Do not request separate approval.
- Run `npm run build` before pushing.
- After pushing, wait for `.github/workflows/deploy.yml` to finish.
- Verify the changed behavior at `https://cheul.dev/`.
- Stop and report when the worktree has unrelated changes or `main` diverges from `origin/main`.
- Also stop when verification or deployment fails.

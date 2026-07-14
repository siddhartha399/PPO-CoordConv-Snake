# CLAUDE.md

## Web Tools

Pick the web tool by the question being asked.

- **WebFetch** — known URL, just need its content. Default for a known URL.
- **WebSearch** — quick, general, fact-based lookups; ranked snippets are enough.
- **Exa** — *where should I look*. Discovery (research papers, docs, code, companies, people) when exact keywords are unknown; semantic > keyword match. Finds candidates, doesn't deep-read.
- **Firecrawl** — *read in depth*. Scrape a full page, crawl a site, map URLs, or extract structured fields across many pages. Use when WebFetch returns empty/garbled (JS-heavy, anti-bot).

Rule: WebFetch/WebSearch = "what does this page say" or "what's out there." Exa = "where to look." Firecrawl = "give me everything on this page/site, properly."

## Project

PPO-CoordConv-Snake: GPU-accelerated PPO (4,096 parallel envs) for Snake using CoordConv + zero-downsampling CNN. Trained on a single T4. See `README.md` for architecture and usage.

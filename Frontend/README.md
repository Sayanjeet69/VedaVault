# VedaVault Frontend

The VedaVault V1 visual shell is a standalone Angular application with two routes:

- `/` — the welcome experience
- `/chat` — the responsive mock conversation experience

This milestone intentionally uses a local mock service. It does not call FastAPI, Groq, or any other network API at runtime.

## Local development

```bash
npm install
npm start
```

Open `http://localhost:4200`.

## Validation

```bash
npm run build
npm test -- --watch=false --browsers=ChromeHeadless
```

The permanent TypeScript API models mirror the frozen VedaVault V1 HTTP contract. Mock source-card fields remain isolated from those interfaces so a future HTTP service can replace `MockVedaService` cleanly.

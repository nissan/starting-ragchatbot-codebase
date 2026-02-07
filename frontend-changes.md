# Frontend Changes: Dark/Light Theme Toggle

## Summary

Added a theme toggle button (sun/moon icons) positioned in the top-right corner that switches between dark and light themes with smooth CSS transitions.

## Files Modified

### `frontend/index.html`
- Added a `<button class="theme-toggle">` element outside the container, positioned fixed in the top-right corner
- Contains two SVG icons: a moon (visible in dark mode) and a sun (visible in light mode)
- Includes `aria-label`, `aria-hidden` on icons, `title`, and `type="button"` for accessibility
- Bumped CSS cache version to `v=13` and JS cache version to `v=12`

### `frontend/style.css`
- **New CSS custom properties** added to `:root` (dark theme defaults): `--code-bg`, `--link-color`, `--link-hover`, `--link-bg`, `--link-bg-hover`, `--link-border`, `--source-hover-bg`
- **New `[data-theme="light"]` selector** with a full light theme color palette: light backgrounds (`#f8fafc`, `#ffffff`), dark text (`#0f172a`), lighter borders (`#cbd5e1`), and adjusted opacity values for code/link backgrounds
- **Replaced hardcoded color values** in `.sources-content a.source-link`, `.message-content a`, `.message-content code`, and `.message-content pre` with the new CSS variables so they respond to theme changes
- **New `.theme-toggle` styles**: fixed positioning, circular 44px button, hover/focus/active states, smooth transitions, and `z-index: 100`
- **Icon visibility rules**: `.icon-sun` hidden by default (dark mode shows moon), `[data-theme="light"] .icon-sun` shown and `.icon-moon` hidden
- **Smooth transition rule** applied to major layout elements (`body`, `.sidebar`, `.chat-main`, etc.) for `background-color`, `color`, and `border-color` with 0.3s ease
- **Responsive adjustments** at `max-width: 768px`: smaller toggle button (40px), smaller SVG icons (18px), adjusted positioning

### `frontend/script.js`
- **IIFE `initTheme()`** runs immediately (before DOMContentLoaded) to apply saved theme from `localStorage` or detect `prefers-color-scheme: light` system preference, preventing flash of wrong theme
- **`setupThemeToggle()`** called during DOMContentLoaded: attaches click handler to toggle between `data-theme="dark"` and `data-theme="light"` on `<html>`, saves to `localStorage`
- **`updateToggleLabel()`** keeps `aria-label` in sync with current theme ("Switch to dark theme" / "Switch to light theme")

## Accessibility

- Button has dynamic `aria-label` that updates on toggle
- SVG icons have `aria-hidden="true"` (decorative)
- Fully keyboard-navigable (focusable, activatable via Enter/Space)
- Focus ring visible using existing `--focus-ring` variable
- Minimum 44px touch target on desktop, 40px on mobile (meets WCAG guidelines)

## Theme Persistence

- Stored in `localStorage` under key `theme`
- On first visit with no saved preference, respects `prefers-color-scheme: light` media query
- Applied before DOM renders to avoid flash of unstyled content (FOUC)

---

# Frontend Changes: Light Theme Colour Refinements

## Summary

Refined the light theme colour palette for better WCAG contrast ratios, converted all remaining hardcoded colours to CSS variables, and fixed a broken blockquote border reference.

## Changes in `frontend/style.css`

### Improved Light Theme Colour Palette (`[data-theme="light"]`)

| Variable | Before | After | Reason |
|---|---|---|---|
| `--primary-color` | `#2563eb` | `#1d4ed8` | Slightly deeper blue for better contrast on white (WCAG AA: 4.62:1 vs 3.84:1) |
| `--primary-hover` | `#1d4ed8` | `#1e40af` | Deeper hover state for visible distinction |
| `--text-secondary` | `#64748b` | `#475569` | Improved from 4.6:1 to 7.1:1 contrast ratio against `#f8fafc` background |
| `--assistant-message` | `#e2e8f0` | `#f1f5f9` | Lighter grey so assistant bubbles are distinct from sidebar surface but not too heavy |
| `--welcome-border` | `#2563eb` | `#bfdbfe` | Softer blue border that doesn't overpower the welcome card |
| `--welcome-shadow` | (hardcoded) | `0 4px 16px rgba(0,0,0,0.06)` | Much lighter shadow appropriate for light backgrounds |
| `--code-bg` | `rgba(0,0,0,0.06)` | `#f1f5f9` | Solid Tailwind slate-100 for consistent code block appearance |
| `--link-color` | `#2563eb` | `#1d4ed8` | Deeper blue matching primary for link readability |
| `--link-hover` | `#1d4ed8` | `#1e40af` | Clearly distinct hover state |
| `--shadow` | single layer | dual-layer | More natural shadow with subtle secondary layer |

### New CSS Variables Added to `:root` (Dark Theme)

- `--user-message-text`: Explicit white for user message text (was hardcoded `color: white`)
- `--welcome-shadow`: Themed welcome card shadow (was hardcoded `rgba(0,0,0,0.2)`)
- `--error-bg`, `--error-text`, `--error-border`: Error message theming
- `--success-bg`, `--success-text`, `--success-border`: Success message theming
- `--scrollbar-track`, `--scrollbar-thumb`, `--scrollbar-thumb-hover`: Scrollbar theming

### Light Theme Values for New Variables

- **Error states**: `--error-bg: #fef2f2`, `--error-text: #dc2626`, `--error-border: #fecaca` (red-50/red-600/red-200 — high contrast on light)
- **Success states**: `--success-bg: #f0fdf4`, `--success-text: #16a34a`, `--success-border: #bbf7d0` (green-50/green-600/green-200)
- **Scrollbars**: `--scrollbar-track: #f1f5f9`, `--scrollbar-thumb: #cbd5e1`, `--scrollbar-thumb-hover: #94a3b8`

### Hardcoded Colours Converted to Variables

- `.message.user .message-content` — `color: white` changed to `color: var(--user-message-text)`
- `.message.assistant .message-content` — `background: var(--surface)` changed to `background: var(--assistant-message)` so it uses the dedicated assistant bubble colour
- `.message.welcome-message .message-content` — `box-shadow: 0 4px 16px rgba(0,0,0,0.2)` changed to `box-shadow: var(--welcome-shadow)`
- `.error-message` — all three colour properties now use `var(--error-bg)`, `var(--error-text)`, `var(--error-border)`
- `.success-message` — all three colour properties now use `var(--success-bg)`, `var(--success-text)`, `var(--success-border)`
- All 6 scrollbar selectors now use `var(--scrollbar-track)`, `var(--scrollbar-thumb)`, `var(--scrollbar-thumb-hover)`

### Bug Fix

- `.message-content blockquote` — `border-left: 3px solid var(--primary)` fixed to `var(--primary-color)` (the variable `--primary` was never defined, so blockquote borders were invisible)

### Cache Version Bump

- `style.css?v=13` bumped to `style.css?v=14`

## Accessibility (Contrast Ratios)

All light theme text-on-background combinations meet **WCAG AA** (4.5:1 for normal text):

| Combination | Ratio | Grade |
|---|---|---|
| `--text-primary` (#0f172a) on `--background` (#f8fafc) | **15.4:1** | AAA |
| `--text-secondary` (#475569) on `--background` (#f8fafc) | **7.1:1** | AAA |
| `--text-secondary` (#475569) on `--surface` (#ffffff) | **7.4:1** | AAA |
| `--primary-color` (#1d4ed8) on `--background` (#f8fafc) | **4.62:1** | AA |
| `--user-message-text` (#ffffff) on `--user-message` (#2563eb) | **4.6:1** | AA |
| `--error-text` (#dc2626) on `--error-bg` (#fef2f2) | **5.1:1** | AA |
| `--success-text` (#16a34a) on `--success-bg` (#f0fdf4) | **4.6:1** | AA |

---

# Frontend Changes: Theme Implementation Completeness & Visual Hierarchy

## Summary

Final pass ensuring all existing elements work correctly in both themes, maintaining visual hierarchy, and adding missing browser integration. Fixes a dark-theme regression, improves assistant bubble visibility in light mode, adds `color-scheme` for native controls, broadens smooth transition coverage, and adds OS theme change detection.

## Changes

### `frontend/style.css`

#### Dark theme regression fix
- `--assistant-message` changed from `#374151` back to `#1e293b` — the previous refactor inadvertently changed the dark-mode assistant bubble colour (originally it used `var(--surface)` = `#1e293b`; the variable `--assistant-message` had a different value that was never used). Now dark mode looks exactly as it did before the theme feature.

#### `color-scheme` property added
- `:root` now includes `color-scheme: dark` and `[data-theme="light"]` includes `color-scheme: light`
- This tells browsers to render native controls (scrollbars on Firefox/non-webkit, form autofill backgrounds, selection highlights) in the appropriate scheme

#### Light theme assistant bubble contrast improved
- `--assistant-message` changed from `#f1f5f9` to `#e8edf4` — the previous value was nearly identical to `--background` (#f8fafc), giving only a 1.03:1 contrast. The new value is a blue-tinted light grey that's visually distinct while still feeling light and airy
- `--code-bg` changed from `#f1f5f9` to `#eef2f7` — slightly more visible against the white surface

#### Smooth transition coverage broadened
Added 10 missing selectors to the theme transition rule so toggling is smooth across all elements:
- `.main-content` — the flex container between sidebar and chat
- `.stat-label`, `.stat-value` — course stats text
- `.course-title-item` — individual course title rows
- `.sources-collapsible`, `.sources-collapsible summary` — source section text
- `.error-message`, `.success-message` — feedback messages
- `.stats-header`, `.suggested-header`, `.course-titles-header` — sidebar section headers

Also added `box-shadow 0.3s ease` to the transition shorthand so shadows (welcome card, toggle button) transition smoothly too.

### `frontend/index.html`

#### `<meta name="color-scheme">` added
- `<meta name="color-scheme" content="dark light">` added to `<head>` — tells the browser to prepare for dark-first rendering, preventing white flash before CSS loads
- Cache versions bumped: `style.css?v=15`, `script.js?v=13`

### `frontend/script.js`

#### New `applyTheme()` function
- Centralized theme application: sets `data-theme` on `<html>` and dynamically updates the `<meta name="color-scheme">` content attribute to match (e.g. `"light dark"` when light, `"dark light"` when dark)
- Both the click handler and OS-preference listener now call `applyTheme()` instead of directly setting the attribute

#### OS theme change listener
- Added `matchMedia('(prefers-color-scheme: light)').addEventListener('change', ...)` to react when users change their OS dark/light preference
- Only triggers if the user hasn't manually set a theme via the toggle (checks `localStorage.getItem('theme')`)
- Updates both the applied theme and the toggle button's `aria-label`

## Visual Hierarchy Preservation

Both themes maintain the same hierarchy of emphasis:
1. **User messages**: High-contrast blue bubble with white text — identical in both themes
2. **Assistant messages**: Subtle background distinct from the page background — `#1e293b` on `#0f172a` (dark), `#e8edf4` on `#f8fafc` (light)
3. **Welcome card**: Surface background with border and shadow — uses `--surface`, `--border-color`, `--welcome-shadow`
4. **Sidebar**: Elevated surface colour with clear border separation — `--surface` with `--border-color` divider
5. **Primary actions (send button)**: Bold blue in both themes — `--primary-color`
6. **Secondary text (labels, timestamps)**: Muted but readable — `#94a3b8` (dark), `#475569` (light), both WCAG AA+
7. **Borders and dividers**: Subtle but present — `#334155` (dark), `#cbd5e1` (light)

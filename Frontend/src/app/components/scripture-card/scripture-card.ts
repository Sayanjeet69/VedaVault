import { Component, input, output } from '@angular/core';
import { LucideBookOpen, LucideX } from '@lucide/angular';

@Component({
  selector: 'app-scripture-card',
  imports: [LucideBookOpen, LucideX],
  template: `
    <section class="source-card" [id]="panelId()" aria-label="Scripture source">
      <header>
        <span class="source-icon"><svg lucideBookOpen size="16" strokeWidth="1.7"></svg></span>
        <div><span class="kicker">Cited passage</span><h4>{{ citationLabel() }}</h4></div>
        <button type="button" aria-label="Collapse source" (click)="closed.emit()">
          <svg lucideX size="17" strokeWidth="1.8"></svg>
        </button>
      </header>
      <div class="source-summary">
        <span class="verse-id">{{ verseId() }}</span>
        <p>Verse text, translation, and source metadata are not included in the current API response.</p>
      </div>
    </section>
  `,
  styles: `
    .source-card { overflow: hidden; margin-top: .85rem; border: 1px solid var(--border-active); border-radius: var(--radius-lg); background: linear-gradient(145deg, rgba(255,153,51,.045), rgba(255,255,255,.018)); box-shadow: 0 18px 40px rgba(0,0,0,.24); animation: source-open 260ms var(--ease-out) both; }
    header { display: grid; grid-template-columns: auto 1fr auto; gap: .7rem; align-items: center; padding: .9rem 1rem; border-bottom: 1px solid var(--border-subtle); }
    .source-icon { display: grid; width: 2rem; height: 2rem; place-items: center; border: 1px solid rgba(255,153,51,.24); border-radius: .6rem; color: var(--saffron-primary); background: rgba(255,153,51,.06); }
    .kicker { display: block; color: var(--text-muted); font-size: .63rem; font-weight: 650; letter-spacing: .12em; text-transform: uppercase; }
    h4 { margin: .1rem 0 0; color: var(--text-primary); font-size: .85rem; font-weight: 650; }
    header button { display: grid; width: 2rem; height: 2rem; place-items: center; border: 0; border-radius: .55rem; color: var(--text-muted); background: transparent; }
    header button:hover { color: var(--text-primary); background: rgba(255,255,255,.05); }
    .source-summary { display: flex; min-width: 0; align-items: center; gap: .85rem; padding: 1rem 1.1rem 1.1rem; }
    .verse-id { flex: 0 0 auto; padding: .38rem .52rem; border: 1px solid rgba(255,153,51,.22); border-radius: .5rem; color: var(--saffron-soft); background: rgba(255,153,51,.045); font-family: ui-monospace, monospace; font-size: .7rem; }
    p { margin: 0; overflow-wrap: anywhere; color: var(--text-secondary); font-size: .76rem; line-height: 1.6; }
    @keyframes source-open { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
    @media (max-width: 520px) { .source-summary { align-items: flex-start; flex-direction: column; } }
  `,
})
export class ScriptureCardComponent {
  readonly verseId = input.required<string>();
  readonly panelId = input.required<string>();
  readonly closed = output<void>();

  citationLabel(): string {
    const match = /^BG_(\d+)_(\d+)$/.exec(this.verseId());
    return match ? `Bhagavad Gita ${Number(match[1])}.${Number(match[2])}` : this.verseId();
  }
}

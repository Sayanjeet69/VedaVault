import { Component } from '@angular/core';
import { LucideSparkles } from '@lucide/angular';

@Component({
  selector: 'app-thinking-indicator',
  imports: [LucideSparkles],
  template: `
    <div class="thinking" role="status" aria-live="polite">
      <span class="thinking-mark"><svg lucideSparkles size="16" strokeWidth="1.7"></svg></span>
      <span>Reflecting on the scriptures</span>
      <span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>
    </div>
  `,
  styles: `
    :host { display: block; width: min(100%, var(--conversation-width)); margin: 0 auto; }
    .thinking { display: flex; align-items: center; gap: .7rem; padding: .45rem 0; color: var(--text-secondary); font-size: .85rem; animation: thinking-in 280ms var(--ease-out) both; }
    .thinking-mark { display: grid; width: 2rem; height: 2rem; place-items: center; border: 1px solid var(--border-active); border-radius: .65rem; color: var(--saffron-primary); background: rgba(255,153,51,.055); }
    .dots { display: inline-flex; gap: .25rem; margin-left: -.25rem; }
    .dots i { width: .22rem; height: .22rem; border-radius: 50%; background: var(--saffron-primary); opacity: .2; animation: dot 1.25s infinite ease-in-out; }
    .dots i:nth-child(2) { animation-delay: 140ms; }
    .dots i:nth-child(3) { animation-delay: 280ms; }
    @keyframes dot { 0%, 70%, 100% { opacity: .18; transform: translateY(0); } 35% { opacity: .8; transform: translateY(-2px); } }
    @keyframes thinking-in { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
  `,
})
export class ThinkingIndicatorComponent {}

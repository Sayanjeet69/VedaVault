import { Component, input, output } from '@angular/core';
import { LucideCircleAlert, LucideX } from '@lucide/angular';

import { ErrorNoticeTone } from '../../core/models/client-error.models';

@Component({
  selector: 'app-error-notice',
  imports: [LucideCircleAlert, LucideX],
  template: `
    <section
      class="notice"
      [class.notice--clarification]="tone() === 'clarification'"
      [attr.role]="tone() === 'clarification' ? 'status' : 'alert'"
    >
      <svg lucideCircleAlert size="18" strokeWidth="1.8"></svg>
      <div><strong>{{ title() }}</strong><p>{{ message() }}</p></div>
      <button type="button" aria-label="Dismiss message" (click)="dismiss.emit()">
        <svg lucideX size="17" strokeWidth="1.8"></svg>
      </button>
    </section>
  `,
  styles: `
    .notice { display: grid; grid-template-columns: auto 1fr auto; gap: .75rem; align-items: start; width: min(100%, var(--conversation-width)); margin: 0 auto; padding: .9rem 1rem; border: 1px solid rgba(239,133,125,.24); border-radius: var(--radius-md); color: var(--danger); background: rgba(239,133,125,.055); }
    .notice--clarification { border-color: rgba(255,153,51,.24); color: var(--saffron-primary); background: rgba(255,153,51,.055); }
    strong { display: block; color: var(--text-primary); font-size: .84rem; }
    p { margin: .2rem 0 0; color: var(--text-secondary); font-size: .78rem; }
    button { display: grid; width: 2rem; height: 2rem; margin: -.35rem -.35rem 0 0; place-items: center; border: 0; border-radius: .5rem; color: var(--text-muted); background: transparent; }
    button:hover { color: var(--text-primary); background: rgba(255,255,255,.05); }
  `,
})
export class ErrorNoticeComponent {
  readonly title = input('Something interrupted the journey');
  readonly message = input('Please try again in a moment.');
  readonly tone = input<ErrorNoticeTone>('error');
  readonly dismiss = output<void>();
}

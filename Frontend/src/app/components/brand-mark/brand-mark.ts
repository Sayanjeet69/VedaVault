import { Component, input } from '@angular/core';

@Component({
  selector: 'app-brand-mark',
  template: `
    <span class="brand-mark" [class.brand-mark--hero]="size() === 'hero'" aria-hidden="true">ॐ</span>
  `,
  styles: `
    :host { display: inline-grid; place-items: center; }
    .brand-mark {
      display: inline-grid;
      place-items: center;
      width: 2.4rem;
      height: 2.4rem;
      color: var(--saffron-primary);
      font-family: var(--font-indic);
      font-size: 1.65rem;
      line-height: 1;
      filter: drop-shadow(0 0 12px rgba(255, 153, 51, 0.25));
    }
    .brand-mark--hero {
      width: 7.5rem;
      height: 7.5rem;
      border: 1px solid rgba(255, 153, 51, 0.2);
      border-radius: 2rem;
      background: linear-gradient(145deg, rgba(255,255,255,.035), rgba(255,153,51,.035));
      box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 28px 70px rgba(0,0,0,.42);
      font-size: 4.5rem;
      animation: om-enter 850ms var(--ease-out) both;
    }
    @keyframes om-enter {
      from { opacity: 0; transform: scale(.96); filter: drop-shadow(0 0 0 rgba(255,153,51,0)); }
      to { opacity: 1; transform: scale(1); filter: drop-shadow(0 0 22px rgba(255,153,51,.27)); }
    }
  `,
})
export class BrandMarkComponent {
  readonly size = input<'compact' | 'hero'>('compact');
}

import { Component, input, output } from '@angular/core';
import { LucideBookOpen, LucideChevronDown } from '@lucide/angular';

@Component({
  selector: 'app-citation-chip',
  imports: [LucideBookOpen, LucideChevronDown],
  template: `
    <button
      type="button"
      class="citation"
      [class.citation--active]="expanded()"
      [attr.aria-expanded]="expanded()"
      [attr.aria-controls]="panelId()"
      (click)="activated.emit()"
    >
      <svg lucideBookOpen size="14" strokeWidth="1.8"></svg>
      <span>{{ label() }}</span>
      <svg class="chevron" lucideChevronDown size="13" strokeWidth="1.8"></svg>
    </button>
  `,
  styles: `
    .citation { display: inline-flex; min-height: 2.35rem; align-items: center; gap: .45rem; padding: .42rem .67rem; border: 1px solid rgba(255,153,51,.27); border-radius: .65rem; color: #e9b77c; background: rgba(255,153,51,.035); font-size: .73rem; font-weight: 650; letter-spacing: .025em; transition: transform var(--duration-fast), border-color var(--duration-fast), background var(--duration-fast); }
    .citation:hover { transform: translateY(-1px); border-color: rgba(255,153,51,.5); background: rgba(255,153,51,.075); }
    .citation--active { border-color: var(--saffron-muted); color: var(--saffron-primary); background: rgba(255,153,51,.09); }
    .chevron { transition: transform var(--duration-fast); }
    .citation--active .chevron { transform: rotate(180deg); }
  `,
})
export class CitationChipComponent {
  readonly label = input.required<string>();
  readonly expanded = input(false);
  readonly panelId = input.required<string>();
  readonly activated = output<void>();
}

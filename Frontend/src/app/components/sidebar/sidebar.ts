import { Component, ElementRef, HostListener, effect, input, output, viewChild } from '@angular/core';
import {
  LucideBookOpen,
  LucideInfo,
  LucideLanguages,
  LucideMessageSquare,
  LucidePlus,
  LucideSparkles,
  LucideX,
} from '@lucide/angular';
import { RouterLink } from '@angular/router';

import { AnswerMode, V1LanguageCode } from '../../core/models/api.models';
import { LANGUAGE_OPTIONS, MODE_OPTIONS } from '../../core/models/chat.models';
import { BrandMarkComponent } from '../brand-mark/brand-mark';

@Component({
  selector: 'app-sidebar',
  imports: [
    RouterLink,
    BrandMarkComponent,
    LucideBookOpen,
    LucideInfo,
    LucideLanguages,
    LucideMessageSquare,
    LucidePlus,
    LucideSparkles,
    LucideX,
  ],
  templateUrl: './sidebar.html',
  styleUrl: './sidebar.css',
})
export class SidebarComponent {
  readonly open = input(false);
  readonly mobile = input(false);
  readonly language = input.required<V1LanguageCode>();
  readonly mode = input.required<AnswerMode>();
  readonly hasMessages = input(false);
  readonly busy = input(false);
  readonly closed = output<void>();
  readonly newJourney = output<void>();
  readonly languageChange = output<V1LanguageCode>();
  readonly modeChange = output<AnswerMode>();
  readonly closeButton = viewChild<ElementRef<HTMLButtonElement>>('closeButton');
  readonly drawer = viewChild<ElementRef<HTMLElement>>('drawer');
  readonly languageOptions = LANGUAGE_OPTIONS;
  readonly modeOptions = MODE_OPTIONS;

  constructor() {
    effect(() => {
      if (this.mobile() && this.open()) {
        queueMicrotask(() => this.closeButton()?.nativeElement.focus());
      }
    });
  }

  @HostListener('document:keydown.escape')
  closeOnEscape(): void {
    if (this.mobile() && this.open()) {
      this.closed.emit();
    }
  }

  @HostListener('document:keydown.tab', ['$event'])
  containDrawerFocus(event: Event): void {
    if (!this.mobile() || !this.open()) {
      return;
    }
    const keyboardEvent = event as KeyboardEvent;
    const focusable = Array.from(
      this.drawer()?.nativeElement.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ) ?? [],
    ).filter((element) => !element.hasAttribute('inert'));
    if (!focusable.length) {
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (keyboardEvent.shiftKey && document.activeElement === first) {
      keyboardEvent.preventDefault();
      last.focus();
    } else if (!keyboardEvent.shiftKey && document.activeElement === last) {
      keyboardEvent.preventDefault();
      first.focus();
    }
  }

  startNewJourney(): void {
    if (this.busy()) {
      return;
    }
    this.newJourney.emit();
    if (this.mobile()) {
      this.closed.emit();
    }
  }

  updateLanguage(event: Event): void {
    this.languageChange.emit((event.target as HTMLSelectElement).value as V1LanguageCode);
  }

  updateMode(event: Event): void {
    this.modeChange.emit((event.target as HTMLSelectElement).value as AnswerMode);
  }
}

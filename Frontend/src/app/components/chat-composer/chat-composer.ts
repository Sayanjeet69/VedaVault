import { Component, ElementRef, effect, input, output, viewChild } from '@angular/core';
import { LucideArrowUp, LucideChevronDown } from '@lucide/angular';

import { AnswerMode, V1LanguageCode } from '../../core/models/api.models';
import { LANGUAGE_OPTIONS, MODE_OPTIONS } from '../../core/models/chat.models';

@Component({
  selector: 'app-chat-composer',
  imports: [LucideArrowUp, LucideChevronDown],
  templateUrl: './chat-composer.html',
  styleUrl: './chat-composer.css',
})
export class ChatComposerComponent {
  readonly value = input('');
  readonly language = input.required<V1LanguageCode>();
  readonly mode = input.required<AnswerMode>();
  readonly busy = input(false);
  readonly valueChange = output<string>();
  readonly languageChange = output<V1LanguageCode>();
  readonly modeChange = output<AnswerMode>();
  readonly send = output<void>();
  readonly textarea = viewChild<ElementRef<HTMLTextAreaElement>>('textarea');
  readonly languageOptions = LANGUAGE_OPTIONS;
  readonly modeOptions = MODE_OPTIONS;

  constructor() {
    effect(() => {
      this.value();
      queueMicrotask(() => this.resizeTextarea());
    });
  }

  onInput(event: Event): void {
    const target = event.target as HTMLTextAreaElement;
    this.valueChange.emit(target.value);
    this.resizeTextarea(target);
  }

  onKeydown(event: KeyboardEvent): void {
    const canUseDesktopShortcut = window.matchMedia('(min-width: 769px)').matches;
    if (
      event.key === 'Enter' &&
      !event.shiftKey &&
      !event.isComposing &&
      canUseDesktopShortcut &&
      this.canSend
    ) {
      event.preventDefault();
      this.send.emit();
    }
  }

  submit(event: SubmitEvent): void {
    event.preventDefault();
    if (this.canSend) {
      this.send.emit();
    }
  }

  updateLanguage(event: Event): void {
    this.languageChange.emit((event.target as HTMLSelectElement).value as V1LanguageCode);
  }

  updateMode(event: Event): void {
    this.modeChange.emit((event.target as HTMLSelectElement).value as AnswerMode);
  }

  get canSend(): boolean {
    return this.value().trim().length > 0 && !this.busy();
  }

  get currentModeDescription(): string {
    return MODE_OPTIONS.find((option) => option.value === this.mode())?.description ?? '';
  }

  focus(): void {
    this.textarea()?.nativeElement.focus();
  }

  private resizeTextarea(target = this.textarea()?.nativeElement): void {
    if (!target) {
      return;
    }
    target.style.height = 'auto';
    target.style.height = `${Math.min(target.scrollHeight, 160)}px`;
  }
}

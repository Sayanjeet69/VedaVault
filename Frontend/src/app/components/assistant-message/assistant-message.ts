import { Component, computed, input, signal } from '@angular/core';
import { LucideQuote, LucideSparkles } from '@lucide/angular';

import { AssistantChatMessage } from '../../core/models/chat.models';
import { BrandMarkComponent } from '../brand-mark/brand-mark';
import { CitationChipComponent } from '../citation-chip/citation-chip';
import { ScriptureCardComponent } from '../scripture-card/scripture-card';

@Component({
  selector: 'app-assistant-message',
  imports: [
    LucideQuote,
    LucideSparkles,
    BrandMarkComponent,
    CitationChipComponent,
    ScriptureCardComponent,
  ],
  templateUrl: './assistant-message.html',
  styleUrl: './assistant-message.css',
})
export class AssistantMessageComponent {
  readonly message = input.required<AssistantChatMessage>();
  readonly expandedSource = signal<string | null>(null);
  readonly interpretation = computed(() => this.normalizedText(this.message().response.interpretation));
  readonly application = computed(() => this.normalizedText(this.message().response.application));
  readonly limitations = computed(() =>
    this.message()
      .response.limitations.map((limitation) => limitation.trim())
      .filter((limitation) => limitation.length > 0),
  );

  toggleSource(sourceId: string): void {
    this.expandedSource.update((current) => (current === sourceId ? null : sourceId));
  }

  citationLabel(sourceId: string): string {
    const match = /^BG_(\d+)_(\d+)$/.exec(sourceId);
    if (!match) {
      return sourceId;
    }
    return `BG ${Number(match[1])}.${Number(match[2])}`;
  }

  panelId(sourceId: string): string {
    return `source-${this.message().id}-${sourceId.replace(/[^a-zA-Z0-9_-]/g, '-')}`;
  }

  private normalizedText(value: string | null): string | null {
    const normalized = value?.trim();
    return normalized ? normalized : null;
  }
}

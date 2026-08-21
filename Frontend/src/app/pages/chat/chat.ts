import {
  Component,
  DestroyRef,
  ElementRef,
  OnDestroy,
  computed,
  inject,
  signal,
  viewChild,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { LucideMenu, LucideShieldCheck } from '@lucide/angular';
import { finalize } from 'rxjs';

import { AssistantMessageComponent } from '../../components/assistant-message/assistant-message';
import { BrandMarkComponent } from '../../components/brand-mark/brand-mark';
import { ChatComposerComponent } from '../../components/chat-composer/chat-composer';
import { ErrorNoticeComponent } from '../../components/error-notice/error-notice';
import { SidebarComponent } from '../../components/sidebar/sidebar';
import { ThinkingIndicatorComponent } from '../../components/thinking-indicator/thinking-indicator';
import { UserMessageComponent } from '../../components/user-message/user-message';
import { AnswerMode, AnswerRequest, V1LanguageCode } from '../../core/models/api.models';
import { ChatMessage } from '../../core/models/chat.models';
import { ChatErrorState, VedaVaultClientError } from '../../core/models/client-error.models';
import { VedaVaultConversationService } from '../../core/services/vedavault-conversation.service';

interface PromptSuggestion {
  label: string;
  language: V1LanguageCode;
}

@Component({
  selector: 'app-chat-page',
  imports: [
    LucideMenu,
    LucideShieldCheck,
    AssistantMessageComponent,
    BrandMarkComponent,
    ChatComposerComponent,
    ErrorNoticeComponent,
    SidebarComponent,
    ThinkingIndicatorComponent,
    UserMessageComponent,
  ],
  templateUrl: './chat.html',
  styleUrl: './chat.css',
  host: { class: 'page-host' },
})
export class ChatPage implements OnDestroy {
  private readonly conversation = inject(VedaVaultConversationService);
  private readonly destroyRef = inject(DestroyRef);

  readonly draft = signal('');
  readonly language = signal<V1LanguageCode>('en');
  readonly mode = signal<AnswerMode>('philosophical');
  readonly messages = signal<ChatMessage[]>([]);
  readonly thinking = signal(false);
  readonly resetting = signal(false);
  readonly drawerOpen = signal(false);
  readonly error = signal<ChatErrorState | null>(null);
  readonly hasMessages = computed(() => this.messages().length > 0);
  readonly busy = computed(() => this.thinking() || this.resetting());
  readonly sessionId = this.conversation.currentSessionId;
  readonly suggestions: readonly PromptSuggestion[] = [
    { label: 'What does the Gita say about anger?', language: 'en' },
    { label: 'कर्मयोग क्या है?', language: 'hi' },
    { label: 'জীবনে ব্যর্থতা নিয়ে গীতা কী বলে?', language: 'bn' },
    { label: 'कर्मण्येवाधिकारस्ते का अर्थ?', language: 'sa' },
  ];
  readonly isMobile = signal(false);
  readonly messageViewport = viewChild<ElementRef<HTMLElement>>('messageViewport');
  readonly composer = viewChild(ChatComposerComponent);
  readonly menuButton = viewChild<ElementRef<HTMLButtonElement>>('menuButton');

  private readonly mobileQuery = window.matchMedia('(max-width: 900px)');
  private nextMessageNumber = 1;

  constructor() {
    this.isMobile.set(this.mobileQuery.matches);
    this.mobileQuery.addEventListener('change', this.handleViewportChange);
  }

  ngOnDestroy(): void {
    this.mobileQuery.removeEventListener('change', this.handleViewportChange);
  }

  chooseSuggestion(suggestion: PromptSuggestion): void {
    this.draft.set(suggestion.label);
    this.language.set(suggestion.language);
    queueMicrotask(() => this.composer()?.focus());
  }

  sendMessage(): void {
    const query = this.draft().trim();
    if (!query || this.busy()) {
      return;
    }

    const request: AnswerRequest = {
      query,
      input_language: this.language(),
      response_language: this.language(),
      mode: this.mode(),
    };
    this.messages.update((messages) => [
      ...messages,
      {
        id: this.messageId('user'),
        role: 'user',
        content: query,
        language: this.language(),
      },
    ]);
    this.draft.set('');
    this.error.set(null);
    this.thinking.set(true);
    this.scrollToLatest();

    this.conversation
      .answer(request)
      .pipe(
        finalize(() => {
          this.thinking.set(false);
          this.scrollToLatest();
        }),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({
        next: (response) => {
          this.messages.update((messages) => [
            ...messages,
            {
              id: this.messageId('assistant'),
              role: 'assistant',
              response,
            },
          ]);
        },
        error: (error: unknown) => {
          this.error.set(this.toErrorState(error));
        },
      });
  }

  startNewJourney(): void {
    if (this.busy()) {
      return;
    }

    this.resetting.set(true);
    this.error.set(null);
    this.conversation
      .startNewJourney()
      .pipe(
        finalize(() => this.resetting.set(false)),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe(() => this.resetLocalConversation());
  }

  private resetLocalConversation(): void {
    this.messages.set([]);
    this.draft.set('');
    this.thinking.set(false);
    this.error.set(null);
    queueMicrotask(() => this.composer()?.focus());
  }

  openDrawer(): void {
    this.drawerOpen.set(true);
  }

  closeDrawer(): void {
    this.drawerOpen.set(false);
    queueMicrotask(() => this.menuButton()?.nativeElement.focus());
  }

  private readonly handleViewportChange = (event: MediaQueryListEvent): void => {
    this.isMobile.set(event.matches);
    if (!event.matches) {
      this.drawerOpen.set(false);
    }
  };

  private messageId(role: 'user' | 'assistant'): string {
    return `${role}-${this.nextMessageNumber++}`;
  }

  private scrollToLatest(): void {
    setTimeout(() => {
      const viewport = this.messageViewport()?.nativeElement;
      viewport?.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' });
    });
  }

  private toErrorState(error: unknown): ChatErrorState {
    if (error instanceof VedaVaultClientError) {
      return { title: error.title, message: error.userMessage, tone: error.tone };
    }
    return {
      title: 'Something interrupted the journey',
      message: 'VedaVault could not complete the request. Please try again.',
      tone: 'error',
    };
  }
}

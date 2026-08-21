import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { Observable, of, Subject, throwError } from 'rxjs';

import { AnswerRequest, AnswerResponse } from '../../core/models/api.models';
import { VedaVaultClientError } from '../../core/models/client-error.models';
import { VedaVaultConversationService } from '../../core/services/vedavault-conversation.service';
import { ChatPage } from './chat';

class ConversationStub {
  readonly answerResult = new Subject<AnswerResponse>();
  readonly currentSessionId = () => null;
  readonly answer = jasmine
    .createSpy('answer')
    .and.callFake((_request: AnswerRequest): Observable<AnswerResponse> => this.answerResult);
  readonly startNewJourney = jasmine
    .createSpy('startNewJourney')
    .and.returnValue(of(undefined));
}

describe('ChatPage', () => {
  let conversation: ConversationStub;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatPage],
      providers: [
        provideRouter([]),
        { provide: VedaVaultConversationService, useClass: ConversationStub },
      ],
    }).compileComponents();
    conversation = TestBed.inject(VedaVaultConversationService) as unknown as ConversationStub;
  });

  it('uses suggestions to populate the composer without sending', () => {
    const fixture = TestBed.createComponent(ChatPage);
    fixture.detectChanges();

    fixture.componentInstance.chooseSuggestion(fixture.componentInstance.suggestions[2]);

    expect(fixture.componentInstance.draft()).toContain('গীতা');
    expect(fixture.componentInstance.language()).toBe('bn');
    expect(fixture.componentInstance.messages()).toEqual([]);
    expect(conversation.answer).not.toHaveBeenCalled();
  });

  it('maps language and mode, then reflects the real request lifecycle', () => {
    const fixture = TestBed.createComponent(ChatPage);
    fixture.detectChanges();
    fixture.componentInstance.language.set('bn');
    fixture.componentInstance.mode.set('application');
    fixture.componentInstance.draft.set('Explain right action.');

    fixture.componentInstance.sendMessage();
    fixture.detectChanges();

    expect(conversation.answer).toHaveBeenCalledWith({
      query: 'Explain right action.',
      input_language: 'bn',
      response_language: 'bn',
      mode: 'application',
    });
    expect(fixture.componentInstance.thinking()).toBeTrue();
    expect(fixture.nativeElement.querySelector('app-thinking-indicator')).not.toBeNull();

    conversation.answerResult.next(response);
    conversation.answerResult.complete();
    fixture.detectChanges();

    expect(fixture.componentInstance.thinking()).toBeFalse();
    expect(fixture.componentInstance.messages().length).toBe(2);
    expect(fixture.nativeElement.textContent).toContain('Act with steadiness.');
    expect(fixture.nativeElement.textContent).toContain('BG 2.47');
  });

  it('renders a friendly API failure and permits a clean New Journey', () => {
    const fixture = TestBed.createComponent(ChatPage);
    conversation.answer.and.returnValue(
      throwError(
        () =>
          new VedaVaultClientError(
            'rate_limited',
            'Please pause for a moment',
            'VedaVault is receiving many requests right now. Please try again shortly.',
          ),
      ),
    );
    fixture.detectChanges();
    fixture.componentInstance.draft.set('A question');

    fixture.componentInstance.sendMessage();
    fixture.detectChanges();

    expect(fixture.componentInstance.thinking()).toBeFalse();
    expect(fixture.nativeElement.textContent).toContain('Please pause for a moment');

    fixture.componentInstance.startNewJourney();
    fixture.detectChanges();

    expect(conversation.startNewJourney).toHaveBeenCalledTimes(1);
    expect(fixture.componentInstance.messages()).toEqual([]);
    expect(fixture.componentInstance.draft()).toBe('');
  });

  const response: AnswerResponse = {
    session_id: 'backend-session',
    query: 'Explain right action.',
    retrieval_query: 'right action',
    response_language: 'bn',
    mode: 'application',
    evidence_sufficient: true,
    scriptural_teaching: [{ statement: 'Act with steadiness.', cited_verse_ids: ['BG_02_47'] }],
    interpretation: 'Give full care to the action.',
    application: 'Choose the next responsible step.',
    limitations: [],
    cited_verse_ids: ['BG_02_47'],
    retrieved_verse_ids: ['BG_02_47'],
  };
});

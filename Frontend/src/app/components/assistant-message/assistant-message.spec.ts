import { TestBed } from '@angular/core/testing';

import { AssistantChatMessage } from '../../core/models/chat.models';
import { AssistantMessageComponent } from './assistant-message';

describe('AssistantMessageComponent', () => {
  const message: AssistantChatMessage = {
    id: 'assistant-live-test',
    role: 'assistant',
    response: {
      session_id: 'backend-session',
      query: 'What is desire?',
      retrieval_query: 'desire',
      response_language: 'en',
      mode: 'philosophical',
      evidence_sufficient: true,
      scriptural_teaching: [
        { statement: 'Desire can veil discernment.', cited_verse_ids: ['BG_03_40'] },
      ],
      interpretation: 'Notice where urgency narrows attention.',
      application: null,
      limitations: [],
      cited_verse_ids: ['BG_03_40'],
      retrieved_verse_ids: ['BG_03_40'],
    },
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [AssistantMessageComponent] }).compileComponents();
  });

  it('renders real response sections and omits empty optional sections', () => {
    const fixture = TestBed.createComponent(AssistantMessageComponent);
    fixture.componentRef.setInput('message', message);
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Desire can veil discernment.');
    expect(fixture.nativeElement.textContent).toContain('Notice where urgency narrows attention.');
    expect(fixture.nativeElement.textContent).not.toContain('Practical application');
    expect(fixture.nativeElement.querySelector('.limitations')).toBeNull();
  });

  it('expands a minimal real citation card without fabricated scripture text', () => {
    const fixture = TestBed.createComponent(AssistantMessageComponent);
    fixture.componentRef.setInput('message', message);
    fixture.detectChanges();

    const citation = fixture.nativeElement.querySelector('.citation') as HTMLButtonElement;
    expect(citation.textContent).toContain('BG 3.40');
    citation.click();
    fixture.detectChanges();

    const card = fixture.nativeElement.querySelector('app-scripture-card');
    expect(card.textContent).toContain('BG_03_40');
    expect(card.textContent).toContain('not included in the current API response');
    expect(card.textContent).not.toContain('Translation');
    expect(citation.getAttribute('aria-expanded')).toBe('true');

    const collapse = card.querySelector('header button') as HTMLButtonElement;
    collapse.click();
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('app-scripture-card')).toBeNull();
  });
});

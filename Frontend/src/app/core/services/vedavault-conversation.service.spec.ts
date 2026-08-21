import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { VEDAVAULT_API_BASE_URL } from '../config/api.config';
import { AnswerRequest, AnswerResponse } from '../models/api.models';
import { VedaVaultConversationService } from './vedavault-conversation.service';

describe('VedaVaultConversationService', () => {
  const baseUrl = 'http://127.0.0.1:8000';
  const request: AnswerRequest = {
    query: 'Explain right action.',
    input_language: 'bn',
    response_language: 'bn',
    mode: 'application',
  };

  let service: VedaVaultConversationService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: VEDAVAULT_API_BASE_URL, useValue: baseUrl },
      ],
    });
    service = TestBed.inject(VedaVaultConversationService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('omits a first-turn session, stores the returned ID, and reuses it on the follow-up', () => {
    service.answer(request).subscribe();
    const first = http.expectOne(`${baseUrl}/answer`);
    expect(first.request.body).toEqual(request);
    expect(first.request.body.session_id).toBeUndefined();
    expect(first.request.body.input_language).toBe('bn');
    expect(first.request.body.response_language).toBe('bn');
    expect(first.request.body.mode).toBe('application');
    first.flush(responseFor('backend-session-1'));

    expect(service.currentSessionId()).toBe('backend-session-1');

    service.answer({ ...request, query: 'Explain that in Bengali.' }).subscribe();
    const followUp = http.expectOne(`${baseUrl}/answer`);
    expect(followUp.request.body.session_id).toBe('backend-session-1');
    expect(followUp.request.body.response_language).toBe('bn');
    followUp.flush(responseFor('backend-session-1'));
  });

  it('deletes the backend session before clearing its in-memory ID', () => {
    createSession('backend-session-2');
    let completed = false;

    service.startNewJourney().subscribe({ complete: () => (completed = true) });
    expect(service.currentSessionId()).toBe('backend-session-2');
    const deletion = http.expectOne(`${baseUrl}/sessions/backend-session-2`);
    expect(deletion.request.method).toBe('DELETE');
    deletion.flush({ status: 'deleted', session_id: 'backend-session-2' });

    expect(completed).toBeTrue();
    expect(service.currentSessionId()).toBeNull();
  });

  it('still clears the local session when the backend session is already gone', () => {
    createSession('expired-session');
    let completed = false;

    service.startNewJourney().subscribe({ complete: () => (completed = true) });
    http
      .expectOne(`${baseUrl}/sessions/expired-session`)
      .flush({ detail: 'unknown session_id' }, { status: 404, statusText: 'Not Found' });

    expect(completed).toBeTrue();
    expect(service.currentSessionId()).toBeNull();
  });

  function responseFor(sessionId: string): AnswerResponse {
    return {
      session_id: sessionId,
      query: request.query,
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
  }

  function createSession(sessionId: string): void {
    service.answer(request).subscribe();
    http.expectOne(`${baseUrl}/answer`).flush(responseFor(sessionId));
  }
});

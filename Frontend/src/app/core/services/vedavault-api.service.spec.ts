import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { VEDAVAULT_API_BASE_URL } from '../config/api.config';
import { AnswerRequest, AnswerResponse } from '../models/api.models';
import { VedaVaultClientError } from '../models/client-error.models';
import { VedaVaultApiService } from './vedavault-api.service';

describe('VedaVaultApiService', () => {
  const baseUrl = 'http://127.0.0.1:8000';
  const request: AnswerRequest = {
    query: 'What does the Gita say about desire?',
    input_language: 'en',
    response_language: 'en',
    mode: 'philosophical',
  };
  const response: AnswerResponse = {
    session_id: 'session-live-1',
    query: request.query,
    retrieval_query: 'desire discernment',
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
    retrieved_verse_ids: ['BG_03_40', 'BG_03_37'],
  };

  let service: VedaVaultApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: VEDAVAULT_API_BASE_URL, useValue: baseUrl },
      ],
    });
    service = TestBed.inject(VedaVaultApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('posts the exact API request without secrets or authorization headers', () => {
    let actual: AnswerResponse | undefined;
    service.answer(request).subscribe((value) => (actual = value));

    const pending = http.expectOne(`${baseUrl}/answer`);
    expect(pending.request.method).toBe('POST');
    expect(pending.request.body).toEqual(request);
    expect(pending.request.headers.has('Authorization')).toBeFalse();
    pending.flush(response);

    expect(actual).toEqual(response);
  });

  it('uses the backend safe message for a clarification response', () => {
    let actual: VedaVaultClientError | undefined;
    service.answer(request).subscribe({ error: (error: VedaVaultClientError) => (actual = error) });

    http.expectOne(`${baseUrl}/answer`).flush(
      {
        error: 'clarification_required',
        message: 'Please clarify which kind of desire you mean.',
        query: request.query,
        clarification_required: true,
      },
      { status: 409, statusText: 'Conflict' },
    );

    expect(actual?.kind).toBe('clarification');
    expect(actual?.tone).toBe('clarification');
    expect(actual?.userMessage).toBe('Please clarify which kind of desire you mean.');
  });

  it('maps rate limiting without exposing an upstream payload', () => {
    let actual: VedaVaultClientError | undefined;
    service.answer(request).subscribe({ error: (error: VedaVaultClientError) => (actual = error) });

    http.expectOne(`${baseUrl}/answer`).flush(
      { message: 'provider-internal-detail-should-not-leak' },
      { status: 429, statusText: 'Too Many Requests' },
    );

    expect(actual?.kind).toBe('rate_limited');
    expect(actual?.userMessage).toBe(
      'VedaVault is receiving many requests right now. Please try again shortly.',
    );
    expect(actual?.userMessage).not.toContain('provider-internal-detail');
  });

  it('maps service and network failures to calm actionable messages', () => {
    const failures: VedaVaultClientError[] = [];

    service.answer(request).subscribe({ error: (error: VedaVaultClientError) => failures.push(error) });
    http
      .expectOne(`${baseUrl}/answer`)
      .flush({}, { status: 503, statusText: 'Service Unavailable' });

    service.answer(request).subscribe({ error: (error: VedaVaultClientError) => failures.push(error) });
    http.expectOne(`${baseUrl}/answer`).error(new ProgressEvent('error'));

    expect(failures[0].kind).toBe('service_unavailable');
    expect(failures[0].userMessage).toContain('knowledge service is temporarily unavailable');
    expect(failures[1].kind).toBe('network');
    expect(failures[1].userMessage).toContain('backend is running');
  });
});

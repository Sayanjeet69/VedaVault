import { inject, Injectable, signal } from '@angular/core';
import { catchError, finalize, map, Observable, of, tap } from 'rxjs';

import { AnswerRequest, AnswerResponse } from '../models/api.models';
import { VedaVaultApiService } from './vedavault-api.service';

@Injectable({ providedIn: 'root' })
export class VedaVaultConversationService {
  private readonly api = inject(VedaVaultApiService);
  private readonly sessionId = signal<string | null>(null);

  readonly currentSessionId = this.sessionId.asReadonly();

  answer(request: AnswerRequest): Observable<AnswerResponse> {
    const currentSessionId = this.sessionId();
    const payload: AnswerRequest = currentSessionId
      ? { ...request, session_id: currentSessionId }
      : this.withoutSessionId(request);

    return this.api.answer(payload).pipe(tap((response) => this.sessionId.set(response.session_id)));
  }

  startNewJourney(): Observable<void> {
    const sessionId = this.sessionId();
    if (!sessionId) {
      return of(undefined);
    }

    return this.api.deleteSession(sessionId).pipe(
      map(() => undefined),
      // A fresh local journey must remain possible if the remote session is already gone
      // or cannot be reached. The process-local backend session will expire independently.
      catchError(() => of(undefined)),
      finalize(() => {
        if (this.sessionId() === sessionId) {
          this.sessionId.set(null);
        }
      }),
    );
  }

  private withoutSessionId(request: AnswerRequest): AnswerRequest {
    const { session_id: _sessionId, ...payload } = request;
    return payload;
  }
}

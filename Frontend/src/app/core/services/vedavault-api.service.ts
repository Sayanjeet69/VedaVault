import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { catchError, Observable, throwError } from 'rxjs';

import { VEDAVAULT_API_BASE_URL } from '../config/api.config';
import { AnswerRequest, AnswerResponse, DeleteSessionResponse, HealthResponse } from '../models/api.models';
import { VedaVaultClientError } from '../models/client-error.models';

@Injectable({ providedIn: 'root' })
export class VedaVaultApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(VEDAVAULT_API_BASE_URL).replace(/\/$/, '');

  answer(request: AnswerRequest): Observable<AnswerResponse> {
    return this.http
      .post<AnswerResponse>(`${this.baseUrl}/answer`, request)
      .pipe(catchError((error: unknown) => throwError(() => this.toClientError(error))));
  }

  deleteSession(sessionId: string): Observable<DeleteSessionResponse> {
    return this.http
      .delete<DeleteSessionResponse>(`${this.baseUrl}/sessions/${encodeURIComponent(sessionId)}`)
      .pipe(catchError((error: unknown) => throwError(() => this.toClientError(error))));
  }

  health(): Observable<HealthResponse> {
    return this.http
      .get<HealthResponse>(`${this.baseUrl}/health`)
      .pipe(catchError((error: unknown) => throwError(() => this.toClientError(error))));
  }

  private toClientError(error: unknown): VedaVaultClientError {
    if (!(error instanceof HttpErrorResponse)) {
      return new VedaVaultClientError(
        'unexpected',
        'Something interrupted the journey',
        'VedaVault could not complete the request. Please try again.',
      );
    }

    if (error.status === 0) {
      return new VedaVaultClientError(
        'network',
        'Unable to reach VedaVault',
        'Check that the backend is running and try again.',
      );
    }

    if (error.status === 409) {
      return new VedaVaultClientError(
        'clarification',
        'A little more clarity is needed',
        this.safeBackendMessage(error.error) ??
          'Please add a little more detail so VedaVault can retrieve the right scripture.',
        'clarification',
      );
    }

    if (error.status === 429) {
      return new VedaVaultClientError(
        'rate_limited',
        'Please pause for a moment',
        'VedaVault is receiving many requests right now. Please try again shortly.',
      );
    }

    if (error.status === 502) {
      return new VedaVaultClientError(
        'upstream_unavailable',
        'The response could not be completed',
        'VedaVault\'s generation service is temporarily unavailable. Please try again in a moment.',
      );
    }

    if (error.status === 503) {
      return new VedaVaultClientError(
        'service_unavailable',
        'Knowledge service unavailable',
        'VedaVault\'s knowledge service is temporarily unavailable. Please try again in a moment.',
      );
    }

    if (error.status === 404) {
      return new VedaVaultClientError(
        'session_missing',
        'This journey is no longer available',
        'Start a new journey and ask your question again.',
      );
    }

    if (error.status === 400 || error.status === 422) {
      return new VedaVaultClientError(
        'invalid_request',
        'Please review the question',
        'VedaVault could not understand this request. Check the question and try again.',
      );
    }

    return new VedaVaultClientError(
      'unexpected',
      'Something interrupted the journey',
      'VedaVault could not complete the request. Please try again.',
    );
  }

  private safeBackendMessage(payload: unknown): string | null {
    if (typeof payload !== 'object' || payload === null || !('message' in payload)) {
      return null;
    }
    const message = payload.message;
    return typeof message === 'string' && message.trim() ? message.trim() : null;
  }
}

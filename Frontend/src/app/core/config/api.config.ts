import { InjectionToken } from '@angular/core';

/**
 * Override this token at bootstrap time when the API is deployed elsewhere.
 * Keeping the development URL here prevents transport details leaking into UI components.
 */
export const VEDAVAULT_API_BASE_URL = new InjectionToken<string>('VEDAVAULT_API_BASE_URL', {
  providedIn: 'root',
  factory: () => 'http://127.0.0.1:8000',
});

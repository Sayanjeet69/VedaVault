export type VedaVaultClientErrorKind =
  | 'clarification'
  | 'invalid_request'
  | 'session_missing'
  | 'rate_limited'
  | 'upstream_unavailable'
  | 'service_unavailable'
  | 'network'
  | 'unexpected';

export type ErrorNoticeTone = 'error' | 'clarification';

export class VedaVaultClientError extends Error {
  constructor(
    readonly kind: VedaVaultClientErrorKind,
    readonly title: string,
    readonly userMessage: string,
    readonly tone: ErrorNoticeTone = 'error',
  ) {
    super(userMessage);
    this.name = 'VedaVaultClientError';
  }
}

export interface ChatErrorState {
  title: string;
  message: string;
  tone: ErrorNoticeTone;
}

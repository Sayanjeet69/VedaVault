export type V1LanguageCode = 'en' | 'hi' | 'bn' | 'sa';

export type AnswerMode = 'textual' | 'philosophical' | 'application';

export interface AnswerRequest {
  query: string;
  input_language?: V1LanguageCode | null;
  response_language?: V1LanguageCode | null;
  mode?: AnswerMode;
  session_id?: string | null;
}

export interface ScripturalTeaching {
  statement: string;
  cited_verse_ids: string[];
}

export interface AnswerResponse {
  session_id: string;
  query: string;
  retrieval_query: string;
  response_language: V1LanguageCode;
  mode: AnswerMode;
  evidence_sufficient: boolean;
  scriptural_teaching: ScripturalTeaching[];
  interpretation: string | null;
  application: string | null;
  limitations: string[];
  cited_verse_ids: string[];
  retrieved_verse_ids: string[];
}

export interface HealthResponse {
  status: 'ok';
  service: 'vedavault';
  rag_version: 'v1';
}

export interface DeleteSessionResponse {
  status: 'deleted';
  session_id: string;
}

export type VedaVaultErrorCode =
  | 'clarification_required'
  | 'evidence_unavailable'
  | 'upstream_rate_limited'
  | 'upstream_service_error'
  | 'service_unavailable'
  | 'internal_error';

export interface ApiErrorResponse {
  error: VedaVaultErrorCode;
  message: string;
}

export interface ClarificationRequiredResponse extends ApiErrorResponse {
  error: 'clarification_required';
  query: string;
  clarification_required: true;
}

export interface HttpDetailError {
  detail: string;
}

export interface ValidationIssue {
  loc: Array<string | number>;
  msg: string;
  type: string;
  input?: unknown;
  ctx?: Record<string, unknown>;
}

export interface ValidationErrorResponse {
  detail: ValidationIssue[];
}

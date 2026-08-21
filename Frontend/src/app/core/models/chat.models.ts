import { AnswerMode, AnswerResponse, V1LanguageCode } from './api.models';

export interface SelectOption<T extends string> {
  value: T;
  label: string;
  description: string;
}

export const LANGUAGE_OPTIONS: readonly SelectOption<V1LanguageCode>[] = [
  { value: 'en', label: 'English', description: 'English' },
  { value: 'hi', label: 'हिन्दी', description: 'Hindi' },
  { value: 'bn', label: 'বাংলা', description: 'Bengali' },
  { value: 'sa', label: 'संस्कृत', description: 'Sanskrit' },
];

export const MODE_OPTIONS: readonly SelectOption<AnswerMode>[] = [
  { value: 'textual', label: 'Textual', description: 'Close to retrieved scripture' },
  { value: 'philosophical', label: 'Philosophical', description: 'Teaching + interpretation' },
  { value: 'application', label: 'Application', description: 'Practical guidance' },
];

export interface UserChatMessage {
  id: string;
  role: 'user';
  content: string;
  language: V1LanguageCode;
}

export interface AssistantChatMessage {
  id: string;
  role: 'assistant';
  response: AnswerResponse;
}

export type ChatMessage = UserChatMessage | AssistantChatMessage;

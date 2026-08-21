import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () =>
      import('./pages/welcome/welcome').then((component) => component.WelcomePage),
    title: 'VedaVault — Ancient wisdom, modern clarity',
  },
  {
    path: 'chat',
    loadComponent: () => import('./pages/chat/chat').then((component) => component.ChatPage),
    title: 'A new journey — VedaVault',
  },
  { path: '**', redirectTo: '' },
];

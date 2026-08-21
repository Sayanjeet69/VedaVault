import { Component, input } from '@angular/core';

import { UserChatMessage } from '../../core/models/chat.models';

@Component({
  selector: 'app-user-message',
  template: `
    <article class="user-row" aria-label="Your message">
      <div class="user-message">
        <span class="accent" aria-hidden="true"></span>
        <p>{{ message().content }}</p>
      </div>
    </article>
  `,
  styles: `
    :host { display: block; width: min(100%, var(--conversation-width)); margin: 0 auto; }
    .user-row { display: flex; justify-content: flex-end; animation: message-in 260ms var(--ease-out) both; }
    .user-message { position: relative; max-width: min(78%, 37rem); overflow: hidden; padding: .85rem 1rem; border: 1px solid var(--border-subtle); border-radius: 1rem 1rem .35rem 1rem; background: linear-gradient(145deg, #191919, #141414); box-shadow: 0 10px 30px rgba(0,0,0,.18); }
    .accent { position: absolute; top: .65rem; right: 0; bottom: .65rem; width: 2px; border-radius: 2px 0 0 2px; background: linear-gradient(transparent, var(--saffron-muted), transparent); opacity: .7; }
    p { margin: 0; overflow-wrap: anywhere; color: #e7e5e2; font-size: .9rem; line-height: 1.65; white-space: pre-wrap; }
    @keyframes message-in { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
    @media (max-width: 600px) { .user-message { max-width: 88%; } }
  `,
})
export class UserMessageComponent {
  readonly message = input.required<UserChatMessage>();
}

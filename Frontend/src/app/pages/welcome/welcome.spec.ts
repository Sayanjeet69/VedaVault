import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { routes } from '../../app.routes';
import { WelcomePage } from './welcome';

describe('WelcomePage', () => {
  it('keeps the premium welcome CTA pointed at the chat route', async () => {
    await TestBed.configureTestingModule({
      imports: [WelcomePage],
      providers: [provideRouter(routes)],
    }).compileComponents();
    const fixture = TestBed.createComponent(WelcomePage);
    fixture.detectChanges();

    const journeyLink = fixture.nativeElement.querySelector('.journey-cta') as HTMLAnchorElement;
    expect(journeyLink.textContent).toContain('Begin your journey');
    expect(journeyLink.getAttribute('href')).toBe('/chat');
  });
});

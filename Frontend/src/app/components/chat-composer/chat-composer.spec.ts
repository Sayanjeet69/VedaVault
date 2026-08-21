import { TestBed } from '@angular/core/testing';

import { ChatComposerComponent } from './chat-composer';

describe('ChatComposerComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [ChatComposerComponent] }).compileComponents();
  });

  it('keeps send disabled for blank input', () => {
    const fixture = TestBed.createComponent(ChatComposerComponent);
    fixture.componentRef.setInput('language', 'en');
    fixture.componentRef.setInput('mode', 'textual');
    fixture.componentRef.setInput('value', '   ');
    fixture.detectChanges();

    const sendButton = fixture.nativeElement.querySelector('button[type="submit"]') as HTMLButtonElement;
    expect(sendButton.disabled).toBeTrue();
  });

  it('emits send for a non-empty submitted message', () => {
    const fixture = TestBed.createComponent(ChatComposerComponent);
    fixture.componentRef.setInput('language', 'en');
    fixture.componentRef.setInput('mode', 'philosophical');
    fixture.componentRef.setInput('value', 'What is right action?');
    const emitted = jasmine.createSpy('send');
    fixture.componentInstance.send.subscribe(emitted);
    fixture.detectChanges();

    fixture.componentInstance.submit(new SubmitEvent('submit'));

    expect(emitted).toHaveBeenCalledTimes(1);
  });

  it('uses Enter as a desktop shortcut but preserves normal mobile input', () => {
    const fixture = TestBed.createComponent(ChatComposerComponent);
    fixture.componentRef.setInput('language', 'en');
    fixture.componentRef.setInput('mode', 'philosophical');
    fixture.componentRef.setInput('value', 'A meaningful question');
    const emitted = jasmine.createSpy('send');
    fixture.componentInstance.send.subscribe(emitted);
    fixture.detectChanges();
    const media = spyOn(window, 'matchMedia');
    const preventDefault = jasmine.createSpy('preventDefault');
    const enterEvent = {
      key: 'Enter',
      shiftKey: false,
      isComposing: false,
      preventDefault,
    } as unknown as KeyboardEvent;

    media.and.returnValue({ matches: true } as MediaQueryList);
    fixture.componentInstance.onKeydown(enterEvent);
    expect(emitted).toHaveBeenCalledTimes(1);
    expect(preventDefault).toHaveBeenCalledTimes(1);

    media.and.returnValue({ matches: false } as MediaQueryList);
    fixture.componentInstance.onKeydown(enterEvent);
    expect(emitted).toHaveBeenCalledTimes(1);
  });
});

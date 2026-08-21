import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

import { BrandMarkComponent } from '../../components/brand-mark/brand-mark';

@Component({
  selector: 'app-welcome-page',
  imports: [RouterLink, BrandMarkComponent],
  templateUrl: './welcome.html',
  styleUrl: './welcome.css',
  host: { class: 'page-host' },
})
export class WelcomePage {}

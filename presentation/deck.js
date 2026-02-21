/**
 * deck.js — Shared navigation utilities for the presentation selector.
 * Each deck (deck1.html, deck2.html, deck3.html) is fully self-contained
 * and does not depend on this file. This script enhances index.html only.
 */

(function () {
  'use strict';

  // Animate cards in on load with stagger
  function initCardStagger() {
    const cards = document.querySelectorAll('.card');
    cards.forEach(function (card, i) {
      card.style.opacity = '0';
      card.style.transform = 'translateY(24px)';
      card.style.transition = 'opacity .55s cubic-bezier(.16,1,.3,1), transform .55s cubic-bezier(.16,1,.3,1), box-shadow .3s ease, border-color .3s ease';
      setTimeout(function () {
        card.style.opacity = '1';
        card.style.transform = 'translateY(0)';
      }, 120 + i * 100);
    });
  }

  // Animate hero on load
  function initHeroFade() {
    var hero = document.querySelector('.hero');
    if (!hero) return;
    hero.style.opacity = '0';
    hero.style.transform = 'translateY(16px)';
    hero.style.transition = 'opacity .7s ease, transform .7s cubic-bezier(.16,1,.3,1)';
    setTimeout(function () {
      hero.style.opacity = '1';
      hero.style.transform = 'none';
    }, 60);
  }

  // Keyboard shortcut: 1/2/3 to open each deck directly
  function initKeyboardShortcuts() {
    document.addEventListener('keydown', function (e) {
      if (e.key === '1') { window.open('deck1.html', '_blank'); }
      if (e.key === '2') { window.open('deck2.html', '_blank'); }
      if (e.key === '3') { window.open('deck3.html', '_blank'); }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    initHeroFade();
    initCardStagger();
    initKeyboardShortcuts();
  });
})();

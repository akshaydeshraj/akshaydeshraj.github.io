(function() {
  'use strict';

  function getPreferredTheme() {
    var saved = localStorage.getItem('theme');
    if (saved) {
      return saved;
    }
    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  }

  function setTheme(theme) {
    var html = document.documentElement;
    html.classList.remove('theme-light', 'theme-dark');
    if (theme === 'dark') {
      html.classList.add('theme-dark');
    }
    // Light theme uses default :root styles, no class needed
    localStorage.setItem('theme', theme);
  }

  function toggleTheme() {
    var current = document.documentElement.classList.contains('theme-dark') ? 'dark' : 'light';
    var next = current === 'dark' ? 'light' : 'dark';
    setTheme(next);
  }

  // Initialize toggle button when DOM is ready
  document.addEventListener('DOMContentLoaded', function() {
    var toggleBtn = document.querySelector('.theme-toggle');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', toggleTheme);
    }
  });

  // Listen for system theme changes
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
      // Only auto-switch if user hasn't set a preference
      if (!localStorage.getItem('theme')) {
        setTheme(e.matches ? 'dark' : 'light');
      }
    });
  }
})();

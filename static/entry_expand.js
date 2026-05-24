/**
 * Toggle full entry text on labeling / top cards (delegated — works after DOM updates).
 * Supports Seismo 0.6 (.entry-preview / .entry-full-content) and legacy blocks.
 */
(function () {
    function collapse(card, btn) {
        var preview = card.querySelector('.entry-preview');
        var full = card.querySelector('.entry-full-content');
        if (preview && full) {
            full.style.display = 'none';
            preview.style.display = '';
            if (btn) {
                btn.setAttribute('aria-expanded', 'false');
                btn.textContent = 'expand \u25BC';
            }
            return true;
        }
        var block = card.querySelector('.entry-text-block');
        if (!block) return false;
        var collapsed = block.querySelector('.entry-text-collapsed');
        var legacyFull = block.querySelector('.entry-text-full');
        if (!collapsed || !legacyFull) return false;
        block.classList.remove('is-expanded');
        collapsed.hidden = false;
        legacyFull.hidden = true;
        if (btn) {
            btn.setAttribute('aria-expanded', 'false');
            btn.textContent = 'Show full text';
        }
        return true;
    }

    function expand(card, btn) {
        var preview = card.querySelector('.entry-preview');
        var full = card.querySelector('.entry-full-content');
        if (preview && full) {
            preview.style.display = 'none';
            full.style.display = 'block';
            if (btn) {
                btn.setAttribute('aria-expanded', 'true');
                btn.textContent = 'collapse \u25B2';
            }
            return true;
        }
        var block = card.querySelector('.entry-text-block');
        if (!block) return false;
        var collapsed = block.querySelector('.entry-text-collapsed');
        var legacyFull = block.querySelector('.entry-text-full');
        if (!collapsed || !legacyFull) return false;
        block.classList.add('is-expanded');
        collapsed.hidden = true;
        legacyFull.hidden = false;
        if (btn) {
            btn.setAttribute('aria-expanded', 'true');
            btn.textContent = 'Show less';
        }
        return true;
    }

    document.addEventListener('click', function (e) {
        var btn = e.target.closest('.entry-expand-btn');
        if (!btn) return;
        e.preventDefault();
        var card = btn.closest('.entry-card');
        if (!card) return;
        var full = card.querySelector('.entry-full-content');
        var legacyFull = card.querySelector('.entry-text-full');
        if (full || legacyFull) {
            if (full && full.style.display === 'block') {
                collapse(card, btn);
            } else if (legacyFull && !legacyFull.hidden) {
                collapse(card, btn);
            } else {
                expand(card, btn);
            }
        }
    });
})();

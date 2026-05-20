/**
 * Toggle full entry text on labeling / top cards (delegated — works after DOM updates).
 */
(function () {
    document.addEventListener('click', function (e) {
        var btn = e.target.closest('.entry-expand-btn');
        if (!btn) return;
        e.preventDefault();
        var block = btn.closest('.entry-text-block');
        if (!block) return;
        var collapsed = block.querySelector('.entry-text-collapsed');
        var full = block.querySelector('.entry-text-full');
        if (!collapsed || !full) return;
        var expanded = block.classList.toggle('is-expanded');
        collapsed.hidden = expanded;
        full.hidden = !expanded;
        btn.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        btn.textContent = expanded ? 'Show less' : 'Show full text';
    });
})();

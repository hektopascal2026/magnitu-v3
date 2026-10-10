from unittest.mock import MagicMock

import sync


def test_recipe_push_leaves_version_ownership_to_seismo_without_mutating_export(monkeypatch):
    request = MagicMock()
    request.return_value.json.return_value = {'success': True, 'recipe_version': 99}
    monkeypatch.setattr(sync, '_request', request)
    monkeypatch.setattr(sync, '_profile_target', lambda profile: {'url': 'desk'})
    log = MagicMock()
    monkeypatch.setattr(sync.db, 'log_sync', log)
    recipe = {'version': 1, 'keywords': {'test': 2.0}, 'description': 'recipe'}
    result = sync.push_recipe(recipe, {'id': 7})
    assert result == {'success': True, 'recipe_version': 99}
    payload = request.call_args.kwargs['json']
    assert payload == {'keywords': {'test': 2.0}, 'description': 'recipe'}
    assert recipe['version'] == 1
    assert request.call_args.kwargs['seismo_target'] == {'url': 'desk'}
    assert log.call_args.kwargs['profile_id'] == 7
    assert log.call_args.args[2] == "recipe v99 pushed"

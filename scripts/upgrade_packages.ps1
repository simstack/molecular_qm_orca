date
uv lock --upgrade-package molecular_qm_models
uv lock --upgrade-package molecular_qm_simstack
uv sync --locked
git add uv.lock
git commit -m  "chore: update packages"
git push
echo "done"

date
uv lock --upgrade-package molecular_qm_models
uv lock --upgrade-package molecular_qm_util
uv lock --upgrade-package molecular_qm_simstack
uv lock --upgrade-package molecular_qm_turbomole
uv lock --upgrade-package molecular_qm_orca
uv lock --upgrade-package molecular_qm_dftb
uv sync --locked
git add uv.lock
git commit -m  "chore: update packages"
git push
echo "done"

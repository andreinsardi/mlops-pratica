# Push final para o GitHub

## ✅ Já feito por mim (no navegador)

- Repo criado: **https://github.com/andreinsardi/mlops-pratica** (público)
- Commit inicial: placeholder README na branch `main`

## ⚠️ O que falta (precisa do seu terminal — 30 segundos)

A camada de segurança do Chrome MCP não me deixa enviar arquivos da sua pasta Cowork direto pro upload web do GitHub.
A forma mais rápida é você rodar **5 comandos no PowerShell**:

```powershell
cd "C:\Users\andre\odrive\Google Drive\fgv\FGV\mlops-pratica"
git init -b main
git add -A
git commit -m "Initial commit - MLOps Pratica end-to-end"
git remote add origin https://github.com/andreinsardi/mlops-pratica.git
git push -u origin main --force
```

> **Por que `--force`?** O repo tem um commit placeholder que será sobrescrito pelo seu commit real (com os 78 arquivos completos).

## Autenticação

Na primeira execução do `git push`, o Git vai pedir credenciais. Você tem 3 opções:

1. **GitHub Desktop instalado** → ele já configura o credential helper. Só roda os comandos acima.
2. **GitHub CLI (`gh`)** → `gh auth login --web` antes do push (auth via browser).
3. **Personal Access Token** → gere em https://github.com/settings/tokens (escopo `repo`), use no prompt de senha do git.

## Verificação pós-push

Depois do push, acesse https://github.com/andreinsardi/mlops-pratica e confirme:

- [ ] README.md aparece com badges
- [ ] Pasta `src/mlops_pratica/` com módulos Python
- [ ] Pasta `dags/` com 3 DAGs Airflow
- [ ] Pasta `infra/` com Dockerfiles e scripts GCP
- [ ] `docs/img/` com 8 PNGs (diagramas)
- [ ] `Apresentacao_MLOps_Pratica.pptx` (38 slides padrão FGV)
- [ ] Tab **Actions** mostra o CI rodando

## About do repo (sugestão)

Em https://github.com/andreinsardi/mlops-pratica → engrenagem ⚙ no canto direito:

- **Description:** `Prática MLOps end-to-end com Airflow + MLflow + pgvector + FastAPI. Material da disciplina MLOps - MBA FGV.`
- **Website:** (deixar vazio)
- **Topics:** `mlops`, `airflow`, `mlflow`, `pgvector`, `fastapi`, `docker`, `python`, `fgv`, `hackernews`, `tutorial`

# run-local.ps1 - sobe a stack MLOps Pratica localmente (Docker Compose)
# Uso:
#   .\run-local.ps1
#   .\run-local.ps1 -Logs
#   .\run-local.ps1 -Down
#   .\run-local.ps1 -Rebuild

param(
    [switch]$Logs,
    [switch]$Down,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Test-DockerAvailable {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker nao encontrado. Instale Docker Desktop e tente novamente."
    }

    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker nao esta rodando. Inicie o Docker Desktop e tente novamente."
    }
}

function Ensure-EnvFile {
    if (-not (Test-Path ".env")) {
        if (-not (Test-Path ".env.example")) {
            throw "Arquivo .env.example nao encontrado."
        }

        Copy-Item ".env.example" ".env"
        Write-Host ".env criado a partir de .env.example"
    }
}

function Show-Urls {
    Write-Host ""
    Write-Host "Stack pronta. Acesse:"
    Write-Host "  Airflow:  http://localhost:8080  (admin/admin)"
    Write-Host "  MLflow:   http://localhost:5000"
    Write-Host "  MinIO:    http://localhost:9001  (minioadmin/minioadmin)"
    Write-Host "  FastAPI:  http://localhost:8000/docs"
    Write-Host ""
    Write-Host "Proximo passo: despause e dispare as DAGs no Airflow."
}

Test-DockerAvailable
Ensure-EnvFile

if ($Down) {
    docker compose down
    Write-Host "Stack encerrada."
    exit 0
}

$composeArgs = @("compose", "up", "-d")

if ($Rebuild) {
    $composeArgs += "--build"
}

Write-Host "Subindo stack MLOps Pratica..."
& docker @composeArgs

if ($LASTEXITCODE -ne 0) {
    throw "Falha ao subir a stack com docker compose."
}

docker compose ps
Show-Urls

if ($Logs) {
    docker compose logs -f --tail=200
}

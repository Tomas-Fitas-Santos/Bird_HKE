# Read-only export of Nightjar UUID -> source relative path mapping.
# Execute from the Supervisely-Monitoring-App repository root or use -NightjarRepo.
[CmdletBinding()]
param(
    [string]$NightjarRepo = (Get-Location).Path,
    [string]$Output = "E:\Annotatiathon 2026\Nightjar_media_asset_map.csv"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $NightjarRepo).Path
$composeFile = Join-Path $root "compose.live.yaml"
$envFile = Join-Path $root ".env.live"
if (-not (Test-Path -LiteralPath $composeFile)) {
    throw "Monitoring App compose.live.yaml not found: $composeFile"
}
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Monitoring App .env.live not found: $envFile. Use the ORIGINAL live deployment (do not create a fresh DB)."
}
$envText = Get-Content -LiteralPath $envFile -Raw
$userMatch = [regex]::Match($envText, '(?m)^POSTGRES_USER=([^\r\n]+)')
$dbMatch = [regex]::Match($envText, '(?m)^POSTGRES_DB=([^\r\n]+)')
if (-not $userMatch.Success -or -not $dbMatch.Success) {
    throw "POSTGRES_USER or POSTGRES_DB is missing from .env.live"
}
$dbUser = $userMatch.Groups[1].Value.Trim().Trim('"')
$dbName = $dbMatch.Groups[1].Value.Trim().Trim('"')
if (-not $dbUser -or -not $dbName) {
    throw "Missing PostgreSQL user or database name"
}
# Only the two source Supervisely projects are exported.
$sql = @'
COPY (
    SELECT
        a.id AS asset_id,
        d.native_project_id AS native_project_id,
        a.relative_path,
        a.kind,
        a.size_bytes,
        a.sha256,
        a.frame_width,
        a.frame_height,
        a.video_group,
        a.frame_index
    FROM media_assets AS a
    JOIN media_datasets AS d ON d.id = a.dataset_id
    WHERE d.native_project_id IN (386759, 386817)
    ORDER BY d.native_project_id, a.id
) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)
'@
$outPath = [System.IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Path (Split-Path -Parent $outPath) -Force | Out-Null
$temp = "$outPath.partial"
if (Test-Path -LiteralPath $temp) {
    Remove-Item -LiteralPath $temp -Force
}
Push-Location $root
try {
    # Avoid passing 100k+ Unicode paths through the Windows PowerShell
    # text pipeline. Write UTF-8 CSV inside the existing PostgreSQL container
    # and copy it byte-for-byte to the Windows destination.
    $containerId = (& docker compose --env-file $envFile -f $composeFile ps -q db | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or -not $containerId) {
        throw "Nightjar live DB container is not running. Use the original Nightjar deployment."
    }
    $remote = "/tmp/nightjar-media-asset-map-export.csv"
    try {
        & docker exec $containerId psql -X -q -v ON_ERROR_STOP=1 -U $dbUser -d $dbName -o $remote -c $sql
        if ($LASTEXITCODE -ne 0) {
            throw "Read-only Nightjar database export failed."
        }
        $copySource = "{0}:{1}" -f $containerId, $remote
        & docker cp $copySource $temp
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to copy the read-only media CSV from PostgreSQL container."
        }
    }
    finally {
        & docker exec $containerId rm -f $remote | Out-Null
    }
}
finally {
    Pop-Location
}
$first = @(Get-Content -LiteralPath $temp -TotalCount 2)
if ($first.Count -lt 2) {
    throw "No Nightjar media assets found for Supervisely project IDs 386759 or 386817. Existing output was not replaced."
}
if ($first[0] -notmatch 'asset_id' -or $first[0] -notmatch 'native_project_id') {
    throw "Database output did not contain expected CSV columns. Existing output was not replaced."
}
Move-Item -LiteralPath $temp -Destination $outPath -Force
Write-Host "Read-only UUID-to-original-media mapping exported:"
Write-Host $outPath
Write-Host "Now run the BirdGaze builder with --audit-only."
